use clap::Parser;
use ndarray::Array1;
use ndarray_npy::write_npy;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::dataset::ScoredVector;
use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    Dataset, DatasetGrowable, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
    SpaceUsage, SparseDatasetGrowable, SparseVectorView, VariableBitUniformSparseQuantizer,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Benchmark hybrid variable-bit quantization: top-X% of document components use high bits, rest use low bits"
)]
struct Args {
    /// Sparse dataset in Seismic binary format
    #[clap(short, long)]
    input_file: String,

    /// Sparse queries in Seismic binary format
    #[clap(short, long)]
    query_file: String,

    /// Number of top results
    #[clap(short, long, default_value_t = 10)]
    k: usize,

    /// Number of queries to use
    #[clap(long, default_value_t = 1000)]
    n_queries: usize,

    /// Output TSV file path
    #[clap(short, long, default_value = "hybrid_quantization_results.tsv")]
    output: String,

    /// Directory where per-query recall .npy files are written
    #[clap(long, default_value = ".")]
    recall_dir: String,

    /// Lower percentile for quantizer training
    #[clap(long, default_value_t = 0.0)]
    lower_percentile: f32,

    /// Upper percentile for quantizer training
    #[clap(long, default_value_t = 1.0)]
    upper_percentile: f32,
}

/// Returns (mean_recall, per_query_recall).
fn recall_at_k(
    gt: &[Vec<ScoredVector<DotProduct>>],
    approx: &[Vec<ScoredVector<DotProduct>>],
    k: usize,
) -> (f64, Vec<f32>) {
    let per_query: Vec<f32> = gt
        .iter()
        .zip(approx.iter())
        .map(|(gt_row, approx_row)| {
            let gt_ids: std::collections::HashSet<u64> =
                gt_row.iter().take(k).map(|s| s.vector).collect();
            let found = approx_row
                .iter()
                .take(k)
                .filter(|s| gt_ids.contains(&s.vector))
                .count();
            found as f32 / k as f32
        })
        .collect();
    let mean = per_query.iter().map(|&r| r as f64).sum::<f64>() / per_query.len() as f64;
    (mean, per_query)
}

/// Quantize a value and return the dequantized reconstruction.
#[inline]
fn dequantize(value: f32, quant_step: f32, max_code: f32) -> f32 {
    if quant_step <= 0.0 {
        return 0.0;
    }
    let code = (value / quant_step).clamp(0.0, max_code) as u8;
    code as f32 * quant_step
}

/// Compute component importance scores (average query_val × doc_val across all queries).
/// Returns a vec where index is component ID and value is the total importance.
fn compute_component_importance(
    dataset: &PlainSparseDataset<u16, f32, DotProduct>,
    queries: &PlainSparseDataset<u16, f32, DotProduct>,
    n_queries: usize,
) -> Vec<f32> {
    let dim = dataset.input_dim();
    let mut importance: Vec<f32> = vec![0.0f32; dim];
    let total_n_queries = queries.len().min(n_queries);

    for qi in 0..total_n_queries {
        let query_view = queries.get(qi as u64);
        let q_components = query_view.components();
        let q_values = query_view.values();

        // Build a map of component -> query_value for this query.
        let mut q_map = HashMap::new();
        for (&c, &v) in q_components.iter().zip(q_values.iter()) {
            q_map.insert(c, v);
        }

        // For each document, accumulate importance for intersecting components.
        for doc_view in dataset.iter() {
            let doc_components = doc_view.components();
            let doc_values = doc_view.values();

            for (&c, &dv) in doc_components.iter().zip(doc_values.iter()) {
                if let Some(&qv) = q_map.get(&c) {
                    let idx = c as usize;
                    importance[idx] += qv * dv;
                }
            }
        }
    }

    importance
}

/// Build a PlainSparseDataset<u16, f32, DotProduct> where each document's values
/// are replaced by their hybrid-dequantized reconstructions.
///
/// Selection mode:
/// - If importance_scores is Some: top pct% of components (by importance score) per doc use high bits
/// - If importance_scores is None: top pct% of components (by value) per doc use high bits
fn build_hybrid_dataset(
    dataset_f32: &PlainSparseDataset<u16, f32, DotProduct>,
    quants_high: &[f32],
    max_code_high: f32,
    quants_low: &[f32],
    max_code_low: f32,
    pct: f32,
    importance_scores: Option<&[f32]>,
) -> PlainSparseDataset<u16, f32, DotProduct> {
    let dim = dataset_f32.input_dim();
    let quantizer: PlainSparseQuantizer<u16, f32, DotProduct> = PlainSparseQuantizer::new(dim, dim);
    let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
        SparseDatasetGrowable::new(quantizer);

    for doc_view in dataset_f32.iter() {
        let components = doc_view.components();
        let values = doc_view.values();
        let n = components.len();

        if n == 0 {
            let empty_c: &[u16] = &[];
            let empty_v: &[f32] = &[];
            growable.push(SparseVectorView::new(empty_c, empty_v));
            continue;
        }

        // Determine which components get high bits.
        let cutoff_count = ((pct * n as f32).ceil() as usize).max(1).min(n);
        let use_high_bits: Vec<bool> = if let Some(scores) = importance_scores {
            // Rank by importance score.
            let mut scored: Vec<(usize, f32)> = components
                .iter()
                .enumerate()
                .map(|(i, &c)| (i, scores[c as usize]))
                .collect();
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let cutoff_score = scored[cutoff_count - 1].1;

            // Count how many are strictly above cutoff to handle ties.
            let n_above = scored.iter().take(cutoff_count).filter(|(_, s)| *s > cutoff_score).count();
            let mut remaining_at_cutoff = cutoff_count - n_above;

            components
                .iter()
                .map(|&c| {
                    let score = scores[c as usize];
                    if score > cutoff_score {
                        true
                    } else if score == cutoff_score && remaining_at_cutoff > 0 {
                        remaining_at_cutoff -= 1;
                        true
                    } else {
                        false
                    }
                })
                .collect()
        } else {
            // Rank by value (original behavior).
            let mut vals_sorted: Vec<f32> = values.to_vec();
            vals_sorted.select_nth_unstable_by(cutoff_count - 1, |a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            let threshold = vals_sorted[cutoff_count - 1];

            let n_above = values.iter().filter(|&&v| v > threshold).count();
            let mut remaining_at_threshold = cutoff_count - n_above;

            values
                .iter()
                .map(|&v| {
                    if v > threshold {
                        true
                    } else if v == threshold && remaining_at_threshold > 0 {
                        remaining_at_threshold -= 1;
                        true
                    } else {
                        false
                    }
                })
                .collect()
        };

        let mut new_values: Vec<f32> = Vec::with_capacity(n);
        for (i, (&c, &v)) in components.iter().zip(values.iter()).enumerate() {
            let idx = c as usize;
            let use_high = use_high_bits[i];

            let recon = if use_high {
                dequantize(v, quants_high[idx], max_code_high)
            } else {
                dequantize(v, quants_low[idx], max_code_low)
            };
            new_values.push(recon);
        }

        growable.push(SparseVectorView::new(components, &new_values));
    }

    growable.into()
}

/// Compute theoretical memory usage of the hybrid encoding in GiB (values only).
///
/// For each document with n components, top ceil(pct * n) use high_bits, rest use low_bits.
/// Each value also costs 1 selector bit. Component IDs and offsets are excluded since
/// they are the same across all methods.
fn compute_hybrid_memory_gib(
    dataset: &PlainSparseDataset<u16, f32, DotProduct>,
    high_bits: u8,
    low_bits: u8,
    pct: f32,
) -> f64 {
    let mut total_bits: u64 = 0;

    for doc_view in dataset.iter() {
        let n = doc_view.components().len() as u64;
        if n == 0 {
            continue;
        }
        let n_high = ((pct * n as f32).ceil() as u64).max(1).min(n);
        let n_low = n - n_high;

        // selector bit: 1 bit per component
        total_bits += n;
        // value bits
        total_bits += n_high * high_bits as u64 + n_low * low_bits as u64;
    }

    total_bits as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0)
}

/// Compute theoretical memory for pure uniform quantization (values only, no hybrid).
fn compute_pure_memory_gib(dataset: &PlainSparseDataset<u16, f32, DotProduct>, nbits: u8) -> f64 {
    let mut total_bits: u64 = 0;

    for doc_view in dataset.iter() {
        let n = doc_view.components().len() as u64;
        // value bits only
        total_bits += n * nbits as u64;
    }

    total_bits as f64 / (8.0 * 1024.0 * 1024.0 * 1024.0)
}

struct BenchResult {
    label: String,
    low_bits: u8,
    high_bits: u8,
    pct: f32,
    size_gib: f64,
    build_time: f64,
    search_time: f64,
    recall: f64,
}

fn main() {
    let args = Args::parse();

    fs::create_dir_all(&args.recall_dir).expect("failed to create recall output directory");

    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    // ── Load dataset and queries ────────────────────────────────────
    println!("Loading dataset (u16 components)...");
    let dataset_f32: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");

    println!("Loading queries (u16 components)...");
    let queries: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.query_file).expect("failed to read queries");

    let n = dataset_f32.len();
    let dim = dataset_f32.input_dim();
    let nnz = dataset_f32.nnz();
    let f32_size = dataset_f32.space_usage_GiB();
    let total_n_queries = queries.len();
    let n_queries = args.n_queries.min(total_n_queries);

    println!("Dataset: {n} docs, dim={dim}, nnz={nnz}");
    println!("Using {n_queries} queries out of {total_n_queries}");

    // ── Ground truth: <u16, f32> ────────────────────────────────────
    println!("\n=== <u16, f32> ground truth ===");
    println!("Size: {f32_size:.3} GiB");
    let start = Instant::now();
    let gt: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_f32.search(queries.get(qi as u64), args.k))
        .collect();
    let search_time_f32 = start.elapsed().as_secs_f64();
    println!("Search: {search_time_f32:.3}s");

    // ── Training data ───────────────────────────────────────────────
    println!("\nLoading training data...");
    let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file).expect("failed to re-read for training");

    // ── Train quantizers for each unique bit level ──────────────────
    let low_bits_values: Vec<u8> = vec![2, 3, 4];
    let high_bits: u8 = 8;
    let pct_values: Vec<f32> = vec![0.10, 0.15, 0.20];

    // Collect all unique bit levels we need quantizers for.
    let mut unique_bits: Vec<u8> = low_bits_values.clone();
    if !unique_bits.contains(&high_bits) {
        unique_bits.push(high_bits);
    }
    unique_bits.sort();
    unique_bits.dedup();

    // Train and cache quantizers.
    let mut quants_map: std::collections::HashMap<u8, (Vec<f32>, f32)> =
        std::collections::HashMap::new();
    for &nbits in &unique_bits {
        println!("Training {nbits}-bit quantizer...");
        let quantizer = VariableBitUniformSparseQuantizer::<u16, DotProduct>::train(
            &training_data,
            args.lower_percentile,
            args.upper_percentile,
            nbits,
        );
        quants_map.insert(nbits, (quantizer.quants().to_vec(), quantizer.max_val()));
    }
    drop(training_data);

    let (quants_high, max_code_high) = &quants_map[&high_bits];

    // ── Compute component importance (query_val × doc_val averaged across queries) ────────────
    println!("\nComputing component importance scores...");
    let importance_scores = compute_component_importance(&dataset_f32, &queries, n_queries);

    // ── Run benchmarks ──────────────────────────────────────────────
    let mut results: Vec<BenchResult> = Vec::new();

    // --- Pure baselines for each low_bits level ---
    for &low_bits in &low_bits_values {
        let label = format!("pure_{low_bits}bit");
        println!("\n=== {label} ===");

        let (quants_low, max_code_low) = &quants_map[&low_bits];

        let start = Instant::now();
        let dataset_pure = build_hybrid_dataset(
            &dataset_f32,
            quants_low,
            *max_code_low,
            quants_low,
            *max_code_low,
            1.0, // 100% high = all use the same quantizer
            None,
        );
        let build_time = start.elapsed().as_secs_f64();
        let size_gib = compute_pure_memory_gib(&dataset_f32, low_bits);
        println!("Build:  {build_time:.3}s");
        println!("Size:   {size_gib:.3} GiB (theoretical)");

        let start = Instant::now();
        let approx: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| dataset_pure.search(queries.get(qi as u64), args.k))
            .collect();
        let search_time = start.elapsed().as_secs_f64();
        println!("Search: {search_time:.3}s");

        let (recall, per_query_recall) = recall_at_k(&gt, &approx, args.k);
        println!("Recall@{}: {recall:.4}", args.k);

        let npy_path = format!("{}/recalls_pure_{}bits.npy", args.recall_dir, low_bits);
        let arr = Array1::from(per_query_recall);
        write_npy(&npy_path, &arr).expect("failed to write per-query recall npy");
        println!("Per-query recall written to {npy_path}");

        results.push(BenchResult {
            label,
            low_bits,
            high_bits: low_bits,
            pct: 0.0,
            size_gib,
            build_time,
            search_time,
            recall,
        });
    }

    // --- Pure 8-bit baseline ---
    {
        let label = "pure_8bit".to_string();
        println!("\n=== {label} ===");

        let start = Instant::now();
        let dataset_8bit = build_hybrid_dataset(
            &dataset_f32,
            quants_high,
            *max_code_high,
            quants_high,
            *max_code_high,
            1.0,
            None,
        );
        let build_time = start.elapsed().as_secs_f64();
        let size_gib = compute_pure_memory_gib(&dataset_f32, high_bits);
        println!("Build:  {build_time:.3}s");
        println!("Size:   {size_gib:.3} GiB (theoretical)");

        let start = Instant::now();
        let approx: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| dataset_8bit.search(queries.get(qi as u64), args.k))
            .collect();
        let search_time = start.elapsed().as_secs_f64();
        println!("Search: {search_time:.3}s");

        let (recall, per_query_recall) = recall_at_k(&gt, &approx, args.k);
        println!("Recall@{}: {recall:.4}", args.k);

        let npy_path = format!("{}/recalls_pure_8bits.npy", args.recall_dir);
        let arr = Array1::from(per_query_recall);
        write_npy(&npy_path, &arr).expect("failed to write per-query recall npy");
        println!("Per-query recall written to {npy_path}");

        results.push(BenchResult {
            label,
            low_bits: 8,
            high_bits: 8,
            pct: 0.0,
            size_gib,
            build_time,
            search_time,
            recall,
        });
    }

    // --- Hybrid configurations ---
    for &low_bits in &low_bits_values {
        let (quants_low, max_code_low) = &quants_map[&low_bits];

        for &pct in &pct_values {
            let label = format!("hybrid_{low_bits}b+8b_top{:.0}pct", pct * 100.0);
            println!("\n=== {label} ===");

            let start = Instant::now();
            let dataset_hybrid = build_hybrid_dataset(
                &dataset_f32,
                quants_high,
                *max_code_high,
                quants_low,
                *max_code_low,
                pct,
                Some(&importance_scores),
            );
            let build_time = start.elapsed().as_secs_f64();
            let size_gib = compute_hybrid_memory_gib(&dataset_f32, high_bits, low_bits, pct);
            println!("Build:  {build_time:.3}s");
            println!("Size:   {size_gib:.3} GiB (theoretical, incl. 1 selector bit/value)");

            let start = Instant::now();
            let approx: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(pb_style.clone())
                .map(|qi| dataset_hybrid.search(queries.get(qi as u64), args.k))
                .collect();
            let search_time = start.elapsed().as_secs_f64();
            println!("Search: {search_time:.3}s");

            let (recall, per_query_recall) = recall_at_k(&gt, &approx, args.k);
            println!("Recall@{}: {recall:.4}", args.k);

            let npy_path = format!(
                "{}/recalls_hybrid_{low_bits}b_8b_top{:.0}pct.npy",
                args.recall_dir,
                pct * 100.0
            );
            let arr = Array1::from(per_query_recall);
            write_npy(&npy_path, &arr).expect("failed to write per-query recall npy");
            println!("Per-query recall written to {npy_path}");

            results.push(BenchResult {
                label,
                low_bits,
                high_bits,
                pct,
                size_gib,
                build_time,
                search_time,
                recall,
            });
        }
    }

    // ── Summary table (stdout) ──────────────────────────────────────
    println!("\n=== Summary ===");
    println!(
        "{:<30} {:>6} {:>6} {:>6} {:>10} {:>12} {:>12} {:>10}",
        "config", "lo_bit", "hi_bit", "pct", "Size GiB", "Build (s)", "Search (s)", "Recall@k"
    );
    println!("{:-<96}", "");
    println!(
        "{:<30} {:>6} {:>6} {:>6} {:>10.3} {:>12} {:>12.3} {:>10}",
        "f32", "-", "-", "-", f32_size, "-", search_time_f32, "1.0000"
    );
    for r in &results {
        let pct_str = if r.pct > 0.0 {
            format!("{:.0}%", r.pct * 100.0)
        } else {
            "-".to_string()
        };
        println!(
            "{:<30} {:>6} {:>6} {:>6} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
            r.label,
            r.low_bits,
            r.high_bits,
            pct_str,
            r.size_gib,
            r.build_time,
            r.search_time,
            r.recall
        );
    }

    // ── Write TSV ───────────────────────────────────────────────────
    let mut f = std::fs::File::create(&args.output).expect("failed to create output TSV");
    writeln!(
        f,
        "config\tlow_bits\thigh_bits\tpct\tsize_gib\tbuild_time_s\tsearch_time_s\trecall_at_k"
    )
    .unwrap();
    writeln!(
        f,
        "f32\t-\t-\t-\t{f32_size:.6}\t\t{search_time_f32:.6}\t1.000000"
    )
    .unwrap();
    for r in &results {
        let pct_str = if r.pct > 0.0 {
            format!("{:.2}", r.pct)
        } else {
            "".to_string()
        };
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            r.label,
            r.low_bits,
            r.high_bits,
            pct_str,
            r.size_gib,
            r.build_time,
            r.search_time,
            r.recall
        )
        .unwrap();
    }
    println!("\nResults written to {}", args.output);
}
