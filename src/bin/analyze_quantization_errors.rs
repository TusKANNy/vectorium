use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    Dataset, DatasetGrowable, PlainSparseDataset, VariableBitUniformSparseDataset,
    VariableBitUniformSparseDatasetGrowable, VariableBitUniformSparseQuantizer,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Analyse per-component quantization error for 4-bit and 8-bit variable-bit uniform quantizers"
)]
struct Args {
    /// Sparse dataset in Seismic binary format
    #[clap(short, long)]
    input_file: String,

    /// Sparse queries in Seismic binary format
    #[clap(short, long)]
    query_file: String,

    /// Pre-computed ground-truth file (format written by compute_groundtruth)
    /// Each line: query_id TAB doc_id TAB rank TAB score
    #[clap(short, long)]
    groundtruth_file: String,

    /// Output directory for per-query analysis files
    #[clap(short, long, default_value = "quantization_analysis")]
    output_dir: String,

    /// Number of queries to sample
    #[clap(long, default_value_t = 10)]
    n_samples: usize,

    /// Number of top results from the ground truth to inspect
    #[clap(short, long, default_value_t = 10)]
    k: usize,

    /// How deep to search when computing quantized ranks (must be >= k)
    #[clap(long, default_value_t = 500)]
    rank_depth: usize,

    /// Lower percentile for quantizer training
    #[clap(long, default_value_t = 0.0)]
    lower_percentile: f32,

    /// Upper percentile for quantizer training
    #[clap(long, default_value_t = 1.0)]
    upper_percentile: f32,

    /// Optional JSON file mapping token -> component_id (e.g. token_to_id_mapping.json).
    /// The mapping is inverted to annotate component IDs with their token strings.
    #[clap(long)]
    token_to_id_file: Option<String>,

    /// Percentage of each document's top components (by value) that would use 8-bit
    /// quantization in the hybrid scheme. The rest uses 4-bit.
    /// The correction analysis is shown only when 8-bit and 4-bit recall differ.
    #[clap(long, default_value_t = 20.0)]
    hybrid_top_pct: f32,
}

// ── Vocabulary ────────────────────────────────────────────────────────────────

/// Load a `token -> id` JSON file and return an `id -> token` lookup vector.
/// The JSON must be a flat object: `{"token": id, ...}`.
fn load_id_to_token(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let token_to_id: HashMap<String, usize> = serde_json::from_reader(reader)?;

    let max_id = token_to_id.values().copied().max().unwrap_or(0);
    let mut id_to_token = vec![String::new(); max_id + 1];
    for (token, id) in token_to_id {
        id_to_token[id] = token;
    }
    Ok(id_to_token)
}

// ── Ground truth ──────────────────────────────────────────────────────────────

struct GtEntry {
    doc_id: u64,
    rank: usize,
    score: f32,
}

/// Load the ground-truth TSV into a map: query_id -> Vec<GtEntry> (sorted by rank).
fn load_groundtruth(
    path: &str,
) -> Result<HashMap<usize, Vec<GtEntry>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut map: HashMap<usize, Vec<GtEntry>> = HashMap::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 4 {
            eprintln!(
                "Warning: skipping malformed line {} (expected 4 columns, got {})",
                line_no + 1,
                parts.len()
            );
            continue;
        }
        let query_id: usize = parts[0].parse()?;
        let doc_id: u64 = parts[1].parse()?;
        let rank: usize = parts[2].parse()?;
        let score: f32 = parts[3].parse()?;

        map.entry(query_id).or_default().push(GtEntry {
            doc_id,
            rank,
            score,
        });
    }

    for entries in map.values_mut() {
        entries.sort_by_key(|e| e.rank);
    }

    Ok(map)
}

// ── Quantization helpers ──────────────────────────────────────────────────────

#[inline]
fn quantize(value: f32, quant_step: f32, max_code: f32) -> (u8, f32) {
    if quant_step <= 0.0 {
        return (0, 0.0);
    }
    let code = (value / quant_step).clamp(0.0, max_code) as u8;
    (code, code as f32 * quant_step)
}

// ── Search helpers ────────────────────────────────────────────────────────────

fn rank_map(results: &[vectorium::dataset::ScoredVector<DotProduct>]) -> HashMap<u64, usize> {
    results
        .iter()
        .enumerate()
        .map(|(i, sv)| (sv.vector, i + 1))
        .collect()
}

fn recall_at_k(
    gt_top_k: &[&GtEntry],
    quantized_results: &[vectorium::dataset::ScoredVector<DotProduct>],
    k: usize,
) -> f64 {
    let gt_ids: std::collections::HashSet<u64> =
        gt_top_k.iter().take(k).map(|e| e.doc_id).collect();
    let found = quantized_results
        .iter()
        .take(k)
        .filter(|sv| gt_ids.contains(&sv.vector))
        .count();
    found as f64 / gt_ids.len() as f64
}

fn build_quantized_dataset(
    dataset_f32: &PlainSparseDataset<u16, f32, DotProduct>,
    quantizer: VariableBitUniformSparseQuantizer<u16, DotProduct>,
) -> VariableBitUniformSparseDataset<u16, DotProduct> {
    let mut growable: VariableBitUniformSparseDatasetGrowable<u16, DotProduct> =
        VariableBitUniformSparseDatasetGrowable::new(quantizer);
    for vec in dataset_f32.iter() {
        growable.push(vec);
    }
    growable.into()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    assert!(
        args.rank_depth >= args.k,
        "rank_depth ({}) must be >= k ({})",
        args.rank_depth,
        args.k
    );

    let output_dir = PathBuf::from(&args.output_dir);
    std::fs::create_dir_all(&output_dir)?;

    // ── Load dataset ────────────────────────────────────────────────
    println!("Loading dataset (u16 components)...");
    let dataset: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file)?;
    println!(
        "Dataset: {} docs, dim={}, nnz={}",
        dataset.len(),
        dataset.input_dim(),
        dataset.nnz()
    );

    // ── Load queries ────────────────────────────────────────────────
    println!("Loading queries (u16 components)...");
    let queries: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.query_file)?;
    println!("Queries: {} total", queries.len());

    // ── Load vocabulary (optional) ───────────────────────────────────
    let vocab: Option<Vec<String>> = match &args.token_to_id_file {
        Some(path) => {
            println!("Loading token->id mapping from {}...", path);
            let v = load_id_to_token(path)?;
            println!("Vocabulary: {} tokens", v.len());
            Some(v)
        }
        None => None,
    };

    // Helper: look up the token string for a component id.
    let token_of = |c: u16| -> &str {
        vocab
            .as_ref()
            .and_then(|v| v.get(c as usize))
            .map(|s| s.as_str())
            .unwrap_or("")
    };

    // ── Load ground truth ───────────────────────────────────────────
    println!("Loading ground truth from {}...", args.groundtruth_file);
    let groundtruth = load_groundtruth(&args.groundtruth_file)?;
    println!("Ground truth: {} queries covered", groundtruth.len());

    // ── Train quantizers ─────────────────────────────────────────────
    println!("Re-loading dataset for quantizer training...");
    let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file)?;

    println!("Training 8-bit quantizer...");
    let quantizer_8bit = VariableBitUniformSparseQuantizer::<u16, DotProduct>::train(
        &training_data,
        args.lower_percentile,
        args.upper_percentile,
        8,
    );
    let quants_8bit: Vec<f32> = quantizer_8bit.quants().to_vec();
    let max_code_8bit = quantizer_8bit.max_val();

    println!("Training 4-bit quantizer...");
    let quantizer_4bit = VariableBitUniformSparseQuantizer::<u16, DotProduct>::train(
        &training_data,
        args.lower_percentile,
        args.upper_percentile,
        4,
    );
    let quants_4bit: Vec<f32> = quantizer_4bit.quants().to_vec();
    let max_code_4bit = quantizer_4bit.max_val();

    drop(training_data);

    // ── Build quantized datasets (needed for search / rank) ──────────
    println!("Building 8-bit quantized dataset...");
    let dataset_8bit = build_quantized_dataset(&dataset, quantizer_8bit);

    println!("Building 4-bit quantized dataset...");
    let dataset_4bit = build_quantized_dataset(&dataset, quantizer_4bit);

    let rank_depth = args.rank_depth.min(dataset.len());

    // ── Sample queries ───────────────────────────────────────────────
    let mut valid_query_ids: Vec<usize> = groundtruth
        .keys()
        .copied()
        .filter(|&qid| (qid as u64) < queries.len() as u64)
        .collect();
    valid_query_ids.sort();

    if valid_query_ids.is_empty() {
        eprintln!("No valid queries found. Exiting.");
        return Ok(());
    }

    let n_samples = args.n_samples.min(valid_query_ids.len());
    let step = valid_query_ids.len() / n_samples;
    let sampled_ids: Vec<usize> = (0..n_samples).map(|i| valid_query_ids[i * step]).collect();
    println!("Sampled {} query IDs: {:?}", sampled_ids.len(), sampled_ids);

    // ── Analyse each sampled query ────────────────────────────────────
    let has_vocab = vocab.is_some();

    for &query_id in &sampled_ids {
        let query_vec = queries.get(query_id as u64);
        let q_components = query_vec.components();
        let q_values = query_vec.values();

        let gt_entries = match groundtruth.get(&query_id) {
            Some(e) => e,
            None => {
                eprintln!("Query {} not found in ground truth, skipping.", query_id);
                continue;
            }
        };

        let top_k: Vec<&GtEntry> = gt_entries.iter().take(args.k).collect();

        let q_comp_map: HashMap<u16, f32> = q_components
            .iter()
            .copied()
            .zip(q_values.iter().copied())
            .collect();

        // ── Quantized search for recall / ranks ───────────────────────
        let results_8bit = dataset_8bit.search(queries.get(query_id as u64), rank_depth);
        let results_4bit = dataset_4bit.search(queries.get(query_id as u64), rank_depth);

        let ranks_8bit = rank_map(&results_8bit);
        let ranks_4bit = rank_map(&results_4bit);

        let recall_8bit = recall_at_k(&top_k, &results_8bit, args.k);
        let recall_4bit = recall_at_k(&top_k, &results_4bit, args.k);

        // ── Write output file ─────────────────────────────────────────
        let out_path = output_dir.join(format!("query_{:06}.txt", query_id));
        let mut out = File::create(&out_path)?;

        writeln!(out, "=== Query {} ===", query_id)?;
        writeln!(out)?;
        writeln!(
            out,
            "Recall@{k} (vs ground truth):  8-bit = {:.4}   4-bit = {:.4}",
            recall_8bit,
            recall_4bit,
            k = args.k
        )?;
        writeln!(
            out,
            "Rank search depth: {} (docs ranked beyond this shown as >{})",
            rank_depth, rank_depth
        )?;
        writeln!(out)?;

        // ── Query vector table (sorted by decreasing query value) ────────
        let mut q_sorted: Vec<(u16, f32)> = q_components
            .iter()
            .copied()
            .zip(q_values.iter().copied())
            .collect();
        q_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        writeln!(out, "Query components ({} non-zeros):", q_sorted.len())?;
        if has_vocab {
            writeln!(out, "  {:>10}  {:<30}  {:>14}", "component", "token", "query_value")?;
        } else {
            writeln!(out, "  {:>10}  {:>14}", "component", "query_value")?;
        }
        for &(c, v) in &q_sorted {
            if has_vocab {
                writeln!(out, "  {:>10}  {:<30}  {:>14.6}", c, token_of(c), v)?;
            } else {
                writeln!(out, "  {:>10}  {:>14.6}", c, v)?;
            }
        }
        writeln!(out)?;

        // ── Per-document analysis ─────────────────────────────────────
        writeln!(out, "Top-{} documents (from ground truth):", top_k.len())?;
        writeln!(out)?;

        for entry in &top_k {
            let doc_id = entry.doc_id;
            let doc_vec = dataset.get(doc_id);
            let doc_comp_map: HashMap<u16, f32> = doc_vec
                .components()
                .iter()
                .copied()
                .zip(doc_vec.values().iter().copied())
                .collect();

            let rank_8bit_str = ranks_8bit
                .get(&doc_id)
                .map(|r| r.to_string())
                .unwrap_or_else(|| format!(">{}", rank_depth));
            let rank_4bit_str = ranks_4bit
                .get(&doc_id)
                .map(|r| r.to_string())
                .unwrap_or_else(|| format!(">{}", rank_depth));

            // Build component -> rank within the document (sorted by decreasing value, 1-indexed).
            let mut doc_sorted: Vec<(u16, f32)> = doc_comp_map
                .iter()
                .map(|(&c, &v)| (c, v))
                .collect();
            doc_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let doc_comp_rank: HashMap<u16, usize> = doc_sorted
                .iter()
                .enumerate()
                .map(|(i, &(c, _))| (c, i + 1))
                .collect();

            // (component, query_val, original_val, doc_rank, recon_8, err_8, dmg_8, recon_4, err_4, dmg_4)
            // dmg = query_val * err — the contribution of this component's quantization error
            // to the final dot-product score loss.
            let mut intersecting: Vec<(u16, f32, f32, usize, f32, f32, f32, f32, f32, f32)> = q_components
                .iter()
                .filter_map(|&c| {
                    doc_comp_map.get(&c).map(|&orig| {
                        let idx = c as usize;
                        let qv = q_comp_map[&c];
                        let (_, r8) = quantize(orig, quants_8bit[idx], max_code_8bit);
                        let (_, r4) = quantize(orig, quants_4bit[idx], max_code_4bit);
                        let e8 = orig - r8;
                        let e4 = orig - r4;
                        let doc_rank = doc_comp_rank[&c];
                        (c, qv, orig, doc_rank, r8, e8, qv * e8, r4, e4, qv * e4)
                    })
                })
                .collect();

            // Hybrid cutoff: top hybrid_top_pct% of document components use 8-bit.
            let n_doc_components = doc_comp_map.len();
            let hybrid_cutoff = ((args.hybrid_top_pct / 100.0) * n_doc_components as f32).ceil() as usize;

            let dot_f32: f32 = intersecting.iter().map(|t| t.1 * t.2).sum();
            let dot_8bit: f32 = intersecting.iter().map(|t| t.1 * t.4).sum();
            let dot_4bit: f32 = intersecting.iter().map(|t| t.1 * t.7).sum();
            let total_dmg_8: f32 = intersecting.iter().map(|t| t.6).sum();
            let total_dmg_4: f32 = intersecting.iter().map(|t| t.9).sum();

            // Hybrid dot: for components in the top% by doc value, use 8-bit recon; rest use 4-bit.
            let dot_hybrid: f32 = intersecting
                .iter()
                .map(|t| {
                    let recon = if t.3 <= hybrid_cutoff { t.4 } else { t.7 };
                    t.1 * recon
                })
                .sum();
            let total_dmg_hybrid: f32 = intersecting
                .iter()
                .map(|t| {
                    let dmg = if t.3 <= hybrid_cutoff { t.6 } else { t.9 };
                    dmg
                })
                .sum();

            writeln!(
                out,
                "--- Doc {} | gt_rank: {}  8bit_rank: {}  4bit_rank: {} | gt_score: {:.6} ---",
                doc_id, entry.rank, rank_8bit_str, rank_4bit_str, entry.score
            )?;
            writeln!(
                out,
                "  Doc has {} components, hybrid cutoff: top {} use 8-bit ({:.0}%)",
                n_doc_components, hybrid_cutoff, args.hybrid_top_pct
            )?;
            writeln!(out, "  Intersection size: {} components", intersecting.len())?;
            writeln!(
                out,
                "  Dot (intersection only):  f32={:.6}  8bit={:.6}  4bit={:.6}",
                dot_f32, dot_8bit, dot_4bit
            )?;
            writeln!(
                out,
                "  Total score damage:       8bit={:.6}  4bit={:.6}",
                total_dmg_8, total_dmg_4
            )?;

            // Show hybrid correction when 8-bit and 4-bit recall differ.
            if (recall_8bit - recall_4bit).abs() > 1e-9 {
                let damage_4bit = dot_f32 - dot_4bit;
                let damage_hybrid = dot_f32 - dot_hybrid;
                let recovery_pct = if damage_4bit.abs() > 1e-9 {
                    100.0 * (1.0 - damage_hybrid / damage_4bit)
                } else {
                    0.0
                };
                writeln!(
                    out,
                    "  Hybrid (top-{:.0}% 8bit):   dot={:.6}  damage={:.6}  (recovers {:.1}% of 4bit loss)",
                    args.hybrid_top_pct, dot_hybrid, total_dmg_hybrid, recovery_pct
                )?;
            }
            writeln!(out)?;

            if intersecting.is_empty() {
                writeln!(out, "  (no intersecting components)")?;
            } else {
                // Sort by decreasing 4-bit damage — worst offenders first.
                intersecting.sort_by(|a, b| {
                    b.9.partial_cmp(&a.9).unwrap_or(std::cmp::Ordering::Equal)
                });

                if has_vocab {
                    writeln!(
                        out,
                        "  {:>10}  {:<25}  {:>12}  {:>14}  {:>8}  {:>4}  {:>12}  {:>10}  {:>10}  {:>12}  {:>10}  {:>10}",
                        "component", "token", "query_val", "original_val", "doc_rank", "hyb",
                        "recon_8bit", "err_8bit", "dmg_8bit", "recon_4bit", "err_4bit", "dmg_4bit"
                    )?;
                } else {
                    writeln!(
                        out,
                        "  {:>10}  {:>12}  {:>14}  {:>8}  {:>4}  {:>12}  {:>10}  {:>10}  {:>12}  {:>10}  {:>10}",
                        "component", "query_val", "original_val", "doc_rank", "hyb",
                        "recon_8bit", "err_8bit", "dmg_8bit", "recon_4bit", "err_4bit", "dmg_4bit"
                    )?;
                }

                for &(c, qv, orig, doc_rank, r8, e8, d8, r4, e4, d4) in &intersecting {
                    let hyb_label = if doc_rank <= hybrid_cutoff { "8b" } else { "4b" };
                    if has_vocab {
                        writeln!(
                            out,
                            "  {:>10}  {:<25}  {:>12.6}  {:>14.6}  {:>8}  {:>4}  {:>12.6}  {:>10.6}  {:>10.6}  {:>12.6}  {:>10.6}  {:>10.6}",
                            c, token_of(c), qv, orig, doc_rank, hyb_label, r8, e8, d8, r4, e4, d4
                        )?;
                    } else {
                        writeln!(
                            out,
                            "  {:>10}  {:>12.6}  {:>14.6}  {:>8}  {:>4}  {:>12.6}  {:>10.6}  {:>10.6}  {:>12.6}  {:>10.6}  {:>10.6}",
                            c, qv, orig, doc_rank, hyb_label, r8, e8, d8, r4, e4, d4
                        )?;
                    }
                }
            }
            writeln!(out)?;
        }

        println!("Written: {}", out_path.display());
    }

    println!("\nDone. Analysis files written to '{}'.", args.output_dir);
    Ok(())
}
