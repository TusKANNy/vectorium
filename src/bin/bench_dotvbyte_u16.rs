use clap::Parser;
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::dataset::{ConvertInto, ScoredVector};
use vectorium::distances::DotProduct;
use vectorium::encoders::dotvbyte_scalaru8::DotVByteScalarU8Encoder;
use vectorium::readers;
use vectorium::{
    Block8FixedU8Encoder, Dataset, DatasetGrowable, DotVByteFixedU8Encoder, FixedU8Q,
    PackedSparseDataset, PackedSparseDatasetGrowable, PlainSparseDataset,
    ReverseExpSparseQuantizer, ScalarSparseDataset, SpaceUsage, SparseDatasetGrowable,
    UniformSparseQuantizer,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Compare sparse encoding methods: f32, scalar quantizers, Block8, DotVByte"
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
}

fn recall_at_k(
    gt: &[Vec<ScoredVector<DotProduct>>],
    approx: &[Vec<ScoredVector<DotProduct>>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;
    for (gt_row, approx_row) in gt.iter().zip(approx.iter()) {
        let gt_ids: std::collections::HashSet<u64> =
            gt_row.iter().take(k).map(|s| s.vector).collect();
        let found = approx_row
            .iter()
            .take(k)
            .filter(|s| gt_ids.contains(&s.vector))
            .count();
        total_recall += found as f64 / k as f64;
    }
    total_recall / gt.len() as f64
}

fn main() {
    let args = Args::parse();

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
    let run_dvb_u16 = dim <= u16::MAX as usize;
    let nnz = dataset_f32.nnz();
    let f32_size = dataset_f32.space_usage_GiB();
    let total_n_queries = queries.len();
    let n_queries = args.n_queries.min(total_n_queries);

    println!("Dataset: {n} docs, dim={dim}, nnz={nnz}");
    println!("Using {n_queries} queries out of {total_n_queries}");

    // ── 1. Ground truth: <u16, f32> ─────────────────────────────────
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

    // ── 2. <u16, fixedu8> ───────────────────────────────────────────
    println!("\n=== <u16, FixedU8> ===");
    let start = Instant::now();
    let dataset_fixedu8: ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct> =
        (&dataset_f32).convert_into();
    let build_time_fixedu8 = start.elapsed().as_secs_f64();
    let fixedu8_size = dataset_fixedu8.space_usage_GiB();
    println!("Build:  {build_time_fixedu8:.3}s");
    println!("Size:   {fixedu8_size:.3} GiB");

    let start = Instant::now();
    let results_fixedu8: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_fixedu8.search(queries.get(qi as u64), args.k))
        .collect();
    let search_time_fixedu8 = start.elapsed().as_secs_f64();
    println!("Search: {search_time_fixedu8:.3}s");

    // ── Shared training data for trainable sparse quantizers ────────
    let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file).expect("failed to re-read for training");

    // ── 3. <u16, scalar> (uniform quantization) ─────────────────────
    println!("\n=== <u16, scalar> (uniform quantization) ===");

    let start = Instant::now();
    let quantizer = UniformSparseQuantizer::<u16, DotProduct>::train(&training_data, 0.0, 1.0);
    let train_time_scalar = start.elapsed().as_secs_f64();
    println!("Train:  {train_time_scalar:.3}s");

    let start = Instant::now();
    let mut growable: SparseDatasetGrowable<UniformSparseQuantizer<u16, DotProduct>> =
        SparseDatasetGrowable::new(quantizer);
    for vec in dataset_f32.iter() {
        growable.push(vec);
    }
    let dataset_scalar: vectorium::UniformSparseDataset<u16, DotProduct> = growable.into();
    let build_time_scalar = start.elapsed().as_secs_f64();
    let scalar_size = dataset_scalar.space_usage_GiB();
    println!("Build:  {build_time_scalar:.3}s");
    println!("Size:   {scalar_size:.3} GiB");

    let start = Instant::now();
    let results_scalar: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_scalar.search(queries.get(qi as u64), args.k))
        .collect();
    let search_time_scalar = start.elapsed().as_secs_f64();
    println!("Search: {search_time_scalar:.3}s");

    // ── 4. <u16, reverse-exp> ───────────────────────────────────────
    println!("\n=== <u16, reverse-exp> ===");
    let start = Instant::now();
    let revexp_quantizer = ReverseExpSparseQuantizer::<u16, DotProduct>::train(&training_data);
    let train_time_revexp = start.elapsed().as_secs_f64();
    println!("Train:  {train_time_revexp:.3}s");

    let start = Instant::now();
    let mut growable: SparseDatasetGrowable<ReverseExpSparseQuantizer<u16, DotProduct>> =
        SparseDatasetGrowable::new(revexp_quantizer);
    for vec in dataset_f32.iter() {
        growable.push(vec);
    }
    let dataset_revexp: vectorium::ReverseExpSparseDataset<u16, DotProduct> = growable.into();
    let build_time_revexp = start.elapsed().as_secs_f64();
    let revexp_size = dataset_revexp.space_usage_GiB();
    println!("Build:  {build_time_revexp:.3}s");
    println!("Size:   {revexp_size:.3} GiB");

    let start = Instant::now();
    let results_revexp: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_revexp.search(queries.get(qi as u64), args.k))
        .collect();
    let search_time_revexp = start.elapsed().as_secs_f64();
    println!("Search: {search_time_revexp:.3}s");

    // ── 5. Block8 <packed, fixedu8> ─────────────────────────────────
    let (block8_size, build_time_block8, search_time_block8, results_block8) = if run_dvb_u16 {
        println!("\n=== Block8 <packed, fixedu8> ===");
        let dataset_for_block8: PlainSparseDataset<u16, f32, DotProduct> =
            readers::read_seismic_format(&args.input_file).expect("failed to read dataset");

        let start = Instant::now();
        let dataset_block8: PackedSparseDataset<Block8FixedU8Encoder> =
            dataset_for_block8.convert_into();
        let build_time = start.elapsed().as_secs_f64();
        let size = dataset_block8.space_usage_GiB();
        println!("Build:  {build_time:.3}s");
        println!("Size:   {size:.3} GiB");

        let start = Instant::now();
        let results: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| dataset_block8.search(queries.get(qi as u64), args.k))
            .collect();
        let search_time = start.elapsed().as_secs_f64();
        println!("Search: {search_time:.3}s");
        (
            Some(size),
            Some(build_time),
            Some(search_time),
            Some(results),
        )
    } else {
        println!(
            "\n=== Block8 <packed, fixedu8> ===\nSkipping: dim {} exceeds u16 max {}",
            dim,
            u16::MAX
        );
        (None, None, None, None)
    };

    // ── 6. DotVByte <packed, fixedu8> ───────────────────────────────
    let (dvb_size, build_time_dvb, search_time_dvb, results_dvb) = if run_dvb_u16 {
        println!("\n=== DotVByte <packed, fixedu8> ===");
        // Re-load to get an owned dataset for convert_into (consumes it)
        let dataset_for_dvb: PlainSparseDataset<u16, f32, DotProduct> =
            readers::read_seismic_format(&args.input_file).expect("failed to read dataset");
        let queries_for_dvb: PlainSparseDataset<u16, f32, DotProduct> =
            readers::read_seismic_format(&args.query_file).expect("failed to read queries");

        let start = Instant::now();
        let dataset_dvb: PackedSparseDataset<DotVByteFixedU8Encoder> =
            dataset_for_dvb.convert_into();
        let build_time = start.elapsed().as_secs_f64();
        let size = dataset_dvb.space_usage_GiB();
        println!("Build:  {build_time:.3}s");
        println!("Size:   {size:.3} GiB");

        let start = Instant::now();
        let results: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| dataset_dvb.search(queries_for_dvb.get(qi as u64), args.k))
            .collect();
        let search_time = start.elapsed().as_secs_f64();
        println!("Search: {search_time:.3}s");
        (
            Some(size),
            Some(build_time),
            Some(search_time),
            Some(results),
        )
    } else {
        println!(
            "\n=== DotVByte <packed, fixedu8> ===\nSkipping: dim {} exceeds u16 max {}",
            dim,
            u16::MAX
        );
        (None, None, None, None)
    };

    // ── 7. DotVByte <packed, scalaru8> ──────────────────────────────
    let (
        dvb_scalaru8_size,
        train_time_dvb_scalaru8,
        build_time_dvb_scalaru8,
        search_time_dvb_scalaru8,
        results_dvb_scalaru8,
    ) = if run_dvb_u16 {
        println!("\n=== DotVByte <packed, scalaru8> ===");

        let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
            readers::read_seismic_format(&args.input_file)
                .expect("failed to re-read for DotVByte scalaru8 training");

        let start = Instant::now();
        let mut encoder = DotVByteScalarU8Encoder::new(dim, dim);
        encoder.train::<f32>(&training_data);
        let train_time = start.elapsed().as_secs_f64();
        println!("Train:  {train_time:.3}s");
        drop(training_data);

        let start = Instant::now();
        let mut growable: PackedSparseDatasetGrowable<DotVByteScalarU8Encoder> =
            PackedSparseDatasetGrowable::new(encoder);
        for vec in dataset_f32.iter() {
            growable.push(vec);
        }
        let dataset_dvb_scalaru8: PackedSparseDataset<DotVByteScalarU8Encoder> = growable.into();
        let build_time = start.elapsed().as_secs_f64();
        let size = dataset_dvb_scalaru8.space_usage_GiB();
        println!("Build:  {build_time:.3}s");
        println!("Size:   {size:.3} GiB");

        let start = Instant::now();
        let results: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| dataset_dvb_scalaru8.search(queries.get(qi as u64), args.k))
            .collect();
        let search_time = start.elapsed().as_secs_f64();
        println!("Search: {search_time:.3}s");

        (
            Some(size),
            Some(train_time),
            Some(build_time),
            Some(search_time),
            Some(results),
        )
    } else {
        println!(
            "\n=== DotVByte <packed, scalaru8> ===\nSkipping: dim {} exceeds u16 max {}",
            dim,
            u16::MAX
        );
        (None, None, None, None, None)
    };

    // ── Results ────────────────────────────────────────────────────
    let recall_fixedu8 = recall_at_k(&gt, &results_fixedu8, args.k);
    let recall_scalar = recall_at_k(&gt, &results_scalar, args.k);
    let recall_revexp = recall_at_k(&gt, &results_revexp, args.k);
    let recall_block8 = results_block8
        .as_ref()
        .map(|results| recall_at_k(&gt, results, args.k));
    let recall_dvb = results_dvb
        .as_ref()
        .map(|results| recall_at_k(&gt, results, args.k));
    let recall_dvb_scalaru8 = results_dvb_scalaru8
        .as_ref()
        .map(|results| recall_at_k(&gt, results, args.k));

    println!("\n=== Summary ===");
    println!(
        "{:<25} {:>10} {:>12} {:>12} {:>10}",
        "Method", "Size GiB", "Build (s)", "Search (s)", "Recall@k"
    );
    println!("{:-<71}", "");
    println!(
        "{:<25} {:>10.3} {:>12} {:>12.3} {:>10}",
        "<u16, f32>", f32_size, "-", search_time_f32, "1.0000"
    );
    println!(
        "{:<25} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
        "<u16, fixedu8>", fixedu8_size, build_time_fixedu8, search_time_fixedu8, recall_fixedu8
    );
    println!(
        "{:<25} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
        "<u16, scalar>", scalar_size, build_time_scalar, search_time_scalar, recall_scalar
    );
    println!(
        "{:<25} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
        "<u16, reverse-exp>",
        revexp_size,
        train_time_revexp + build_time_revexp,
        search_time_revexp,
        recall_revexp
    );
    if let (Some(size), Some(build), Some(search), Some(recall)) = (
        block8_size,
        build_time_block8,
        search_time_block8,
        recall_block8,
    ) {
        println!(
            "{:<25} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
            "Block8 <packed, fixedu8>", size, build, search, recall
        );
    } else {
        println!(
            "{:<25} {:>10} {:>12} {:>12} {:>10}",
            "Block8 <packed, fixedu8>", "skipped", "-", "-", "-"
        );
    }
    if let (Some(size), Some(build), Some(search), Some(recall)) =
        (dvb_size, build_time_dvb, search_time_dvb, recall_dvb)
    {
        println!(
            "{:<25} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
            "DotVByte <packed, fixedu8>", size, build, search, recall
        );
    } else {
        println!(
            "{:<25} {:>10} {:>12} {:>12} {:>10}",
            "DotVByte <packed, fixedu8>", "skipped", "-", "-", "-"
        );
    }
    if let (Some(size), Some(train), Some(build), Some(search), Some(recall)) = (
        dvb_scalaru8_size,
        train_time_dvb_scalaru8,
        build_time_dvb_scalaru8,
        search_time_dvb_scalaru8,
        recall_dvb_scalaru8,
    ) {
        println!(
            "{:<25} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
            "DotVByte <packed, scalaru8>",
            size,
            train + build,
            search,
            recall
        );
    } else {
        println!(
            "{:<25} {:>10} {:>12} {:>12} {:>10}",
            "DotVByte <packed, scalaru8>", "skipped", "-", "-", "-"
        );
    }
}
