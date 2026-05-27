use clap::Parser;
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::dataset::{ConvertInto, ScoredVector};
use vectorium::distances::DotProduct;
use vectorium::encoders::sparse_scalar::ScalarSparseQuantizer;
use vectorium::readers;
use vectorium::{
    Dataset, DatasetGrowable, DotPacking8FixedU8Encoder, DotVByteFixedU8Encoder,
    DotVByteU32FixedU8Encoder, DotVByteU32ScalarU8Encoder, FixedU8Q,
    OptimisticDotPacking8U32ScalarU8Encoder, OptimisticDotVByteFixedU8Encoder,
    OptimisticDotVByteScalarU8Encoder, PackedSparseDataset, PackedSparseDatasetGrowable,
    PlainSparseDataset, SpaceUsage, SparseVectorEncoder,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Compare DotVByte u16 vs u32 component encodings for sparse dot product"
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

/// Build a PackedSparseDataset with the u32 DotVByte encoder from a plain sparse dataset.
fn build_dotvbyte_u32(
    dataset: &PlainSparseDataset<u32, f32, DotProduct>,
) -> PackedSparseDataset<DotVByteU32FixedU8Encoder> {
    let dim = dataset.output_dim();

    let scalar = ScalarSparseQuantizer::<u32, f32, FixedU8Q, DotProduct>::new(dim, dim);

    let mut encoder = DotVByteU32FixedU8Encoder::new(dim, dim);

    const SAMPLE_RATE: usize = 20;
    let sample_size = if dataset.len() / SAMPLE_RATE < 50_000 {
        dataset.len()
    } else {
        dataset.len() / SAMPLE_RATE
    };
    encoder.train(dataset.iter().take(sample_size));

    let mut growable: PackedSparseDatasetGrowable<DotVByteU32FixedU8Encoder> =
        PackedSparseDatasetGrowable::new(encoder);

    for v in dataset.iter() {
        let q_vec = scalar.encode_vector(v);
        growable.push(q_vec.as_view());
    }

    growable.into()
}

/// Build a PackedSparseDataset with the optimistic DotVByte u32 encoder from a plain sparse dataset.
fn build_dotvbyte_optimistic_u32(
    dataset: &PlainSparseDataset<u32, f32, DotProduct>,
) -> PackedSparseDataset<OptimisticDotVByteFixedU8Encoder> {
    let dim = dataset.output_dim();

    let scalar = ScalarSparseQuantizer::<u32, f32, FixedU8Q, DotProduct>::new(dim, dim);

    let mut encoder = OptimisticDotVByteFixedU8Encoder::new(dim, dim);

    const SAMPLE_RATE: usize = 20;
    let sample_size = if dataset.len() / SAMPLE_RATE < 50_000 {
        dataset.len()
    } else {
        dataset.len() / SAMPLE_RATE
    };
    encoder.train(dataset.iter().take(sample_size));

    let mut growable: PackedSparseDatasetGrowable<OptimisticDotVByteFixedU8Encoder> =
        PackedSparseDatasetGrowable::new(encoder);

    for v in dataset.iter() {
        let q_vec = scalar.encode_vector(v);
        growable.push(q_vec.as_view());
    }

    growable.into()
}

/// Build a PackedSparseDataset with the u32 DotVByte scalaru8 encoder from a plain sparse dataset.
fn build_dotvbyte_u32_scalaru8(
    dataset: &PlainSparseDataset<u32, f32, DotProduct>,
    training_data: &PlainSparseDataset<u32, f32, vectorium::SquaredEuclideanDistance>,
) -> PackedSparseDataset<DotVByteU32ScalarU8Encoder> {
    let dim = dataset.output_dim();

    let mut encoder = DotVByteU32ScalarU8Encoder::new(dim, dim);
    encoder.train::<f32>(training_data);

    let mut growable: PackedSparseDatasetGrowable<DotVByteU32ScalarU8Encoder> =
        PackedSparseDatasetGrowable::new(encoder);

    for v in dataset.iter() {
        growable.push(v);
    }

    growable.into()
}

/// Build a PackedSparseDataset with the optimistic u32 DotVByte scalaru8 encoder.
fn build_dotvbyte_optimistic_u32_scalaru8(
    dataset: &PlainSparseDataset<u32, f32, DotProduct>,
    training_data: &PlainSparseDataset<u32, f32, vectorium::SquaredEuclideanDistance>,
) -> PackedSparseDataset<OptimisticDotVByteScalarU8Encoder> {
    let dim = dataset.output_dim();

    let mut encoder = OptimisticDotVByteScalarU8Encoder::new(dim, dim);
    encoder.train::<f32>(training_data);

    let mut growable: PackedSparseDatasetGrowable<OptimisticDotVByteScalarU8Encoder> =
        PackedSparseDatasetGrowable::new(encoder);

    for v in dataset.iter() {
        growable.push(v);
    }

    growable.into()
}

/// Build a PackedSparseDataset with the optimistic u32 DotPacking8 scalaru8 encoder.
fn build_dotpacking8_optimistic_u32_scalaru8(
    dataset: &PlainSparseDataset<u32, f32, DotProduct>,
    training_data: &PlainSparseDataset<u32, f32, vectorium::SquaredEuclideanDistance>,
) -> PackedSparseDataset<OptimisticDotPacking8U32ScalarU8Encoder> {
    let dim = dataset.output_dim();

    let mut encoder = OptimisticDotPacking8U32ScalarU8Encoder::new(dim);
    encoder.train(training_data);

    let mut growable: PackedSparseDatasetGrowable<OptimisticDotPacking8U32ScalarU8Encoder> =
        PackedSparseDatasetGrowable::new(encoder);

    for v in dataset.iter() {
        growable.push(v);
    }

    growable.into()
}

fn main() {
    let args = Args::parse();

    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    // ── Load dataset as f32 with u32 components first ──
    println!("Loading dataset (u32 components)...");
    let dataset_u32: PlainSparseDataset<u32, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");

    println!("Loading queries (u32 components)...");
    let queries_u32: PlainSparseDataset<u32, f32, DotProduct> =
        readers::read_seismic_format(&args.query_file).expect("failed to read queries");
    let training_data_u32: PlainSparseDataset<u32, f32, vectorium::SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file)
            .expect("failed to re-read dataset for scalaru8 training");

    let n = dataset_u32.len();
    let dim = dataset_u32.input_dim();
    let run_dvb_u16 = dim <= u16::MAX as usize;
    let nnz = dataset_u32.nnz();
    let f32_size = dataset_u32.space_usage_GiB();
    let total_n_queries = queries_u32.len();
    let n_queries = args.n_queries.min(total_n_queries);

    println!("Dataset: {n} docs, dim={dim}, nnz={nnz}");
    println!("Using {n_queries} queries out of {total_n_queries}");

    // ── Ground truth: f32 with u32 ──────────────────────────────────
    println!("\n=== f32 ground truth (u32 components) ===");
    println!("f32 size: {f32_size:.3} GiB");
    let start = Instant::now();
    let gt: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_u32.search(queries_u32.get(qi as u64), args.k))
        .collect();
    let search_time_f32 = start.elapsed().as_secs_f64();
    println!("f32 search: {search_time_f32:.3}s");

    // ── DotVByte u16 ────────────────────────────────────────────────
    let (dataset_dvb_u16_size, build_time_u16, search_time_u16, results_u16) = if run_dvb_u16 {
        println!("\n=== DotVByte u16 ===");
        println!("Loading dataset (u16 components)...");
        let dataset_u16: PlainSparseDataset<u16, f32, DotProduct> =
            readers::read_seismic_format(&args.input_file).expect("failed to read dataset");
        println!("Loading queries (u16 components)...");
        let queries_u16: PlainSparseDataset<u16, f32, DotProduct> =
            readers::read_seismic_format(&args.query_file).expect("failed to read queries");

        let start = Instant::now();
        let dataset_dvb_u16: PackedSparseDataset<DotVByteFixedU8Encoder> =
            dataset_u16.convert_into();
        let build_time_u16 = start.elapsed().as_secs_f64();
        let dataset_dvb_u16_size = dataset_dvb_u16.space_usage_GiB();
        println!("Build:  {build_time_u16:.3}s");
        println!("Size:   {:.3} GiB", dataset_dvb_u16_size);

        let start = Instant::now();
        let results_u16: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| dataset_dvb_u16.search(queries_u16.get(qi as u64), args.k))
            .collect();
        let search_time_u16 = start.elapsed().as_secs_f64();
        println!("Search: {search_time_u16:.3}s");
        (
            Some(dataset_dvb_u16_size),
            Some(build_time_u16),
            Some(search_time_u16),
            Some(results_u16),
        )
    } else {
        println!(
            "\n=== DotVByte u16 ===\nSkipping: dataset dim {} exceeds u16 max {}",
            dim,
            u16::MAX
        );
        (None, None, None, None)
    };

    // ── DotVByte u32 ────────────────────────────────────────────────
    println!("\n=== DotVByte u32 ===");
    let start = Instant::now();
    let dataset_dvb_u32 = build_dotvbyte_u32(&dataset_u32);
    let build_time_u32 = start.elapsed().as_secs_f64();
    println!("Build:  {build_time_u32:.3}s");
    println!("Size:   {:.3} GiB", dataset_dvb_u32.space_usage_GiB());

    let start = Instant::now();
    let results_u32: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_dvb_u32.search(queries_u32.get(qi as u64), args.k))
        .collect();
    let search_time_u32 = start.elapsed().as_secs_f64();
    println!("Search: {search_time_u32:.3}s");

    // ── DotVByte optimistic u32 ─────────────────────────────────────
    println!("\n=== DotVByte optimistic u32 ===");
    let start = Instant::now();
    let dataset_dvb_opt_u32 = build_dotvbyte_optimistic_u32(&dataset_u32);
    let build_time_opt_u32 = start.elapsed().as_secs_f64();
    println!("Build:  {build_time_opt_u32:.3}s");
    println!("Size:   {:.3} GiB", dataset_dvb_opt_u32.space_usage_GiB());

    let start = Instant::now();
    let results_opt_u32: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_dvb_opt_u32.search(queries_u32.get(qi as u64), args.k))
        .collect();
    let search_time_opt_u32 = start.elapsed().as_secs_f64();
    println!("Search: {search_time_opt_u32:.3}s");

    // ── DotVByte u32 scalaru8 ───────────────────────────────────────
    println!("\n=== DotVByte u32 scalaru8 ===");
    let start = Instant::now();
    let dataset_dvb_u32_scalaru8 = build_dotvbyte_u32_scalaru8(&dataset_u32, &training_data_u32);
    let build_time_u32_scalaru8 = start.elapsed().as_secs_f64();
    println!("Build:  {build_time_u32_scalaru8:.3}s");
    println!(
        "Size:   {:.3} GiB",
        dataset_dvb_u32_scalaru8.space_usage_GiB()
    );

    let start = Instant::now();
    let results_u32_scalaru8: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_dvb_u32_scalaru8.search(queries_u32.get(qi as u64), args.k))
        .collect();
    let search_time_u32_scalaru8 = start.elapsed().as_secs_f64();
    println!("Search: {search_time_u32_scalaru8:.3}s");

    // ── DotVByte optimistic u32 scalaru8 ────────────────────────────
    println!("\n=== DotVByte optimistic u32 scalaru8 ===");
    let start = Instant::now();
    let dataset_dvb_opt_u32_scalaru8 =
        build_dotvbyte_optimistic_u32_scalaru8(&dataset_u32, &training_data_u32);
    let build_time_opt_u32_scalaru8 = start.elapsed().as_secs_f64();
    println!("Build:  {build_time_opt_u32_scalaru8:.3}s");
    println!(
        "Size:   {:.3} GiB",
        dataset_dvb_opt_u32_scalaru8.space_usage_GiB()
    );

    let start = Instant::now();
    let results_opt_u32_scalaru8: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_dvb_opt_u32_scalaru8.search(queries_u32.get(qi as u64), args.k))
        .collect();
    let search_time_opt_u32_scalaru8 = start.elapsed().as_secs_f64();
    println!("Search: {search_time_opt_u32_scalaru8:.3}s");

    // ── DotPacking8 optimistic u32 scalaru8 ─────────────────────────
    println!("\n=== DotPacking8 optimistic u32 scalaru8 ===");
    let start = Instant::now();
    let dataset_block8_opt_u32_scalaru8 =
        build_dotpacking8_optimistic_u32_scalaru8(&dataset_u32, &training_data_u32);
    let build_time_block8_opt_u32_scalaru8 = start.elapsed().as_secs_f64();
    println!("Build:  {build_time_block8_opt_u32_scalaru8:.3}s");
    println!(
        "Size:   {:.3} GiB",
        dataset_block8_opt_u32_scalaru8.space_usage_GiB()
    );

    let start = Instant::now();
    let results_block8_opt_u32_scalaru8: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_block8_opt_u32_scalaru8.search(queries_u32.get(qi as u64), args.k))
        .collect();
    let search_time_block8_opt_u32_scalaru8 = start.elapsed().as_secs_f64();
    println!("Search: {search_time_block8_opt_u32_scalaru8:.3}s");

    // ── DotPacking8 FixedU8 ─────────────────────────────────────────────
    let (dataset_block8_size, build_time_block8, search_time_block8, results_block8) =
        if run_dvb_u16 {
            println!("\n=== DotPacking8 FixedU8 ===");
            println!("Loading dataset (u16 components)...");
            let dataset_u16: PlainSparseDataset<u16, f32, DotProduct> =
                readers::read_seismic_format(&args.input_file).expect("failed to read dataset");
            println!("Loading queries (u16 components)...");
            let queries_u16: PlainSparseDataset<u16, f32, DotProduct> =
                readers::read_seismic_format(&args.query_file).expect("failed to read queries");

            let start = Instant::now();
            let dataset_block8: PackedSparseDataset<DotPacking8FixedU8Encoder> =
                dataset_u16.convert_into();
            let build_time_block8 = start.elapsed().as_secs_f64();
            let dataset_block8_size = dataset_block8.space_usage_GiB();
            println!("Build:  {build_time_block8:.3}s");
            println!("Size:   {:.3} GiB", dataset_block8_size);

            let start = Instant::now();
            let results_block8: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(pb_style.clone())
                .map(|qi| dataset_block8.search(queries_u16.get(qi as u64), args.k))
                .collect();
            let search_time_block8 = start.elapsed().as_secs_f64();
            println!("Search: {search_time_block8:.3}s");
            (
                Some(dataset_block8_size),
                Some(build_time_block8),
                Some(search_time_block8),
                Some(results_block8),
            )
        } else {
            println!(
                "\n=== Block8 FixedU8 ===\nSkipping: dataset dim {} exceeds u16 max {}",
                dim,
                u16::MAX
            );
            (None, None, None, None)
        };

    // ── Results ────────────────────────────────────────────────────
    let recall_u16 = results_u16
        .as_ref()
        .map(|results| recall_at_k(&gt, results, args.k));
    let recall_u32 = recall_at_k(&gt, &results_u32, args.k);
    let recall_opt_u32 = recall_at_k(&gt, &results_opt_u32, args.k);
    let recall_u32_scalaru8 = recall_at_k(&gt, &results_u32_scalaru8, args.k);
    let recall_opt_u32_scalaru8 = recall_at_k(&gt, &results_opt_u32_scalaru8, args.k);
    let recall_block8_opt_u32_scalaru8 =
        recall_at_k(&gt, &results_block8_opt_u32_scalaru8, args.k);
    let recall_block8 = results_block8
        .as_ref()
        .map(|results| recall_at_k(&gt, results, args.k));

    println!("\n=== Summary ===");
    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>10}",
        "Method", "Size GiB", "Build (s)", "Search (s)", "Recall@k"
    );
    println!("{:-<66}", "");
    println!(
        "{:<20} {:>10.3} {:>12} {:>12.3} {:>10}",
        "f32 (u32)", f32_size, "-", search_time_f32, "1.0000"
    );
    if let (Some(size), Some(build), Some(search), Some(recall)) = (
        dataset_dvb_u16_size,
        build_time_u16,
        search_time_u16,
        recall_u16,
    ) {
        println!(
            "{:<20} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
            "DotVByte u16", size, build, search, recall
        );
    } else {
        println!(
            "{:<20} {:>10} {:>12} {:>12} {:>10}",
            "DotVByte u16", "skipped", "-", "-", "-"
        );
    }
    println!(
        "{:<20} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
        "DotVByte u32",
        dataset_dvb_u32.space_usage_GiB(),
        build_time_u32,
        search_time_u32,
        recall_u32
    );
    println!(
        "{:<20} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
        "DotVByte opt u32",
        dataset_dvb_opt_u32.space_usage_GiB(),
        build_time_opt_u32,
        search_time_opt_u32,
        recall_opt_u32
    );
    println!(
        "{:<20} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
        "DotVByte u32 scalar",
        dataset_dvb_u32_scalaru8.space_usage_GiB(),
        build_time_u32_scalaru8,
        search_time_u32_scalaru8,
        recall_u32_scalaru8
    );
    println!(
        "{:<20} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
        "DotVByte opt u32 s",
        dataset_dvb_opt_u32_scalaru8.space_usage_GiB(),
        build_time_opt_u32_scalaru8,
        search_time_opt_u32_scalaru8,
        recall_opt_u32_scalaru8
    );
    println!(
        "{:<20} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
        "Block8 opt u32 s",
        dataset_block8_opt_u32_scalaru8.space_usage_GiB(),
        build_time_block8_opt_u32_scalaru8,
        search_time_block8_opt_u32_scalaru8,
        recall_block8_opt_u32_scalaru8
    );
    if let (Some(size), Some(build), Some(search), Some(recall)) = (
        dataset_block8_size,
        build_time_block8,
        search_time_block8,
        recall_block8,
    ) {
        println!(
            "{:<20} {:>10.3} {:>12.3} {:>12.3} {:>10.4}",
            "Block8 FixedU8", size, build, search, recall
        );
    } else {
        println!(
            "{:<20} {:>10} {:>12} {:>12} {:>10}",
            "Block8 FixedU8", "skipped", "-", "-", "-"
        );
    }
}
