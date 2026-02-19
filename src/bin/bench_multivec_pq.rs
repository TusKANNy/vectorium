use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use vectorium::{
    Dataset, DatasetGrowable, DenseMultiVectorView, DenseVectorView, MultiVectorDataset,
    MultivecProductQuantizer, PlainDenseDatasetGrowable, PlainDenseQuantizer,
    SquaredEuclideanDistance,
};

const SEED: u64 = 42;
const N_DOCS: usize = 200;
const MIN_TOKENS: usize = 50;
const MAX_TOKENS: usize = 90;
const TOKEN_DIM: usize = 128;
const M: usize = 16; // PQ subspaces (dsub = TOKEN_DIM / M = 4)
const N_QUERIES: usize = 5;
const QUERY_TOKENS: usize = 32;
const TOP_K: usize = 5;

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    // --- Generate document token lengths ---
    let doc_lengths: Vec<usize> = (0..N_DOCS)
        .map(|_| rng.gen_range(MIN_TOKENS..=MAX_TOKENS))
        .collect();

    let total_tokens: usize = doc_lengths.iter().sum();

    println!("Benchmark: multivector PQ dataset (f32)");
    println!(
        "  {} docs, token lengths [{}, {}], token_dim={}, M={} (dsub={}), total tokens={}",
        N_DOCS,
        MIN_TOKENS,
        MAX_TOKENS,
        TOKEN_DIM,
        M,
        TOKEN_DIM / M,
        total_tokens
    );
    println!("  {} queries x {} query tokens, top-{}", N_QUERIES, QUERY_TOKENS, TOP_K);
    println!();

    // --- Generate flat document data: total_tokens * TOKEN_DIM scalars ---
    let flat_data: Vec<f32> = (0..total_tokens * TOKEN_DIM)
        .map(|_| rng.gen_range(-1.0_f32..1.0))
        .collect();

    // --- Build training set from all token vectors ---
    let train_start = Instant::now();
    let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(TOKEN_DIM);
    let mut training_ds =
        PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
            quantizer,
            total_tokens,
        );
    for token in flat_data.chunks(TOKEN_DIM) {
        training_ds.push(DenseVectorView::new(token));
    }
    let training_ds = training_ds.into();

    // --- Train MultivecProductQuantizer ---
    let encoder = MultivecProductQuantizer::<M, f32>::train(&training_ds);
    let train_elapsed = train_start.elapsed();

    // --- Build PQ-encoded dataset in parallel ---
    let build_start = Instant::now();
    let dataset =
        MultiVectorDataset::from_flat_par(encoder, &flat_data, &doc_lengths);
    let build_elapsed = build_start.elapsed();

    // --- Generate queries ---
    let query_len = QUERY_TOKENS * TOKEN_DIM;
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|_| (0..query_len).map(|_| rng.gen_range(-1.0_f32..1.0)).collect())
        .collect();

    // --- Warmup: one pass through all queries ---
    for q in &queries {
        std::hint::black_box(dataset.search(DenseMultiVectorView::new(q, TOKEN_DIM), TOP_K));
    }

    // --- Timed search: 100 iterations per query ---
    let iterations = 100_u64;
    let mut all_results = Vec::with_capacity(N_QUERIES);
    let mut total_query_ns: u64 = 0;

    for q in &queries {
        let query_view = DenseMultiVectorView::new(q.as_slice(), TOKEN_DIM);
        let start = Instant::now();
        let mut last_results = Vec::new();
        for _ in 0..iterations {
            last_results = std::hint::black_box(dataset.search(query_view, TOP_K));
        }
        total_query_ns += start.elapsed().as_nanos() as u64;
        all_results.push(last_results);
    }

    let avg_query_us =
        total_query_ns as f64 / (N_QUERIES as f64 * iterations as f64) / 1_000.0;

    // --- Print results table ---
    let col_width = 18;

    print!("{:<6}", "Query");
    for rank in 1..=TOP_K {
        print!(" | {:<col_width$}", format!("Rank {rank}"), col_width = col_width);
    }
    println!();
    print!("{}", "-".repeat(6));
    for _ in 0..TOP_K {
        print!("-+-{}", "-".repeat(col_width));
    }
    println!();

    for (qi, results) in all_results.iter().enumerate() {
        print!("Q{:<5}", qi);
        for scored in results {
            let cell = format!("doc{:03} ({:+.4})", scored.vector, scored.distance.0);
            print!(" | {:<col_width$}", cell, col_width = col_width);
        }
        println!();
    }

    println!();
    println!("Training time     : {:.2} ms", train_elapsed.as_secs_f64() * 1_000.0);
    println!("Construction time : {:.2} ms", build_elapsed.as_secs_f64() * 1_000.0);
    println!(
        "Avg query time    : {:.2} µs  ({iterations} iterations × {N_QUERIES} queries)",
        avg_query_us
    );
}
