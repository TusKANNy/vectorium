use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use vectorium::core::vector::DenseMultiVectorView;
use vectorium::{Dataset, DatasetGrowable, MultiVectorDatasetGrowable, PlainMultiVecQuantizer};

const SEED: u64 = 42;
const N_DOCS: usize = 200;
const MIN_TOKENS: usize = 50;
const MAX_TOKENS: usize = 90;
const TOKEN_DIM: usize = 128;
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

    println!("Benchmark: multivector dataset (f32)");
    println!(
        "  {} docs, token lengths [{}, {}], token_dim={}, total tokens={}",
        N_DOCS, MIN_TOKENS, MAX_TOKENS, TOKEN_DIM, total_tokens
    );
    println!(
        "  {} queries x {} query tokens",
        N_QUERIES, QUERY_TOKENS
    );
    println!();

    // --- Generate flat document data: total_tokens * TOKEN_DIM scalars ---
    let flat_data: Vec<f32> = (0..total_tokens * TOKEN_DIM)
        .map(|_| rng.gen_range(-1.0_f32..1.0))
        .collect();

    // --- Build dataset ---
    let encoder = PlainMultiVecQuantizer::<f32>::new(TOKEN_DIM);
    let mut dataset = MultiVectorDatasetGrowable::new(encoder);

    let build_start = Instant::now();
    let mut offset = 0;
    for &n_tokens in &doc_lengths {
        let end = offset + n_tokens * TOKEN_DIM;
        dataset.push(DenseMultiVectorView::new(&flat_data[offset..end], TOKEN_DIM));
        offset = end;
    }
    let build_elapsed = build_start.elapsed();

    let dataset = dataset;

    // --- Generate queries ---
    let query_len = QUERY_TOKENS * TOKEN_DIM;
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|_| {
            (0..query_len)
                .map(|_| rng.gen_range(-1.0_f32..1.0))
                .collect()
        })
        .collect();

    // --- Warmup: run each query once ---
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

    let avg_query_us = total_query_ns as f64 / (N_QUERIES as f64 * iterations as f64) / 1_000.0;

    // --- Plot: one row per query, columns = top-k ranked results ---
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
    println!("Construction time : {:.2} ms", build_elapsed.as_secs_f64() * 1_000.0);
    println!("Avg query time    : {:.2} µs  ({iterations} iterations × {N_QUERIES} queries)", avg_query_us);
}
