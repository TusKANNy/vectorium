use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use vectorium::core::vector::DenseVectorView;
use vectorium::{QueryEvaluator, ScalarMultiVecQuantizer, VectorEncoder};

const SEED: u64 = 777;

fn main() {
    let token_dim = 128;
    let doc_tokens = 200;
    let query_tokens = 32;

    let mut rng = StdRng::seed_from_u64(SEED);

    // Generate random document multivector: 200 tokens x 128 dims
    let doc_data_f32: Vec<f32> = (0..doc_tokens * token_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let doc_data: Vec<f16> = doc_data_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // Generate random query multivector: 32 tokens x 128 dims
    let query_data: Vec<f32> = (0..query_tokens * token_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let encoder = ScalarMultiVecQuantizer::<f32, f16>::new(token_dim);

    let query = DenseVectorView::new(&query_data);
    let doc = DenseVectorView::new(&doc_data);

    // Warmup
    let evaluator = encoder.query_evaluator(query);
    let _ = evaluator.compute_distance(doc);

    // Timed run
    let iterations = 10_000;
    let start = Instant::now();
    let evaluator = encoder.query_evaluator(query);
    for _ in 0..iterations {
        std::hint::black_box(evaluator.compute_distance(doc));
    }
    let elapsed = start.elapsed();

    let score = evaluator.compute_distance(doc);
    println!("token_dim: {token_dim}");
    println!("doc_tokens: {doc_tokens}, query_tokens: {query_tokens}");
    println!("MaxSim score: {}", score.0);
    println!(
        "Time: {:.2} us/query ({iterations} iterations, {:.2} ms total)",
        elapsed.as_micros() as f64 / iterations as f64,
        elapsed.as_millis()
    );
}
