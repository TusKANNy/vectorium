use half::f16;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

use vectorium::core::vector::DenseVectorView;
use vectorium::{DotProduct, ScalarDenseQuantizer, QueryEvaluator, VectorEncoder};

const SEED: u64 = 777;

fn main() {
    let dim = 2048;

    let mut rng = StdRng::seed_from_u64(SEED);

    // Generate random document vector in f16
    let doc_data_f32: Vec<f32> = (0..dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let doc_data: Vec<f16> = doc_data_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // Generate random query vector in f32
    let query_data: Vec<f32> = (0..dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Encoder: f32 query -> f16 documents
    let encoder = ScalarDenseQuantizer::<f32, f16, DotProduct>::new(dim);

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
    println!("dim: {dim}");
    println!("document type: f16");
    println!("DotProduct score: {}", score.0);
    println!(
        "Time: {:.2} ns/query ({iterations} iterations, {:.2} us total)",
        elapsed.as_nanos() as f64 / iterations as f64,
        elapsed.as_micros()
    );
}
