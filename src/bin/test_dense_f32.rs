use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use vectorium::core::vector::DenseVectorView;
use vectorium::{PlainDenseQuantizerDotProduct, QueryEvaluator, VectorEncoder};

const SEED: u64 = 777;

fn main() {
    let dim = 2048;

    let mut rng = StdRng::seed_from_u64(SEED);

    // Generate random document vector
    let doc_data: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Generate random query vector
    let query_data: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let encoder = PlainDenseQuantizerDotProduct::<f32>::new(dim);

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
    println!("DotProduct score: {}", score.0);
    println!(
        "Time: {:.2} us/query ({iterations} iterations, {:.2} ms total)",
        elapsed.as_nanos() as f64 / iterations as f64,
        elapsed.as_micros()
    );
}
