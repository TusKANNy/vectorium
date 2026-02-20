use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vectorium::core::vector::DenseVectorView;
use vectorium::{DotProduct, QueryEvaluator, ScalarDenseQuantizer, VectorEncoder};

const SEED: u64 = 42;

/// Benchmark helper: generates random vector data for a given size.
fn generate_random_data(dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Benchmark: Raw dot product computation without evaluator overhead (f32).
fn bench_raw_dot_product_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_dot_product_f32");

    for dim in [128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let query_data = black_box(generate_random_data(dim));
            let doc_data = black_box(generate_random_data(dim));

            let query = DenseVectorView::new(&query_data);
            let doc = DenseVectorView::new(&doc_data);

            b.iter(|| black_box(vectorium::distances::dot_product_dense(query, doc)));
        });
    }

    group.finish();
}

/// Benchmark: Raw dot product computation without evaluator overhead (f16).
fn bench_raw_dot_product_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_dot_product_f16");

    for dim in [128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let query_data_f32 = black_box(generate_random_data(dim));
            let doc_data_f32 = black_box(generate_random_data(dim));

            let query_data: Vec<f16> = query_data_f32.iter().map(|&x| f16::from_f32(x)).collect();
            let doc_data: Vec<f16> = doc_data_f32.iter().map(|&x| f16::from_f32(x)).collect();

            let query = DenseVectorView::new(&query_data);
            let doc = DenseVectorView::new(&doc_data);

            b.iter(|| black_box(vectorium::distances::dot_product_dense(query, doc)));
        });
    }

    group.finish();
}

/// Measure only compute_distance without evaluator creation overhead.
/// Pre-create evaluator outside the loop, measure only the hot path.
fn bench_f32_compute_distance_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_compute_distance_only");

    for dim in [128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let query_data = black_box(generate_random_data(dim));
            let doc_data = black_box(generate_random_data(dim));

            let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(dim);
            let query = DenseVectorView::new(&query_data);
            let doc = DenseVectorView::new(&doc_data);

            // Create evaluator OUTSIDE the loop
            let evaluator = encoder.query_evaluator(query);

            b.iter(|| {
                // Measure ONLY compute_distance, not evaluator creation
                black_box(evaluator.compute_distance(doc))
            });
        });
    }

    group.finish();
}

/// Measure only evaluator creation overhead.
/// This isolates the query_evaluator() construction cost.
fn bench_f32_evaluator_creation_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_evaluator_creation_only");

    for dim in [128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let query_data = black_box(generate_random_data(dim));

            let encoder = black_box(ScalarDenseQuantizer::<f32, f32, DotProduct>::new(dim));
            let query = DenseVectorView::new(&query_data);

            b.iter(|| {
                // Measure ONLY evaluator creation, not distance computation
                black_box(encoder.query_evaluator(query))
            });
        });
    }

    group.finish();
}

/// Measure f16 compute_distance without evaluator creation overhead.
/// Pre-create evaluator outside the loop, measure only the hot path.
fn bench_f16_compute_distance_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("f16_compute_distance_only");

    for dim in [128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let query_data_f32 = black_box(generate_random_data(dim));
            let doc_data_f32 = black_box(generate_random_data(dim));

            let query_data: Vec<f32> = query_data_f32.iter().copied().collect();
            let doc_data: Vec<f16> = doc_data_f32.iter().map(|&x| f16::from_f32(x)).collect();

            let encoder = ScalarDenseQuantizer::<f32, f16, DotProduct>::new(dim);
            let query = DenseVectorView::new(&query_data);
            let doc = DenseVectorView::new(&doc_data);

            // Create evaluator OUTSIDE the loop
            let evaluator = encoder.query_evaluator(query);

            b.iter(|| {
                // Measure ONLY compute_distance, not evaluator creation
                black_box(evaluator.compute_distance(doc))
            });
        });
    }

    group.finish();
}

/// Raw dot product with f32 query and f16 doc.
/// This matches the actual computation in f16_query_evaluator (f32 x f16 dot product).
fn bench_raw_dot_product_f32_query_f16_doc(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_dot_product_f32_query_f16_doc");

    for dim in [128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let query_data_f32 = black_box(generate_random_data(dim));
            let doc_data_f32 = black_box(generate_random_data(dim));

            let query_data: Vec<f32> = query_data_f32.iter().copied().collect();
            let doc_data: Vec<f16> = doc_data_f32.iter().map(|&x| f16::from_f32(x)).collect();

            let query = DenseVectorView::new(&query_data);
            let doc = DenseVectorView::new(&doc_data);

            b.iter(|| black_box(vectorium::distances::dot_product_dense(query, doc)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_raw_dot_product_f32,
    bench_raw_dot_product_f16,
    bench_raw_dot_product_f32_query_f16_doc,
    bench_f32_evaluator_creation_only,
    bench_f32_compute_distance_only,
    bench_f16_compute_distance_only,
);
criterion_main!(benches);
