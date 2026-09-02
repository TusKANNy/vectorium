// Recall / throughput harness for DotPacking8FixedU8 on a sparse dataset.
// No clap: builds with default features.
//
// usage: recall_dotpacking8 <documents.bin> <queries.bin> <groundtruth.tsv> [k] [n_queries] [doc_limit]
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use vectorium::core::index::Index;
use vectorium::core::vector::SparseVectorView;
use vectorium::datasets::readers::read_seismic_format_limit;
use vectorium::{
    DatasetGrowable, Dataset, DotPacking8FixedU8Dataset, DotPacking8FixedU8DatasetGrowable,
    DotPacking8FixedU8Encoder, DotProduct, FlatIndex, PlainSparseDataset, SpaceUsage,
};

type Plain = PlainSparseDataset<u16, f32, DotProduct>;

fn recall(got: &[u64], gt: &[u64]) -> f64 {
    if gt.is_empty() {
        return f64::NAN;
    }
    let hits = vectorium::utils::intersection(got, gt);
    hits as f64 / gt.len() as f64
}

fn topk_ids<I: Index<SearchParams = ()>>(index: &I, q: I::Query<'_>, k: usize) -> Vec<u64> {
    index.search(q, k, &()).iter().map(|s| s.vector).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: {} <documents.bin> <queries.bin> <gt.tsv> [k] [n_queries] [doc_limit]", a[0]);
        std::process::exit(2);
    }
    let k: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(10);
    let n_queries: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(500);
    let doc_limit: Option<usize> = a.get(6).and_then(|s| s.parse().ok());

    let t = Instant::now();
    let docs: Plain = read_seismic_format_limit(&a[1], doc_limit).expect("read documents");
    let queries: Plain = read_seismic_format_limit(&a[2], None).expect("read queries");
    println!(
        "loaded {} docs (dim {}, nnz {}) and {} queries in {:.1}s",
        docs.len(), docs.output_dim(), docs.nnz(), queries.len(), t.elapsed().as_secs_f32()
    );

    // ---- value distribution: FixedU8Q saturates above 4.0 (6 fractional bits) ----
    let mut mx = 0f32;
    let mut over = 0usize;
    let mut total = 0usize;
    for v in docs.iter() {
        for &x in v.values() {
            if x > mx { mx = x; }
            if x >= 4.0 { over += 1; }
            total += 1;
        }
    }
    println!(
        "doc values: max {:.4}, {} / {} (={:.4}%) >= 4.0 (FixedU8Q saturation point)",
        mx, over, total, 100.0 * over as f64 / total as f64
    );

    let nq = n_queries.min(queries.len());
    let nnz = docs.nnz();

    // ---- ground truth from file ----
    let gt_txt = std::fs::read_to_string(&a[3]).expect("read gt");
    let mut gt_file: HashMap<usize, Vec<u64>> = HashMap::new();
    for line in gt_txt.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 2 { continue; }
        let (qid, did) = (f[0].parse::<usize>().unwrap(), f[1].parse::<u64>().unwrap());
        let e = gt_file.entry(qid).or_default();
        if e.len() < k { e.push(did); }
    }

    // ---- exact f32 baseline over the same collection ----
    let exact = FlatIndex::from(docs);
    let t = Instant::now();
    let exact_top: Vec<Vec<u64>> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let q = queries.get(i as u64);
            topk_ids(&exact, SparseVectorView::new(q.components(), q.values()), k)
        })
        .collect();
    let exact_ms = t.elapsed().as_secs_f64() * 1000.0 / nq as f64;

    // Sanity: our exact f32 search must reproduce the shipped ground truth.
    let mut agree = 0.0;
    let mut n_gt = 0;
    for i in 0..nq {
        if let Some(g) = gt_file.get(&i) {
            agree += recall(&exact_top[i], g);
            n_gt += 1;
        }
    }
    println!(
        "exact f32 search: {:.1} ms/query | agreement with gt file: {:.4} over {} queries",
        exact_ms, agree / n_gt as f64, n_gt
    );

    let docs = exact.into_dataset();
    let plain_bytes = docs.space_usage_bytes();
    println!(
        "plain f32 sparse dataset: {:.1} MiB ({:.2} bits/nnz)",
        plain_bytes as f64 / (1024.0 * 1024.0),
        8.0 * plain_bytes as f64 / docs.nnz() as f64
    );

    // ---- DotPacking8, untrained and trained ----
    for trained in [false, true] {
        let mut enc = DotPacking8FixedU8Encoder::new(docs.output_dim());
        if trained {
            // NOTE: the public `train` takes a SquaredEuclideanDistance dataset, which a
            // DotProduct collection cannot satisfy; go through train_components directly.
            enc.train_components(docs.iter());
        }
        let t = Instant::now();
        let mut g = DotPacking8FixedU8DatasetGrowable::new(enc);
        for v in docs.iter() { g.push(v); }
        let packed: DotPacking8FixedU8Dataset = g.into();
        let build_s = t.elapsed().as_secs_f32();
        let bytes = packed.space_usage_bytes();
        let index = FlatIndex::from(packed);

        let t = Instant::now();
        let got: Vec<Vec<u64>> = (0..nq)
            .into_par_iter()
            .map(|i| {
                let q = queries.get(i as u64);
                topk_ids(&index, SparseVectorView::new(q.components(), q.values()), k)
            })
            .collect();
        let ms = t.elapsed().as_secs_f64() * 1000.0 / nq as f64;

        let r_exact: f64 = (0..nq).map(|i| recall(&got[i], &exact_top[i])).sum::<f64>() / nq as f64;
        let r_file: f64 = {
            let mut s = 0.0; let mut n = 0;
            for i in 0..nq {
                if let Some(g) = gt_file.get(&i) { s += recall(&got[i], g); n += 1; }
            }
            if n == 0 { f64::NAN } else { s / n as f64 }
        };
        println!(
            "DotPacking8FixedU8 {:9}: recall@{k} vs exact {:.4} | vs gt-file {:.4} | {:.1} ms/query ({:.2}x) | {:.1} MiB ({:.2} bits/nnz, {:.2}x smaller) | build {:.1}s",
            if trained { "TRAINED" } else { "untrained" },
            r_exact, r_file, ms, exact_ms / ms,
            bytes as f64 / (1024.0 * 1024.0),
            8.0 * bytes as f64 / nnz as f64,
            plain_bytes as f64 / bytes as f64,
            build_s
        );
    }
}
