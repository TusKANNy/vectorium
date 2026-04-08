use clap::Parser;

use vectorium::distances::DotProduct;
use vectorium::utils::permute_components_with_bisection;
use vectorium::{Dataset, PlainSparseDataset};
use vectorium::readers;

#[derive(Parser, Debug)]
#[clap(author, version, about = "Analyze delta distributions in sparse vectors with and without component reordering")]
struct Args {
    /// Sparse dataset in Seismic binary format
    #[clap(short, long)]
    input_file: String,

    /// Sample rate: use 1/N of the dataset for bisection training (0 = use all)
    #[clap(long, default_value_t = 20)]
    sample_rate: usize,
}

struct DeltaStats {
    total_deltas: u64,
    one_byte: u64,    // delta <= 255
    two_byte: u64,    // 256 <= delta <= 65535
    larger: u64,      // delta > 65535
    max_delta: u64,
}

impl DeltaStats {
    fn new() -> Self {
        Self {
            total_deltas: 0,
            one_byte: 0,
            two_byte: 0,
            larger: 0,
            max_delta: 0,
        }
    }

    fn add(&mut self, delta: u64) {
        self.total_deltas += 1;
        if delta <= 255 {
            self.one_byte += 1;
        } else if delta <= 65535 {
            self.two_byte += 1;
        } else {
            self.larger += 1;
        }
        self.max_delta = self.max_delta.max(delta);
    }

    fn print(&self, label: &str) {
        println!("\n=== {} ===", label);
        println!("Total deltas:     {}", self.total_deltas);
        println!("1-byte (<=255):   {} ({:.4}%)", self.one_byte, 100.0 * self.one_byte as f64 / self.total_deltas as f64);
        println!("2-byte (256..=65535): {} ({:.4}%)", self.two_byte, 100.0 * self.two_byte as f64 / self.total_deltas as f64);
        println!(">2-byte (>65535): {} ({:.4}%)", self.larger, 100.0 * self.larger as f64 / self.total_deltas as f64);
        println!("Max delta:        {}", self.max_delta);
    }
}

fn main() {
    let args = Args::parse();

    println!("Loading dataset...");
    let dataset: PlainSparseDataset<u32, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");

    let n = dataset.len();
    let dim = dataset.input_dim();
    println!("Dataset: {} vectors, dim={}", n, dim);

    // Compute bisection permutation
    println!("Computing bisection permutation...");
    let sample_size = if args.sample_rate == 0 || n / args.sample_rate < 50_000 {
        n
    } else {
        n / args.sample_rate
    };

    let permutation = permute_components_with_bisection::<u32, _>(
        dim,
        dataset.iter().take(sample_size).map(|v| v.components().to_vec()),
    );

    // Analyze deltas
    println!("Analyzing deltas...");
    let mut stats_no_reorder = DeltaStats::new();
    let mut stats_reordered = DeltaStats::new();

    // Also track per-position stats: what fraction of "first delta in a chunk of 8" are >255
    // This tells us about the SIMD pack structure
    let mut per_position_no_reorder = [0u64; 8];
    let mut per_position_reordered = [0u64; 8];
    let mut per_position_total = [0u64; 8];

    for i in 0..n {
        let vec = dataset.get(i as u64);
        let components = vec.components();

        if components.is_empty() {
            continue;
        }

        // Without reordering: components are already sorted
        let mut prev = 0u32;
        for (j, &c) in components.iter().enumerate() {
            let delta = (c - prev) as u64;
            stats_no_reorder.add(delta);
            let pos = j % 8;
            per_position_total[pos] += 1;
            if delta > 255 {
                per_position_no_reorder[pos] += 1;
            }
            prev = c;
        }

        // With reordering: remap then sort
        let mut remapped: Vec<u32> = components.iter().map(|&c| permutation[c as usize] as u32).collect();
        remapped.sort_unstable();

        let mut prev = 0u32;
        for (j, &c) in remapped.iter().enumerate() {
            let delta = (c - prev) as u64;
            stats_reordered.add(delta);
            let pos = j % 8;
            if delta > 255 {
                per_position_reordered[pos] += 1;
            }
            prev = c;
        }
    }

    stats_no_reorder.print("Without component reordering");
    stats_reordered.print("With bisection reordering");

    println!("\n=== Per-position in SIMD pack (fraction >1 byte) ===");
    println!("{:>4}  {:>12}  {:>12}  {:>12}", "Pos", "No reorder", "Reordered", "Total");
    for pos in 0..8 {
        let total = per_position_total[pos];
        if total == 0 { continue; }
        println!(
            "{:>4}  {:>11.4}%  {:>11.4}%  {:>12}",
            pos,
            100.0 * per_position_no_reorder[pos] as f64 / total as f64,
            100.0 * per_position_reordered[pos] as f64 / total as f64,
            total,
        );
    }

    // Histogram of delta sizes
    println!("\n=== Delta size histogram (without reordering) ===");
    print_histogram(&stats_no_reorder, &dataset, &None);

    println!("\n=== Delta size histogram (with reordering) ===");
    print_histogram(&stats_reordered, &dataset, &Some(permutation));
}

fn print_histogram(
    _stats: &DeltaStats,
    dataset: &PlainSparseDataset<u32, f32, DotProduct>,
    permutation: &Option<Box<[usize]>>,
) {
    let n = dataset.len();
    let buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    let mut counts = vec![0u64; buckets.len() + 1];

    for i in 0..n {
        let vec = dataset.get(i as u64);
        let components = vec.components();
        if components.is_empty() { continue; }

        let sorted: Vec<u32> = if let Some(perm) = permutation {
            let mut remapped: Vec<u32> = components.iter().map(|&c| perm[c as usize] as u32).collect();
            remapped.sort_unstable();
            remapped
        } else {
            components.to_vec()
        };

        let mut prev = 0u32;
        for &c in &sorted {
            let delta = c - prev;
            let bucket = buckets.iter().position(|&b| (delta as u64) < b).unwrap_or(buckets.len());
            counts[bucket] += 1;
            prev = c;
        }
    }

    println!("{:>10}  {:>12}  {:>8}", "Range", "Count", "%");
    let total: u64 = counts.iter().sum();
    let mut cumulative = 0u64;
    for (i, &count) in counts.iter().enumerate() {
        cumulative += count;
        let range = if i == 0 {
            format!("[0, {})", buckets[0])
        } else if i < buckets.len() {
            format!("[{}, {})", buckets[i - 1], buckets[i])
        } else {
            format!("[{}, ...)", buckets[buckets.len() - 1])
        };
        println!(
            "{:>10}  {:>12}  {:>7.3}%  cum {:>7.3}%",
            range,
            count,
            100.0 * count as f64 / total as f64,
            100.0 * cumulative as f64 / total as f64,
        );
    }
}
