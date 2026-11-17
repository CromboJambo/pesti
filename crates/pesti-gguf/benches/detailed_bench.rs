use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pesti_gguf::parse_gguf;
use std::path::Path;

// Real model files for benchmarking
const LLAMA_VOCAB_PATH: &str = "/home/crombo/llama.cpp/models/ggml-vocab-llama-bpe.gguf"; // 7.5M

fn bench_full_parse_with_timing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full GGUF Parse with Timing");

    group.bench_function(BenchmarkId::from_parameter("pesti_parse_7.5m"), |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            parse_gguf(Path::new(LLAMA_VOCAB_PATH)).expect("Failed to parse");
            let duration = start.elapsed();
            black_box(duration);
        })
    });

    group.finish();
}

fn bench_parse_repeated(c: &mut Criterion) {
    // Parse once, then measure repeated parsing of the same in-memory data
    // This isolates parsing from file I/O
    
    let mut group = c.benchmark_group("Repeated Parse (I/O isolated)");

    // First parse to warm up and get the header
    let _first_parse = parse_gguf(Path::new(LLAMA_VOCAB_PATH)).expect("Failed to parse");

    group.bench_function("pesti_warmup", |b| {
        b.iter(|| {
            // This measures parsing with OS cache (file already in memory)
            let start = std::time::Instant::now();
            parse_gguf(Path::new(LLAMA_VOCAB_PATH)).expect("Failed to parse");
            let duration = start.elapsed();
            black_box(duration);
        })
    });

    group.finish();
}

fn bench_memory_operations(c: &mut Criterion) {
    // Benchmark the memory operations after parsing (no disk I/O)
    
    let header = parse_gguf(Path::new(LLAMA_VOCAB_PATH)).expect("Failed to parse");

    let mut group = c.benchmark_group("In-Memory Operations");

    // Count KV pairs
    group.bench_function("count_kv_pairs", |b| {
        b.iter(|| {
            let count = header.kv_pairs.len();
            black_box(count);
        })
    });

    // Extract all keys
    group.bench_function("extract_all_keys", |b| {
        b.iter(|| {
            let keys: Vec<&str> = header.kv_pairs.iter().map(|p| p.key.as_str()).collect();
            black_box(keys);
        })
    });

    // Count tensors
    group.bench_function("count_tensors", |b| {
        b.iter(|| {
            let count = header.tensors.len();
            black_box(count);
        })
    });

    // Extract all tensor names
    group.bench_function("extract_all_tensor_names", |b| {
        b.iter(|| {
            let names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
            black_box(names);
        })
    });

    group.finish();
}

fn bench_metadata_queries(c: &mut Criterion) {
    // Benchmark common metadata queries
    
    let header = parse_gguf(Path::new(LLAMA_VOCAB_PATH)).expect("Failed to parse");

    let mut group = c.benchmark_group("Metadata Queries");

    // Find specific key
    group.bench_function("find_architecture_key", |b| {
        b.iter(|| {
            let result = header
                .kv_pairs
                .iter()
                .find(|p| p.key == "general.architecture");
            black_box(result.is_some());
        })
    });

    // Get all unique dtypes
    group.bench_function("collect_unique_dtypes", |b| {
        b.iter(|| {
            let mut dtypes = std::collections::HashSet::new();
            for tensor in &header.tensors {
                dtypes.insert(tensor.dtype);
            }
            black_box(dtypes.len());
        })
    });

    // Calculate total parameter count (sum of all tensor element counts)
    group.bench_function("calculate_total_elements", |b| {
        b.iter(|| {
            let total: u64 = header
                .tensors
                .iter()
                .map(|t| t.shape.iter().copied().product::<u64>())
                .sum();
            black_box(total);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_full_parse_with_timing,
    bench_parse_repeated,
    bench_memory_operations,
    bench_metadata_queries,
);

criterion_main!(benches);
