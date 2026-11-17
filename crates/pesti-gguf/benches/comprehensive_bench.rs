use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pesti_gguf::parse_gguf;
use std::path::Path;

// Real model files for benchmarking
const LLAMA_VOCAB_PATH: &str = "/home/crombo/llama.cpp/models/ggml-vocab-llama-bpe.gguf"; // 7.5M
const COMMAND_R_PATH: &str = "/home/crombo/llama.cpp/models/ggml-vocab-command-r.gguf"; // 11M

fn bench_full_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full GGUF Parse");

    group.bench_with_input(
        BenchmarkId::from_parameter("llama_vocab_7.5m"),
        &Path::new(LLAMA_VOCAB_PATH),
        |b, path| {
            b.iter(|| parse_gguf(path).expect("Failed to parse"))
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("command_r_vocab_11m"),
        &Path::new(COMMAND_R_PATH),
        |b, path| {
            b.iter(|| parse_gguf(path).expect("Failed to parse"))
        },
    );

    group.finish();
}

fn bench_kv_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("KV Pair Extraction");

    // Parse once, then benchmark extraction
    let llama_header = parse_gguf(Path::new(LLAMA_VOCAB_PATH)).expect("Failed to parse llama vocab");
    let command_r_header = parse_gguf(Path::new(COMMAND_R_PATH)).expect("Failed to parse command-r vocab");

    group.bench_with_input(
        BenchmarkId::from_parameter("llama_vocab_7.5m"),
        &llama_header,
        |b, header| {
            b.iter(|| {
                let keys: Vec<&str> = header.kv_pairs.iter().map(|p| p.key.as_str()).collect();
                black_box(keys);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("command_r_vocab_11m"),
        &command_r_header,
        |b, header| {
            b.iter(|| {
                let keys: Vec<&str> = header.kv_pairs.iter().map(|p| p.key.as_str()).collect();
                black_box(keys);
            })
        },
    );

    group.finish();
}

fn bench_tensor_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tensor Extraction");

    let llama_header = parse_gguf(Path::new(LLAMA_VOCAB_PATH)).expect("Failed to parse llama vocab");
    let command_r_header = parse_gguf(Path::new(COMMAND_R_PATH)).expect("Failed to parse command-r vocab");

    group.bench_with_input(
        BenchmarkId::from_parameter("llama_vocab_7.5m"),
        &llama_header,
        |b, header| {
            b.iter(|| {
                let shapes: Vec<Vec<u64>> = header.tensors.iter().map(|t| t.shape.clone()).collect();
                black_box(shapes);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("command_r_vocab_11m"),
        &command_r_header,
        |b, header| {
            b.iter(|| {
                let shapes: Vec<Vec<u64>> = header.tensors.iter().map(|t| t.shape.clone()).collect();
                black_box(shapes);
            })
        },
    );

    group.finish();
}

fn bench_dtype_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dtype Detection");

    let llama_header = parse_gguf(Path::new(LLAMA_VOCAB_PATH)).expect("Failed to parse llama vocab");
    let command_r_header = parse_gguf(Path::new(COMMAND_R_PATH)).expect("Failed to parse command-r vocab");

    group.bench_with_input(
        BenchmarkId::from_parameter("llama_vocab_7.5m"),
        &llama_header,
        |b, header| {
            b.iter(|| {
                let dtype_counts: std::collections::HashMap<_, usize> = header
                    .tensors
                    .iter()
                    .map(|t| t.dtype)
                    .fold(std::collections::HashMap::new(), |mut acc, dt| {
                        *acc.entry(dt).or_insert(0) += 1;
                        acc
                    });
                black_box(dtype_counts);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("command_r_vocab_11m"),
        &command_r_header,
        |b, header| {
            b.iter(|| {
                let dtype_counts: std::collections::HashMap<_, usize> = header
                    .tensors
                    .iter()
                    .map(|t| t.dtype)
                    .fold(std::collections::HashMap::new(), |mut acc, dt| {
                        *acc.entry(dt).or_insert(0) += 1;
                        acc
                    });
                black_box(dtype_counts);
            })
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_full_parse,
    bench_kv_extraction,
    bench_tensor_extraction,
    bench_dtype_detection,
);

criterion_main!(benches);
