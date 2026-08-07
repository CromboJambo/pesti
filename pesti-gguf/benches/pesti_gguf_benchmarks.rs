use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pesti_gguf::parse_gguf;
use std::path::Path;

const MODEL_PATH_3B: &str = "conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf";
const MODEL_PATH_05B: &str = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn bench_pesti_gguf_3b(c: &mut Criterion) {
    c.bench_function("pesti_gguf_parse_3b", |b| {
        b.iter(|| {
            parse_gguf(Path::new(MODEL_PATH_3B)).expect("Failed to parse 3B model");
        });
    });
}

fn bench_pesti_gguf_05b(c: &mut Criterion) {
    c.bench_function("pesti_gguf_parse_0.5b", |b| {
        b.iter(|| {
            parse_gguf(Path::new(MODEL_PATH_05B)).expect("Failed to parse 0.5B model");
        });
    });
}

fn bench_pesti_gguf_kv_extraction(c: &mut Criterion) {
    c.bench_function("pesti_gguf_extract_kv_pairs", |b| {
        let header = parse_gguf(Path::new(MODEL_PATH_3B)).expect("Failed to parse");
        
        b.iter(|| {
            // Extract just KV pair keys (common use case)
            let keys: Vec<&str> = header.kv_pairs.iter().map(|p| p.key.as_str()).collect();
            black_box(keys);
        });
    });
}

fn bench_pesti_gguf_tensor_shapes(c: &mut Criterion) {
    c.bench_function("pesti_gguf_extract_tensor_shapes", |b| {
        let header = parse_gguf(Path::new(MODEL_PATH_3B)).expect("Failed to parse");
        
        b.iter(|| {
            // Extract tensor shapes (common use case for inference servers)
            let shapes: Vec<Vec<u64>> = header.tensors.iter().map(|t| t.shape.clone()).collect();
            black_box(shapes);
        });
    });
}

fn bench_pesti_gguf_dtype_detection(c: &mut Criterion) {
    c.bench_function("pesti_gguf_detect_dtypes", |b| {
        let header = parse_gguf(Path::new(MODEL_PATH_3B)).expect("Failed to parse");
        
        b.iter(|| {
            // Count dtypes (common use case for quantization analysis)
            let dtype_counts: std::collections::HashMap<_, usize> = header
                .tensors
                .iter()
                .map(|t| t.dtype)
                .fold(std::collections::HashMap::new(), |mut acc, dt| {
                    *acc.entry(dt).or_insert(0) += 1;
                    acc
                });
            black_box(dtype_counts);
        });
    });
}

criterion_group!(
    benches,
    bench_pesti_gguf_3b,
    bench_pesti_gguf_05b,
    bench_pesti_gguf_kv_extraction,
    bench_pesti_gguf_tensor_shapes,
    bench_pesti_gguf_dtype_detection,
);

criterion_main!(benches);
