use pesti_runner::llama::SamplingConfig;
use pesti_runner::{KvCacheType, LlamaRunnerBuilder};
use std::time::Instant;

fn bench_kv_cache(model_path: &str, kv_type: KvCacheType, label: &str) -> f64 {
    println!("=== {} KV Cache Benchmark ===", label);

    // Fresh runner for each benchmark (KV cache position can't reset without rebuild)
    let runner = LlamaRunnerBuilder::new(model_path)
        .n_ctx(1024)
        .kv_cache_type(kv_type)
        .build()
        .expect("Failed to load model");

    // Warmup run - advances KV cache position
    let warmup_config = SamplingConfig {
        max_tokens: 5,
        ..Default::default()
    };
    let _ = runner.generate("Write a short story about ", &warmup_config);

    // Clear KV cache for timed run
    runner.clear_kv_cache().expect("Failed to clear KV cache");

    // Timed run
    let config = SamplingConfig {
        max_tokens: 10,
        ..Default::default()
    };
    let t_gen = Instant::now();
    let result = runner
        .generate("Write a short story about ", &config)
        .expect("Generation failed");
    let gen_time = t_gen.elapsed().as_secs_f64();

    drop(runner); // Free model before next benchmark

    let tok_per_sec = if result.generated_tokens > 0 {
        result.generated_tokens as f64 / gen_time
    } else {
        0.0
    };

    println!(
        "Generated {} tokens in {:.2}s",
        result.generated_tokens, gen_time
    );
    println!(
        "Decode speed: {:.2} tok/s ({:.0}ms/token)",
        tok_per_sec,
        1000.0 / tok_per_sec.max(0.001)
    );
    println!();

    tok_per_sec
}

fn main() {
    let model_path = "./conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== PESTI KV Cache Optimization Benchmark (Phase 5) ===");
    println!("Model: qwen2.5-0.5b-instruct-q4_k_m\n");

    // Benchmark F32 KV cache (baseline)
    let f32_speed = bench_kv_cache(model_path, KvCacheType::F32, "F32");

    // Benchmark F16 KV cache (optimized - half memory bandwidth)
    let f16_speed = bench_kv_cache(model_path, KvCacheType::F16, "F16");

    println!("=== Phase 5 Results ===");
    println!("F32 KV Cache: {:.2} tok/s", f32_speed);
    println!("F16 KV Cache: {:.2} tok/s", f16_speed);

    if f16_speed > f32_speed {
        let improvement = ((f16_speed - f32_speed) / f32_speed * 100.0).max(0.0);
        println!("Improvement: +{:.1}% tok/s with F16 KV cache", improvement);
    } else {
        println!("F16 KV cache did not improve throughput on this model/size");
    }

    println!("\n=== Benchmark Complete ===");
}
