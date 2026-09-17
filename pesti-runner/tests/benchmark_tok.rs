use std::path::Path;
use std::time::Instant;
use rand::SeedableRng;
use pesti_runner::{LlamaModel, SamplingConfig};

#[test]
fn benchmark_tok_per_second() {
    let model_path = Path::new("../conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    
    if !model_path.exists() {
        eprintln!("Model not found: {:?}", model_path);
        return;
    }

    println!("\n=== PESTI tok/s Benchmark ===");
    let start = Instant::now();
    let mut model = LlamaModel::load_gguf(model_path).expect("Failed to load GGUF model");
    println!("Model loaded in {:.1}s", start.elapsed().as_secs_f64());

    let sampling = SamplingConfig { temperature: 0.7, top_k: 50, top_p: 0.9, seed: Some(42) };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Warmup
    model.generate(&[1], 10, &sampling, &mut rng, &[]).ok();

    // Benchmark
    let bench_start = Instant::now();
    let result_tokens = model.generate(&[1], 50, &sampling, &mut rng, &[])
        .expect("Benchmark generation failed");
    let bench_time = bench_start.elapsed().as_secs_f64();

    let tok_per_sec = result_tokens.len() as f64 / bench_time;
    println!("Generated {} tokens in {:.1}s", result_tokens.len(), bench_time);
    println!("Decode speed: {:.2} tok/s", tok_per_sec);
    println!("Time per token: {:.0}ms", 1000.0 / tok_per_sec);
    println!("=== Benchmark Complete ===");
}
