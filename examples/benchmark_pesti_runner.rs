//! Benchmark pesti-runner's own CUDA inference stack (not llama.cpp FFI).
//! Mirrors the llama.cpp generate() flow using pesti-runner's transformer stack.

use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    if !Path::new(model_path).exists() {
        eprintln!("Model not found: {}", model_path);
        std::process::exit(1);
    }

    println!("Loading GGUF weights from: {}", model_path);
    let load_start = Instant::now();
    let weights = pesti_runner::load_gguf_weights(model_path).expect("Failed to load GGUF weights");
    let load_time = load_start.elapsed();
    println!("Weights loaded in {:.2}s", load_time.as_secs_f64());

    println!("Building model from weights...");
    let build_start = Instant::now();
    let model = pesti_runner::LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    let build_time = build_start.elapsed();
    println!("Model built in {:.2}s", build_time.as_secs_f64());

    // Load tokenizer from same directory as model (tokenizer.json)
    let model_dir = Path::new(model_path)
        .parent()
        .expect("Model has no parent dir");
    let tokenizer_path = model_dir.join("tokenizer.json");
    println!("Loading tokenizer from: {}", tokenizer_path.display());
    let tokenizer = pesti_runner::PestiTokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");

    // Test prompt - same as llama.cpp benchmark
    let prompt = "The quick brown fox jumps over the lazy dog. ";
    println!("Encoding prompt...");
    let input_ids = tokenizer.encode(prompt).expect("Failed to encode prompt");
    println!("Encoded {} tokens", input_ids.len());

    // Warmup: one forward pass
    println!("Running warmup pass...");
    let mut hidden = model.embedding(&input_ids[0]).expect("Failed embedding");
    for i in 1..input_ids.len() {
        let _ = model.forward_layers_with_cache(&hidden, i as i32).expect("Forward failed");
    }
    let logits = model.apply_output_head(&hidden);

    // Find argmax token
    let mut best_idx = 0;
    let mut best_val = f32::MIN;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }

    println!("Warmup complete. Next token: {}", best_idx);

    // Benchmark: 10 decode steps
    let num_decode_steps = 10;
    println!("\nBenchmarking {} decode steps...", num_decode_steps);
    let bench_start = Instant::now();

    for step in 0..num_decode_steps {
        let pos = input_ids.len() + step;
        hidden = model.embedding(&best_idx).expect("Failed embedding");
        hidden = model.forward_layers_with_cache(&hidden, pos as i32).expect("Forward failed");
        let logits = model.apply_output_head(&hidden);

        // Greedy decode
        let mut best_idx_next = 0;
        let mut best_val = f32::MIN;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx_next = i;
            }
        }
        best_idx = best_idx_next;

        // Decode token to text
        let piece = tokenizer.decode(&[best_idx]).expect("Failed decode");
        print!("{}", piece);
    }
    println!();

    let bench_time = bench_start.elapsed();
    let avg_ms = bench_time.as_secs_f64() * 1000.0 / num_decode_steps as f64;
    let tok_s = 1000.0 / avg_ms;

    println!("\n=== Benchmark Results ===");
    println!("Decode steps: {}", num_decode_steps);
    println!("Total time: {:.3}s", bench_time.as_secs_f64());
    println!("Avg per token: {:.2}ms", avg_ms);
    println!("Throughput: {:.1} tok/s", tok_s);

    // Cleanup KV caches
    model.reset_cpu_kv_caches();
}