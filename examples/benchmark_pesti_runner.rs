//! Benchmark pesti-runner's own CUDA inference stack (not llama.cpp FFI).
//! Uses LlamaModel::forward_layers_with_cache() for autoregressive decoding.

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
    let mut model = pesti_runner::LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    let build_time = build_start.elapsed();
    println!("Model built in {:.2}s", build_time.as_secs_f64());

    // Use the model's built-in tokenizer if available, otherwise fall back
    let prompt = "The quick brown fox jumps over the lazy dog. ";
    
    // Encode using llama.cpp FFI tokenizer (same as reference)
    println!("Encoding prompt via llama.cpp FFI...");
    let ctx = pesti_runner::llama::LlamaRunner::builder(model_path)
        .n_ctx(512)
        .build()
        .expect("Failed to create llama context for tokenization");
    
    let input_ids = ctx.encode(prompt, true).expect("Failed to encode prompt");
    println!("Encoded {} tokens", input_ids.len());

    // Warmup: process entire prompt through model layers with KV cache
    println!("Running warmup pass...");
    for (i, &token_id) in input_ids.iter().enumerate() {
        let emb = model.embed(token_id, i).expect("Failed embed");
        let _hidden = model.forward_layers_with_cache(&emb, i).expect("Warmup forward failed");
    }
    println!("Warmup complete.");

    // Benchmark: 10 decode steps using pesti-runner's forward pass
    let num_decode_steps = 10;
    println!("\nBenchmarking {} decode steps...", num_decode_steps);
    
    let bench_start = Instant::now();
    let mut last_token = input_ids[0];
    
    for step in 0..num_decode_steps {
        let pos = input_ids.len() + step;

        // Embed the token
        let emb = model.embed(last_token, pos).expect("Failed embed");

        // Forward through all layers with KV cache
        let hidden = model.forward_layers_with_cache(&emb, pos).expect("Forward failed");

        // Apply output head to get logits
        let logits = model.apply_output_head(&hidden).expect("Logits failed");

        // Greedy decode: find argmax
        let mut best_idx_next = 0;
        let mut best_val = f32::MIN;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx_next = i;
            }
        }
        last_token = best_idx_next as u32;

        // Decode token to text via llama.cpp FFI
        let piece = ctx.token_to_piece(last_token).expect("Failed decode");
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
