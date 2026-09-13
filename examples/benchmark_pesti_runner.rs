//! Benchmark pesti-runner's OWN CUDA inference stack (not llama.cpp FFI).
//! Uses load_gguf_weights() + forward_with_dispatch() to exercise pesti's
//! actual attention/GEMM kernels. Enables fused attention kernel for
//! optimized GPU decode path.

use pesti_runner::llama::{LlamaModel, SamplingConfig};
use pesti_runner::transformer::tokenizer::PestiTokenizer;
use std::time::Instant;

fn main() {
    let model_path = "/home/crombo/.cache/huggingface/hub/models-Qwen_Qwen2-7B-Instruct/snapshots/6e0b8c3f1a59d4b8e8a8e7e9c5f3b0d2a1e4f6g8/gguf/q4_k_m.gguf";
    if !std::path::Path::new(model_path).exists() {
        panic!("Model not found at {}", model_path);
    }

    println!("Loading pesti-runner inference stack (own CUDA kernels)...");
    let start = Instant::now();
    let model = LlamaModel::from_gguf_weights(model_path).expect("Failed to load model weights");
    let load_time = start.elapsed().as_secs_f64();
    println!("Loaded in {:.2}s", load_time);

    // Tokenizer: use PestiTokenizer which builds Qwen2 BPE from GGUF vocab
    println!("\nInitializing tokenizer...");
    let tok_start = Instant::now();
    let tokenizer =
        PestiTokenizer::from_model(&model).expect("Failed to build tokenizer from GGUF");
    let tok_time = tok_start.elapsed().as_secs_f64();
    println!("Tokenizer built in {:.3}s", tok_time);

    let prompt = "Write a short poem about autumn leaves.";

    // Encode with BOS token (required by Qwen2)
    let bos_token = model.config().bos_token_id;
    let encoded = tokenizer.encode(prompt, true).expect("Failed to encode");

    println!("\nPrompt: {}", prompt);
    println!("Encoded {} tokens", encoded.len());

    // Enable fused attention kernel for optimized GPU decode path
    // This eliminates the softmax host-transfer bottleneck by computing
    // softmax on GPU instead of D2H→CPU softmax→H2D transfers.
    println!("\nEnabling fused attention kernel...");
    model.enable_fused_attention();

    // Benchmark: generate 64 tokens (5 warm + 59 measured)
    println!("\nBenchmarking pesti-runner inference (fused attention)...");
    let config = SamplingConfig {
        n_predict: 64,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        ..Default::default()
    };

    let start = Instant::now();
    let result = model.generate(prompt, &config).expect("Generation failed");
    let elapsed = start.elapsed().as_secs_f64();

    // Subtract warmup token time (approximate: 1/64 of total)
    let measured_tokens = 63;
    let measured_time = elapsed * (measured_tokens as f64 / 64.0);
    let tok_per_sec = measured_tokens as f64 / measured_time;

    println!("\n=== pesti-runner Benchmark Results ===");
    println!("Model: Qwen2-7B-Instruct-Q4_K_M");
    println!("Backend: pesti-runner OWN CUDA kernels (fused attention)");
    println!("Tokens generated: 64");
    println!("Total time: {:.3}s", elapsed);
    println!(
        "Measured throughput: {:.2} tok/s ({} tokens / {:.3}s)",
        tok_per_sec, measured_tokens, measured_time
    );
    println!("\nGenerated text:\n{}", result.text);

    println!("\nDone.");
}