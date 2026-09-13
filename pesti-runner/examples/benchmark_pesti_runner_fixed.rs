//! Benchmark pesti-runner's own CUDA inference stack (not llama.cpp FFI).
//! Uses the same model/prompt as the llama.cpp benchmark for fair comparison.

use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    if !Path::new(model_path).exists() {
        eprintln!("Model not found at {}", model_path);
        std::process::exit(1);
    }

    println!("Loading pesti-runner inference stack...");
    let start = Instant::now();
    let model = pesti_runner::LlamaRunnerBuilder::new(model_path)
        .set_gpu_layers(999)
        .build()
        .expect("Failed to load model");
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

    let tokenizer = model.tokenizer();
    let prompt = "What is the meaning of life?";
    let prompt_tokens = tokenizer.tokenize(prompt).unwrap();
    println!("Prompt: '{}'", prompt);
    println!("Prompt tokens: {}", prompt_tokens.len());

    // Prefill: run full forward pass on prompt
    let t_prefill = Instant::now();
    let mut hidden = model.embedding(&prompt_tokens[0]).expect("Embedding failed");
    for layer in &mut model.layers {
        hidden = layer.forward_with_cache(&hidden, &mut layer.cache, 0);
    }
    println!("Prefill done in {:.2}ms", t_prefill.elapsed().as_secs_f64() * 1000.0);

    // Decode loop: generate tokens one at a time
    let mut generated = Vec::new();
    let total_decode_time = Instant::now();
    let max_tokens = 32;

    for i in 0..max_tokens {
        let t_step = Instant::now();

        // Get logits from hidden state
        let logits = model.logits(&hidden).expect("Logits failed");

        // Sample next token (greedy)
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;

        // Check for EOS
        let piece = tokenizer.decode(&[next_token]).expect("Failed to decode token");
        if piece == "< |endoftext|> " {
            println!("EOS reached at token {}", i + 1);
            break;
        }

        generated.push(piece.clone());
        print!("{}", piece);
        std::io::flush_stdout().ok();

        let step_time = t_step.elapsed().as_secs_f64();
        if i == 0 {
            println!("\nFirst decode step: {:.2}ms", step_time * 1000.0);
        }

        // Embed next token and continue decoding
        let token_embed = model.embedding(&[next_token]).expect("Token embed failed");
        hidden = layer.forward_with_cache(&token_embed, &mut layer.cache, i + prompt_tokens.len());
    }

    println!("\n\nGenerated text: {}", generated.join(""));
    let decode_secs = total_decode_time.elapsed().as_secs_f64();
    let tokens_generated = generated.len();
    let tok_per_sec = if decode_secs > 0.0 { tokens_generated as f64 / decode_secs } else { 0.0 };

    println!("\n=== pesti-runner Benchmark Results ===");
    println!("Model: {}", model_path);
    println!("GPU layers: all (999)");
    println!("Tokens generated: {}", tokens_generated);
    println!("Decode time: {:.3}s", decode_secs);
    println!("Average tok/s: {:.2}", tok_per_sec);
}