//! Benchmark pesti-runner's own CUDA inference stack (not llama.cpp FFI).
//! Mirrors the llama.cpp generate() flow using pesti-runner's transformer stack.

use std::time::Instant;
use pesti_runner::{LlamaModel, load_gguf_weights};

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "What is the capital of France?";
    let max_tokens = 64;

    // Load weights directly (like llama.cpp's LlamaModel::load_from_file)
    let t_load = Instant::now();
    let weights = load_gguf_weights(model_path).expect("Failed to load GGUF weights");
    let model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Tokenize prompt (like llama.cpp's str_to_token)
    let t_encode = Instant::now();
    let prompt_tokens = model.encode(prompt, true).expect("Failed to encode prompt");
    println!("Encoded {} tokens in {:.2}ms", prompt_tokens.len(), t_encode.elapsed().as_secs_f64() * 1000.0);

    // Prefill: run full forward pass on prompt (like llama.cpp's context.decode(batch))
    let t_prefill = Instant::now();
    let mut hidden = model.embed(&prompt_tokens).expect("Embedding failed");
    for layer in &mut model.layers {
        hidden = layer.forward_with_dispatch(&hidden, 0, None).expect("Layer forward failed");
    }
    println!("Prefill done in {:.2}ms", t_prefill.elapsed().as_secs_f64() * 1000.0);

    // Decode loop: generate tokens one at a time (like llama.cpp's decode loop)
    let mut generated = Vec::new();
    let mut total_decode_time = 0.0f64;

    for i in 0..max_tokens {
        let t_step = Instant::now();

        // Get logits from hidden state (output head GEMM)
        let logits = model.output_head(&hidden).expect("Output head failed");

        // Sample next token (greedy for benchmark consistency)
        let next_token = LlamaModel::argmax_from_logits(&logits);
        generated.push(next_token);

        // Check for EOS
        if model.is_eos_token(next_token) {
            break;
        }

        // Run single decode step: embed token, forward through layers
        let t_forward = Instant::now();
        let token_embed = model.embed_single(next_token).expect("Token embed failed");
        hidden = model.layers[0].forward_with_dispatch(&token_embed, i + prompt_tokens.len(), None)
            .expect("Layer 0 forward failed");
        for layer in &mut model.layers[1..] {
            hidden = layer.forward_with_dispatch(&hidden, i + prompt_tokens.len(), None)
                .expect("Layer forward failed");
        }
        total_decode_time += t_forward.elapsed().as_secs_f64();

        println!("Token {} ({:?}) in {:.2}ms", i + 1, next_token, t_step.elapsed().as_secs_f64() * 1000.0);
    }

    let decode_speed = if total_decode_time > 0.0 {
        generated.len() as f64 / total_decode_time
    } else {
        0.0
    };

    println!("\n=== pesti-runner CUDA Benchmark Results ===");
    println!("Generated {} tokens", generated.len());
    println!("Decode speed: {:.2} tok/s", decode_speed);
    println!("Total decode time: {:.2}s", total_decode_time);
}