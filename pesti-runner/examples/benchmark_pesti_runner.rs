//! Benchmark pesti-runner's own CUDA inference stack (not llama.cpp FFI).
//! Mirrors the llama.cpp generate() flow using pesti-runner's transformer stack.

use pesti_runner::transformer::{SamplingConfig, LlamaModel};
use pesti_runner::gguf_weight_loader::load_gguf_weights;
use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "What is the capital of France?";

    println!("Loading model from {}", model_path);
    let t_load = Instant::now();
    let weights = load_gguf_weights(Path::new(model_path)).expect("Failed to load GGUF weights");
    let mut model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Tokenize prompt using the tokenizer directly
    let tokenizer = model.tokenizer.as_ref().expect("No tokenizer");
    let t_encode = Instant::now();
    let prompt_tokens = tokenizer.encode(prompt).expect("Failed to encode prompt");
    println!(
        "Encoded {} tokens in {:.2}ms",
        prompt_tokens.len(),
        t_encode.elapsed().as_secs_f64() * 1000.0
    );

    // Prefill: run each prompt token through the model at its position so
    // the KV cache contains the full prompt context.
    let mut logits: Vec<f32> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i).expect("Embedding failed");
        if model.dispatch.is_some() {
            logits = model.forward_with_dispatch(&hidden, i).expect("Forward dispatch failed");
        } else {
            let hidden_out = model.forward_layers(&hidden, i).expect("Forward layers failed");
            logits = model.apply_output_head(&hidden_out).expect("Apply output head failed");
        }
    }
    println!(
        "Prefill done in {:.2}ms",
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    // Decode loop: generate tokens one at a time
    let mut generated = Vec::new();
    let total_decode_time = Instant::now();
    let max_tokens = 32;
    let sampling_config = SamplingConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.95,
        seed: Some(42),
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    for i in 0..max_tokens {
        let t_step = Instant::now();

        // Sample next token (greedy for benchmark)
        let next_token = LlamaModel::argmax_from_logits(&logits);

        // Check for EOS
        let piece = tokenizer
            .decode(&[next_token])
            .expect("Failed to decode token");
        if piece == "< |endoftext|>" {
            println!("EOS reached at token {}", i + 1);
            break;
        }

        generated.push(next_token);

        // Run single decode step: embed token, forward through layers
        let hidden = model.embed(next_token, i + prompt_tokens.len()).expect("Token embed failed");
        if model.dispatch.is_some() {
            logits = model.forward_with_dispatch(&hidden, i + prompt_tokens.len())
                .expect("Forward dispatch failed");
        } else {
            let hidden_out = model.forward_layers(&hidden, i + prompt_tokens.len())
                .expect("Forward layers failed");
            logits = model.apply_output_head(&hidden_out).expect("Apply output head failed");
        }

        println!(
            "Token {} ({:?}) in {:.2}ms",
            i + 1,
            piece,
            t_step.elapsed().as_secs_f64() * 1000.0
        );
    }

    let total_decode = total_decode_time.elapsed().as_secs_f64();
    let decode_speed = if total_decode > 0.0 {
        generated.len() as f64 / total_decode
    } else {
        0.0
    };

    println!("Generated {} tokens", generated.len());
    println!("Decode speed: {:.2} tok/s", decode_speed);
    println!("Total decode time: {:.2}s", total_decode);
}
