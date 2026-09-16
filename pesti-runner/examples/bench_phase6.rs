//! Phase 6 baseline benchmark - cudarc API conformance tok/s measurement
use std::path::Path;
use std::time::Instant;
use pesti_runner::{load_gguf_weights, LlamaModel};

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "The capital of France is";
    let max_tokens = 32;

    println!("Loading model from {}", model_path);
    let t_load = Instant::now();
    let weights = load_gguf_weights(Path::new(model_path)).expect("Failed to load GGUF weights");
    let mut model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Tokenize prompt
    let tokenizer = model.tokenizer.as_ref().expect("No tokenizer");
    let prompt_tokens = tokenizer.encode(prompt).expect("Failed to encode prompt");
    println!("Encoded {} tokens", prompt_tokens.len());

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
        if piece == "< |endoftext|>" {
            println!("EOS reached at token {}", i + 1);
            break;
        }

        generated.push(next_token);

        // Run single decode step
        let token_embed = model.embedding(&next_token).expect("Token embed failed");
        hidden = model.layers[0].forward_with_cache(
            &token_embed,
            &mut model.layers[0].cache,
            i + prompt_tokens.len(),
        );
        for layer in &mut model.layers[1..] {
            hidden = layer.forward_with_cache(&hidden, &mut layer.cache, i + prompt_tokens.len());
        }

        let step_time_ms = t_step.elapsed().as_secs_f64() * 1000.0;
        if i < 3 || i == max_tokens - 1 {
            println!("Token {} ({:?}) in {:.2}ms", i + 1, piece, step_time_ms);
        }
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

    // Decode and display output
    if !generated.is_empty() {
        let generated_text = tokenizer.decode(&generated).expect("Failed to decode");
        println!("Output: {}", &generated_text[..generated_text.len().min(150)]);
    }
}
