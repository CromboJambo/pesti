//! Measure token generation throughput and timing breakdown

use pesti_runner::transformer::{LlamaModel, SamplingConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt_text = "Once upon a time in the land of Rust,";
    let target_tokens = 100;

    println!("=== Baseline Profiling ===");
    println!("Model: {}", model_path);
    println!("Prompt: {}", prompt_text);
    println!("Target tokens: {}", target_tokens);
    println!();

    // Load model
    let start_load = Instant::now();
    let mut model = match LlamaModel::load_gguf(std::path::Path::new(model_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Load failed: {}", e);
            std::process::exit(1);
        }
    };
    println!(
        "✅ Model loaded in {:.2}s",
        start_load.elapsed().as_secs_f64()
    );

    // Encode prompt to tokens (using tokenizer) - do this first
    let prompt_tokens = {
        let tokenizer = model.tokenizer.as_ref().unwrap();
        match tokenizer.encode(prompt_text) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("Encoding failed: {}", e);
                std::process::exit(1);
            }
        }
    };
    println!("✅ Encoded prompt to {} tokens", prompt_tokens.len());

    // Generate tokens with timing
    let start_gen = Instant::now();

    let sampling_config = SamplingConfig {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 40,
        seed: Some(42),
    };

    let stop_tokens: Vec<u32> = vec![151644]; // EOS token

    match model.generate(
        &prompt_tokens,
        target_tokens,
        &sampling_config,
        &mut StdRng::seed_from_u64(42),
        &stop_tokens,
    ) {
        Ok(tokens) => {
            let gen_time = start_gen.elapsed();
            let throughput = target_tokens as f64 / gen_time.as_secs_f64();

            println!(
                "✅ Generated {} tokens in {:.2}s",
                tokens.len(),
                gen_time.as_secs_f64()
            );
            println!("📊 Throughput: {:.2} tok/s", throughput);
            println!();

            // Decode first few tokens for verification - do this after generation
            let tokenizer = model.tokenizer.as_ref().unwrap();
            if let Ok(decoded) = tokenizer.decode(&tokens[..10]) {
                println!("Sample output: \"{}\"", decoded);
            }
        }
        Err(e) => {
            eprintln!("Generation failed: {}", e);
            std::process::exit(1);
        }
    }

    println!();
    println!("=== Baseline Complete ===");
}
