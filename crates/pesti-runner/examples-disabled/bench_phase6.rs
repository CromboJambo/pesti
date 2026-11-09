//! Phase 6 baseline benchmark - cudarc API conformance tok/s measurement
use pesti_runner::transformer::tokenizer::{TokenizerBackend, load_tokenizer_from_gguf};
use pesti_runner::{LlamaModel, load_gguf_weights};
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "The capital of France is";
    let max_tokens = 32;

    println!("Loading model from {}", model_path);
    let t_load = Instant::now();
    let weights = load_gguf_weights(Path::new(model_path)).expect("Failed to load GGUF weights");
    let mut model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Load tokenizer from GGUF file
    let (_config, tokenizer) =
        load_tokenizer_from_gguf(Path::new(model_path), TokenizerBackend::MistralRs)
            .expect("Failed to load tokenizer");

    // Tokenize prompt using the tokenizer directly
    let prompt_tokens = tokenizer.encode(prompt).expect("Failed to encode prompt");
    println!("Encoded {} tokens", prompt_tokens.len());

    // Create sampling config (greedy for benchmark)
    let sampling = pesti_runner::transformer::SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: Some(42),
    };

    // Warmup generation
    println!("Warmup generation...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    model
        .generate(&prompt_tokens, 8, &sampling, &mut rng, &[])
        .expect("Warmup failed");

    // Benchmark generation
    println!("Benchmarking generation...");
    let t_gen = Instant::now();
    let result_tokens = model
        .generate(&prompt_tokens, max_tokens, &sampling, &mut rng, &[])
        .expect("Generation failed");
    let gen_time = t_gen.elapsed().as_secs_f64();

    println!(
        "Generated {} tokens in {:.2}s",
        result_tokens.len(),
        gen_time
    );
    println!(
        "Decode speed: {:.2} tok/s",
        result_tokens.len() as f64 / gen_time
    );

    // Decode and display output
    if !result_tokens.is_empty() {
        let generated_text = tokenizer.decode(&result_tokens).expect("Failed to decode");
        println!(
            "Output: {}",
            &generated_text[..generated_text.len().min(150)]
        );
    }
}
