//! Phase 6 baseline benchmark - cudarc API conformance tok/s measurement
use std::time::Instant;
use pesti_runner::{load_gguf_weights, LlamaModel};
use rand::SeedableRng;

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "The capital of France is";
    let max_tokens = 32;

    println!("Loading model from {}", model_path);
    let t_load = Instant::now();
    let weights = load_gguf_weights(std::path::Path::new(model_path)).expect("Failed to load GGUF weights");
    let mut model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Tokenize prompt (drop borrow before generate)
    let prompt_tokens = {
        let tok = model.tokenizer.as_ref().expect("No tokenizer");
        tok.encode(prompt).expect("Failed to encode prompt")
    };
    println!("Encoded {} tokens", prompt_tokens.len());

    // Create sampling config (greedy)
    let sampling = pesti_runner::transformer::SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: Some(42),
    };

    // Warmup generation
    println!("Warmup generation...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let warmup_tokens = model.generate(&prompt_tokens, 8, &sampling, &mut rng, &[]).expect("Warmup failed");
    println!("Warmup generated {} tokens", warmup_tokens.len());

    // Benchmark generation
    println!("Benchmark generation...");
    let t_gen = Instant::now();
    let result_tokens = model.generate(&prompt_tokens, max_tokens, &sampling, &mut rng, &[]).expect("Generation failed");
    let gen_time = t_gen.elapsed().as_secs_f64();

    println!("Generated {} tokens in {:.2}s", result_tokens.len(), gen_time);
    let tok_per_sec = result_tokens.len() as f64 / gen_time;
    println!("Decode speed: {:.2} tok/s", tok_per_sec);

    // Decode and display output (re-borrow tokenizer after generate)
    if !result_tokens.is_empty() {
        let generated_text = model.tokenizer.as_ref().expect("No tokenizer").decode(&result_tokens).expect("Failed to decode");
        println!("Output: {}", &generated_text[..generated_text.len().min(150)]);
    }
}
