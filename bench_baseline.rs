//! Baseline tok/s benchmark for pesti-runner on qwen2.5-0.5b Q4_K_M.
use pesti_runner::{load_gguf_weights, LlamaModel};
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    if !Path::new(model_path).exists() {
        eprintln!("Model not found: {}", model_path);
        std::process::exit(1);
    }

    println!("=== PESTI Baseline Benchmark ===");
    println!("Model: qwen2.5-0.5b-instruct-q4_k_m (Q4_K_M)");
    println!();

    // Load weights and build model
    let t_load = Instant::now();
    let weights = load_gguf_weights(Path::new(model_path)).expect("Failed to load GGUF weights");
    let mut model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Load tokenizer from GGUF metadata
    let (_config, tokenizer) = pesti_runner::load_tokenizer_from_gguf(
        Path::new(model_path),
        pesti_runner::TokenizerBackend::MistralRs,
    )
    .expect("Failed to load tokenizer");
    model.tokenizer = Some(Box::new(tokenizer));

    // Benchmark prompt
    let prompt = "Write a haiku about programming.";
    let max_tokens = 32;

    // Tokenize
    let t_encode = Instant::now();
    let prompt_tokens = {
        let tok = model.tokenizer.as_ref().expect("No tokenizer");
        tok.encode(prompt).expect("Failed to encode")
    };
    println!("Prompt: \"{}\"", prompt);
    println!(
        "Encoded {} tokens in {:.2}s",
        prompt_tokens.len(),
        t_encode.elapsed().as_secs_f64()
    );

    // Sampling config for greedy decoding (fastest, deterministic)
    let sampling = pesti_runner::SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: Some(42),
    };

    // Warmup run (not measured)
    println!("Running warmup...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    model
        .generate(&prompt_tokens, 8, &sampling, &mut rng, &[151663])
        .expect("Warmup generation failed");

    // Timed run
    println!("Running benchmark...");
    let t_gen = Instant::now();
    rng = rand::rngs::StdRng::seed_from_u64(42);
    let result_tokens = model
        .generate(&prompt_tokens, max_tokens, &sampling, &mut rng, &[151663])
        .expect("Generation failed");
    let gen_time = t_gen.elapsed().as_secs_f64();

    println!();
    println!("=== Results ===");
    println!(
        "Generated {} tokens in {:.2}s",
        result_tokens.len(),
        gen_time
    );

    if !result_tokens.is_empty() {
        let tok_per_sec = result_tokens.len() as f64 / gen_time;
        println!("Decode speed: {:.2} tok/s", tok_per_sec);
        println!("Avg time per token: {:.0}ms", 1000.0 / tok_per_sec);
    }

    // Decode and show output
    if let Some(tok) = model.tokenizer.as_ref() {
        let generated_text = tok.decode(&result_tokens).expect("Failed to decode");
        println!();
        println!(
            "Output: {}",
            &generated_text[..generated_text.len().min(200)]
        );
    }

    println!();
    println!("=== Benchmark Complete ===");
}
