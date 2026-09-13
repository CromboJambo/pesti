//! Benchmark pesti-runner's own CUDA inference stack (not llama.cpp FFI).
//! Uses LlamaModel::from_gguf_weights() + generate() directly.
//!
//! This measures how far pesti-runner gets on its own, without the
//! llama.cpp FFI wrapper used by benchmark_generation.rs.

use pesti_runner::transformer::{LlamaModel, SamplingConfig};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== pesti-runner standalone CUDA benchmark ===");
    println!("Model: {}", model_path);
    println!();

    // Step 1: Load GGUF weights via pesti-runner's own loader
    let load_start = Instant::now();
    println!("Loading GGUF weights (pesti-runner)...");
    let weights = pesti_runner::load_gguf_weights(std::path::Path::new(model_path))?;
    println!(
        "✓ Weights loaded in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );

    // Step 2: Build LlamaModel from weights using pesti-runner's own stack
    let build_start = Instant::now();
    println!("Building LlamaModel (pesti-runner)...");
    let mut model = LlamaModel::from_gguf_weights(weights)?;
    println!(
        "✓ Model built in {:.2}s",
        build_start.elapsed().as_secs_f32()
    );

    // Step 3: Tokenize prompt using pesti-runner's tokenizer
    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(
        std::path::Path::new(model_path),
        backend,
    )?;

    let prompt = "Hello, how are you?";
    let prompt_tokens = tokenizer.encode(prompt)?;
    println!("\nPrompt: \"{}\"", prompt);
    println!("Tokenized to {} tokens", prompt_tokens.len());
    println!("Generating via pesti-runner's own stack (not llama.cpp FFI)...\n");

    // Step 4: Run generation loop using pesti-runner's generate() method
    model.reset_cpu_kv_caches();
    let max_tokens = 128;
    let sampling_config = SamplingConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        seed: Some(42),
    };

    let gen_start = Instant::now();
    let generated = model.generate(
        &prompt_tokens,
        max_tokens,
        &sampling_config,
        &mut rand::rngs::StdRng::seed_from_u64(42),
        &[0],
    )?;
    let gen_time = gen_start.elapsed().as_secs_f64();

    // Decode output for display
    let decoded = tokenizer.decode(&generated)?;

    println!("\n=== Results ===");
    println!("Generated tokens: {}", generated.len());
    println!("Generation time: {:.3}s", gen_time);
    println!(
        "Tokens/sec (pesti-runner own stack): {:.1}",
        generated.len() as f64 / gen_time
    );
    println!("\nOutput (first 200 chars):");
    println!("{}", &decoded[..std::cmp::min(200, decoded.len())]);

    Ok(())
}
