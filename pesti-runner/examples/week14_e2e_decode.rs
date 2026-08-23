//! Week 14 end-to-end decode benchmark with the current PESTI API.
//!
//! This exercises the real autoregressive pipeline against a real GGUF model:
//! 1. Load GGUF weights (`LlamaModel::from_gguf_weights`)
//! 2. Load tokenizer from GGUF (`load_tokenizer_from_gguf`)
//! 3. Tokenize a prompt
//! 4. Generate tokens with `LlamaModel::generate`
//! 5. Decode and report throughput

use std::path::Path;
use std::time::Instant;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== Week 14: Real E2E Decode Benchmark ===");
    println!("Model: {}", model_path);

    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    println!(
        "Loaded weights in {:.2}s; tensors={}, bytes={}",
        t0.elapsed().as_secs_f32(),
        weights.tensors.len(),
        weights.tensors.values().map(|v| v.len()).sum::<usize>()
    );

    let t1 = Instant::now();
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    println!("Built model in {:.2}s", t1.elapsed().as_secs_f32());

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (tokenizer_config, tokenizer) =
        pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path), backend)?;
    println!(
        "Loaded tokenizer; vocab_size={}, bos={:?}, eos={:?}",
        tokenizer_config.vocab_size,
        tokenizer_config.bos_token_id,
        tokenizer_config.eos_token_id
    );

    let prompt = "The quick brown fox jumps over the lazy dog.";
    println!("\nPrompt: {}", prompt);

    let t2 = Instant::now();
    let prompt_tokens = tokenizer.encode(prompt)?;
    let encode_time = t2.elapsed();
    println!(
        "Encoded {} tokens in {:.3}ms",
        prompt_tokens.len(),
        encode_time.as_secs_f32() * 1000.0
    );

    model.reset_cpu_kv_caches();

    let max_tokens = 64;
    let sampling_config = pesti_runner::transformer::SamplingConfig {
        temperature: 0.0,
        top_p: 0.9,
        top_k: 40,
        seed: Some(42),
    };

    let t3 = Instant::now();
    let generated_tokens = model.generate(
        &prompt_tokens,
        max_tokens,
        &sampling_config,
        &mut rand::rngs::StdRng::seed_from_u64(42),
        &[0],
    )?;
    let gen_time = t3.elapsed();

    println!("\n=== Results ===");
    println!("Generated tokens:   {}", generated_tokens.len());
    println!("Generation time:    {:.3}s", gen_time.as_secs_f32());
    println!(
        "Throughput:         {:.2} tok/s",
        generated_tokens.len() as f64 / gen_time.as_secs_f64() as f64
    );

    let decoded = tokenizer.decode(&generated_tokens)?;
    println!("\nGenerated text (first 200 chars):");
    println!("{}", decoded.chars().take(200).collect::<String>());

    let total_time = t0.elapsed();
    println!("\n=== Timeline ===");
    println!("Weight/model/tokenizer setup: {:.2}s", total_time.as_secs_f32());
    println!("Generation:                   {:.3}s", gen_time.as_secs_f32());
    println!("Total measured wall time:     {:.3}s", total_time.as_secs_f32());

    Ok(())
}
