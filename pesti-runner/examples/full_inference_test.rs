//! Full inference test with real GGUF tokenizer

use pesti_runner::transformer::{LlamaModel, SamplingConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");

    println!("=== Week 16 Full Inference Test ===\n");

    // Load weights
    let t0 = Instant::now();
    let _weights = pesti_runner::load_gguf_weights(model_path)?;
    println!("Loaded weights in {:.2}s\n", t0.elapsed().as_secs_f32());

    // Load model with tokenizer
    let t1 = Instant::now();
    let mut model = LlamaModel::load_gguf(model_path)?;
    let model_load_time = t1.elapsed();
    println!("Built model in {:.2}s", model_load_time.as_secs_f32());

    // Check tokenizer
    if let Some(ref tok) = model.tokenizer {
        println!("✅ Real GGUF tokenizer loaded with {} tokens", tok.vocab_size());
    } else {
        println!("⚠️ No tokenizer found");
    }

    // Tokenize prompt
    let prompt = "The quick brown fox jumps over the lazy dog.";
    println!("\nPrompt: {}", prompt);

    let t2 = Instant::now();
    let prompt_tokens: Vec<u32> = if let Some(ref tok) = model.tokenizer {
        match tok.encode(prompt) {
            Ok(tokens) => {
                let encode_time = t2.elapsed();
                println!("Encoded {} tokens in {:.3}ms", tokens.len(), encode_time.as_secs_f32() * 1000.0);
                tokens
            }
            Err(e) => {
                eprintln!("⚠️ Tokenization error: {}", e);
                vec![151644u32] // BOS token as fallback
            }
        }
    } else {
        println!("⚠️ No tokenizer available");
        vec![151644u32]
    };

    // Generate tokens
    let max_tokens = 32;
    let sampling_config = SamplingConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        seed: Some(42),
    };

    let t3 = Instant::now();
    let mut rng = StdRng::seed_from_u64(42);
    let generated_tokens = model.generate(&prompt_tokens, max_tokens, &sampling_config, &mut rng, &[0])?;
    let gen_time = t3.elapsed();

    println!("\n=== Results ===");
    println!("Generated tokens:   {}", generated_tokens.len());
    println!("Generation time:    {:.3}s", gen_time.as_secs_f32());
    let throughput = generated_tokens.len() as f64 / gen_time.as_secs_f64();
    println!("Throughput:         {:.2} tok/s", throughput);

    // Decode output
    if let Some(ref tok) = model.tokenizer {
        match tok.decode(&generated_tokens) {
            Ok(decoded) => {
                println!("\nGenerated text (first 500 chars):");
                println!("{}", decoded.chars().take(500).collect::<String>());
            }
            Err(e) => {
                eprintln!("\n⚠️ Decode error: {}", e);
            }
        }
    }

    Ok(())
}
