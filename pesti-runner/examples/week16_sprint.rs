//! Week 16: GPU Attention Kernel Development Sprint
//!
//! **Goal**: Build on Week 14-15 foundation to achieve production-ready GPU inference
//! **Focus**:
//! - Verify CUDA path is working with real model
//! - Profile performance bottlenecks
//! - Optimize attention kernels for sm_8.9 architecture

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== Week 16: GPU Attention Kernel Sprint ===");
    println!("Model: {}", model_path);

    // Load weights
    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let load_time = t0.elapsed();
    println!("Loaded weights in {:.2}s", load_time.as_secs_f32());

    println!(
        "tensors={}, bytes={}",
        weights.tensors.len(),
        weights.tensors.values().map(|v| v.len()).sum::<usize>()
    );

    // Load model with GPU support
    let t1 = Instant::now();
    let mut model = pesti_runner::transformer::LlamaModel::load_gguf(Path::new(model_path))?;
    let model_load_time = t1.elapsed();
    println!("Built model in {:.2}s", model_load_time.as_secs_f32());

    // Check if CUDA is available
    if let Some(ref dispatch) = model.dispatch {
        if dispatch.gpu_available() {
            println!("✅ CUDA GPU detected and available");
        } else {
            println!("⚠️  CUDA enabled but GPU not available (falling back to CPU)");
        }
    } else {
        println!("ℹ️  CPU-only mode (no CUDA feature or GPU detection failed)");
    }

    // Use tokenizer if available (stored in model.tokenizer)
    let prompt_tokens: Vec<u32> = if let Some(ref tok) = model.tokenizer {
        println!(
            "Loaded real GGUF tokenizer with {} tokens",
            model.tokenizer_config.as_ref().map(|c| c.base_vocab_size).unwrap_or(0)
        );

        // Tokenize prompt
        let prompt = "The quick brown fox jumps over the lazy dog.";
        println!("\nPrompt: {}", prompt);

        let t2 = Instant::now();
        match tok.encode(prompt) {
            Ok(tokens) => {
                let encode_time = t2.elapsed();
                println!(
                    "Encoded {} tokens in {:.3}ms",
                    tokens.len(),
                    encode_time.as_secs_f32() * 1000.0
                );
                tokens
            }
            Err(e) => {
                eprintln!("⚠️  Tokenization error: {} - using fallback", e);
                vec![151644u32] // BOS token as fallback
            }
        }
    } else {
        println!("⚠️  No tokenizer found in GGUF file");
        vec![151644u32] // BOS token as fallback
    };

    // Generate tokens
    let max_tokens = 64;
    let sampling_config = pesti_runner::transformer::SamplingConfig {
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

    // Try to decode if we have a tokenizer
    if let Some(ref tok) = model.tokenizer {
        match tok.decode(&generated_tokens) {
            Ok(decoded) => {
                println!("\nGenerated text (first 500 chars):");
                println!("{}", decoded.chars().take(500).collect::<String>());
            }
            Err(e) => {
                eprintln!("\n⚠️  Decode error: {}", e);
            }
        }
    }

    Ok(())
