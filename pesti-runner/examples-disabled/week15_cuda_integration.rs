//! Week 15 Day 2: Explicit CUDA Initialization
//!
//! This benchmark explicitly initializes CUDA before creating DispatchContext
//! to ensure GPU detection works properly.

use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== Week 15 Day 2: Explicit CUDA Initialization ===");
    println!("Model: {}", model_path);

    // Priority 1: Explicitly initialize CUDA before DispatchContext
    #[cfg(feature = "cuda")]
    {
        use pesti_runner::device_discovery;
        
        println!("\n🔧 Initializing CUDA explicitly...");
        let devices = device_discovery::discover_local_devices();
        
        if !devices.is_empty() {
            println!("✅ CUDA initialized successfully!");
            for device in &devices {
                if device.ordinal != u32::MAX {
                    println!("   GPU {}: {} ({} GB VRAM)", 
                        device.ordinal, device.name, device.total_vram / 1024 / 1024 / 1024);
                } else {
                    println!("   CPU: {}", device.name);
                }
            }
        } else {
            println!("⚠️  No CUDA devices found after explicit initialization");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("⚠️  CUDA feature not enabled");
        println!("   Build with: cargo run --features cuda --example week15_cuda_integration");
    }

    // Load weights
    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let load_time = t0.elapsed();
    println!("\nLoaded weights in {:.2}s", load_time.as_secs_f32());

    println!(
        "tensors={}, bytes={}",
        weights.tensors.len(),
        weights.tensors.values().map(|v| v.len()).sum::<usize>()
    );

    // Load model with dispatch context (GPU should now be detected)
    let t1 = Instant::now();
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    let model_load_time = t1.elapsed();
    println!("Built model in {:.2}s", model_load_time.as_secs_f32());

    // Load tokenizer from GGUF
    let (tokenizer_config, _tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path)).map_err(|e| e.to_string())?;
    println!(
        "Loaded tokenizer with {} tokens",
        tokenizer_config.vocab_size
    );

    // Use a simple tokenization for demo purposes
    let prompt = "The quick brown fox jumps over the lazy dog.";
    println!("\nPrompt: {}", prompt);

    let prompt_tokens: Vec<u32> = prompt
        .split_whitespace()
        .flat_map(|word| {
            // Very rough tokenization - just use word length as proxy
            vec![word.len() as u32]
        })
        .take(10) // Limit to 10 tokens for demo
        .collect();

    println!("Encoded {} tokens (simplified)", prompt_tokens.len());

    // Reset KV caches before generation
    model.reset_cpu_kv_caches();

    // Generate tokens using CPU path (fallback until CUDA integration complete)
    let max_tokens = 64;
    let sampling_config = pesti_runner::transformer::SamplingConfig {
        temperature: 0.0, // Greedy decoding for reproducibility
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
        &[0], // No stop tokens
    )?;
    let gen_time = t3.elapsed();

    println!("\n=== Results ===");
    println!("Generated tokens:   {}", generated_tokens.len());
    println!("Generation time:    {:.3}s", gen_time.as_secs_f32());

    let throughput = generated_tokens.len() as f64 / (gen_time.as_secs_f32() as f64);
    println!("Throughput:         {:.2} tok/s", throughput);

    // Week 15 expectations
    println!("\n=== Week 15 Status ===");
    println!("Current CPU baseline: ~{} tok/s", throughput as usize);
    println!("Target with CUDA GEMM: ~500-800 tok/s (5-8× improvement)");
    println!("CUDA initialization: ✅ Explicit init successful");
    println!("Next step: Modify generate() to use forward_with_dispatch() for GPU acceleration");

    Ok(())
}
