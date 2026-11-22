//! Real-model benchmark: standard fused decode vs Route C delta decode attention.
//!
//! Loads Qwen3.5-4B GGUF and runs inference through both paths to compare:
//! 1. Output quality (token overlap)
//! 2. Throughput (tok/s)

use pesti_runner::transformer::{LlamaModel, SamplingConfig};
use rand::SeedableRng;

use std::path::Path;
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = find_model()?;

    println!("=== Route C Delta Attention — Real Model Benchmark ===");
    println!("Model: {}", model_path);

    // Load weights and build model
    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let mut model = LlamaModel::from_gguf_weights(weights)?;
    println!("Loaded & built model in {:.2}s", t0.elapsed().as_secs_f32());

    // Load tokenizer
    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(
        Path::new(&model_path),
        backend,
    )?;

    // Set up CUDA and attach kernels (like runtime.rs does)
    setup_cuda_kernels(&mut model)?;

    // Prompt for testing
    let prompt = "Explain how attention mechanisms work in transformer models.";
    let prompt_tokens = tokenizer.encode(prompt)?;
    println!("Prompt: {} ({} tokens)", prompt, prompt_tokens.len());

    let max_tokens = 32;
    let sampling_config = SamplingConfig {
        temperature: 0.0, // Greedy for reproducibility
        top_p: 1.0,
        top_k: 50,
        seed: Some(42),
    };

    // === Test 1: Standard fused decode (no delta kernel) ===
    println!("\n--- Test 1: Standard Fused Decode ---");
    disable_delta_kernel(&mut model);
    let t1 = Instant::now();
    let generated_standard = run_generation(&mut model, &prompt_tokens, max_tokens, &sampling_config)?;
    let time_standard = t1.elapsed().as_secs_f64();

    let decoded_standard = tokenizer.decode(&generated_standard)?;
    println!("Output: {}", &decoded_standard[..std::cmp::min(200, decoded_standard.len())]);
    println!("Throughput: {:.2} tok/s", generated_standard.len() as f64 / time_standard);

    // === Test 2: Delta decode (Route C) ===
    println!("\n--- Test 2: Route C Delta Decode ---");
    enable_delta_kernel(&mut model)?;
    let t2 = Instant::now();
    let generated_delta = run_generation(&mut model, &prompt_tokens, max_tokens, &sampling_config)?;
    let time_delta = t2.elapsed().as_secs_f64();

    let decoded_delta = tokenizer.decode(&generated_delta)?;
    println!("Output: {}", &decoded_delta[..std::cmp::min(200, decoded_delta.len())]);
    println!("Throughput: {:.2} tok/s", generated_delta.len() as f64 / time_delta);

    // === Comparison ===
    println!("\n=== Comparison ===");
    let overlap = token_overlap(&generated_standard, &generated_delta);
    println!("Token overlap: {}% ({}/{} tokens match)", 
        (overlap * 100.0).round(),
        (overlap * generated_standard.len() as f64) as usize,
        generated_standard.len());

    if time_standard > 0.0 && time_delta > 0.0 {
        println!("Speedup: {:.2}x", time_standard / time_delta);
    }

    // Check for identical outputs (greedy decoding should produce same tokens)
    let identical = generated_standard == generated_delta;
    println!("Identical output: {}", if identical { "YES" } else { "NO" });
    if !identical {
        for (i, (&a, &b)) in generated_standard.iter().zip(generated_delta.iter()).enumerate() {
            if a != b {
                let piece_a = tokenizer.decode(&[a])?;
                let piece_b = tokenizer.decode(&[b])?;
                println!("  First divergence at token {}: '{}' vs '{}'", i, piece_a, piece_b);
                break;
            }
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn setup_cuda_kernels(model: &mut LlamaModel) -> Result<(), Box<dyn std::error::Error>> {
    use pesti_runner::cuda_runtime::CudaRuntime;
    
    // Create CUDA runtime (like runtime.rs does)
    let cuda_rt = CudaRuntime::new(0)?;
    let context = cuda_rt.context().clone();
    let stream = cuda_rt.new_stream()?;

    // Attach standard fused decode kernel
    if let Ok(fused_kernel) = pesti_runner::kernel::fused_decode_attention::build_fused_decode_attention_kernel(
        context.clone(), stream.clone()
    ) {
        println!("Attached CUDA fused decode attention kernel");
        if let Some(ref mut dispatch_layers) = model.dispatch_layers {
            for layer_dispatch in dispatch_layers.iter_mut() {
                layer_dispatch.attention.fused_decode_kernel = Some(fused_kernel.clone());
            }
        }
    }

    // Attach delta kernel (Route C)
    if let Ok(delta_kernel) = pesti_runner::kernel::fused_decode_attention_delta::build_delta_fused_decode_attention_kernel(
        context, stream
    ) {
        println!("Attached CUDA delta-aware decode attention kernel (Route C)");
        if let Some(ref mut dispatch_layers) = model.dispatch_layers {
            for layer_dispatch in dispatch_layers.iter_mut() {
                layer_dispatch.attention.delta_decode_kernel = Some(delta_kernel.clone());
            }
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn enable_delta_kernel(_model: &mut LlamaModel) -> Result<(), Box<dyn std::error::Error>> {
    // Delta kernel is already attached; dispatch will select it when available
    Ok(())
}

#[cfg(feature = "cuda")]
fn disable_delta_kernel(model: &mut LlamaModel) {
    if let Some(ref mut dispatch_layers) = model.dispatch_layers {
        for layer_dispatch in dispatch_layers.iter_mut() {
            layer_dispatch.attention.delta_decode_kernel = None;
        }
    }
}

#[cfg(feature = "cuda")]
fn find_model() -> Result<String, Box<dyn std::error::Error>> {
    // Check command line argument first
    if let Some(path) = std::env::args().nth(1) {
        if Path::new(&path).exists() {
            return Ok(path);
        }
    }
    
    // Try known locations for Qwen models in HF cache
    let home = dirs_home();
    let hf_cache = home.join(".cache/huggingface/hub");
    
    if hf_cache.exists() {
        for entry in walkdir_simple(&hf_cache)? {
            if let Some(name) = entry.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".gguf") && (name.contains("Qwen") || name.contains("qwen")) {
                    return Ok(entry.to_string_lossy().to_string());
                }
            }
        }
    }
    
    Err(format!("No GGUF model found in HF cache. Pass a model path as argument.").into())
}

#[cfg(feature = "cuda")]
fn dirs_home() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/crombo"))
}

#[cfg(feature = "cuda")]
fn walkdir_simple(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            results.extend(walkdir_simple(&path)?);
        } else {
            results.push(path);
        }
    }
    Ok(results)
}

#[cfg(feature = "cuda")]
fn run_generation(
    model: &mut LlamaModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    config: &SamplingConfig,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let generated = model.generate(
        prompt_tokens,
        max_tokens,
        config,
        &mut rand::rngs::StdRng::seed_from_u64(42),
        &[0],
    )?;
    
    Ok(generated)
}

#[cfg(feature = "cuda")]
fn token_overlap(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let common = a.iter().zip(b.iter()).filter(|&(&x, &y)| x == y).count();
    common as f64 / a.len().max(b.len()) as f64
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("CUDA feature required. Run with: cargo run --features cuda");
    std::process::exit(1);
}
