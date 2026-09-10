//! Week 20: Real tok/s benchmark with fused attention kernel enabled.
//! Measures FP16 KV cache + fused attention throughput on RTX 4070 Ti SUPER.

use std::time::Instant;
use pesti_runner::transformer::{LlamaModel, SamplingConfig};
use pesti_runner::kernel::fused_attention_conformant::{build_fused_attention_kernel_conformant, FusedAttentionArch};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("=== Week 20: Real tok/s Benchmark ===");
    println!("Model: {}", model_path);

    // Load weights and build model
    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(std::path::Path::new(model_path))?;
    let mut model = LlamaModel::from_gguf_weights(weights)?;
    println!("Loaded & built model in {:.2}s", t0.elapsed().as_secs_f32());

    // Enable fused attention kernel on all layers
    #[cfg(feature = "cuda")]
    {
        use std::sync::Arc;
        let ctx = pesti_runner::kernel::dispatch::DispatchContext::new();
        if let Some(stream) = ctx.cuda_stream() {
            if let Ok(kernel) = build_fused_attention_kernel_conformant(
                FusedAttentionArch::MmaSync,
                Arc::new(ctx.cuda_context().clone()),
                Arc::new(stream.clone()),
            ) {
                for layer in &mut model.layers {
                    layer.attention.set_fused_kernel(kernel);
                }
                println!("Fused attention kernel enabled on all {} layers", model.config.num_layers);
            } else {
                println!("Failed to build fused attention kernel");
            }
        }
    }

    // Tokenize prompt
    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(
        std::path::Path::new(model_path), backend)?;
    
    let prompt = "The quick brown fox jumps over the lazy dog.";
    let prompt_tokens = tokenizer.encode(prompt)?;
    println!("Prompt: {} ({} tokens)", prompt, prompt_tokens.len());

    // Generate with timing
    model.reset_cpu_kv_caches();
    let max_tokens = 64;
    let sampling_config = SamplingConfig { temperature: 0.0, top_p: 0.9, top_k: 40, seed: Some(42) };
    
    let t1 = Instant::now();
    let generated = model.generate(&prompt_tokens, max_tokens, &sampling_config, 
        &mut rand::rngs::StdRng::seed_from_u64(42), &[0])?;
    let gen_time = t1.elapsed().as_secs_f64();

    println!("\n=== Results ===");
    println!("Generated tokens: {}", generated.len());
    println!("Generation time: {:.3}s", gen_time);
    println!("Throughput: {:.2} tok/s", generated.len() as f64 / gen_time);
    
    let decoded = tokenizer.decode(&generated)?;
    println!("\nOutput (first 100 chars): {}", &decoded[..std::cmp::min(100, decoded.len())]);

    Ok(())
}