//! Long-sequence benchmark to measure where batched/chunked generation provides value.
use std::path::Path;
use std::time::Instant;
use pesti_runner::{load_gguf_weights, LlamaModel, TokenizerBackend};
use rand::SeedableRng;

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    if !Path::new(model_path).exists() {
        eprintln!("Model not found: {}", model_path);
        std::process::exit(1);
    }

    println!("=== PESTI Long Sequence Benchmark ===");
    println!("Model: qwen2.5-0.5b-instruct-q4_k_m (Q4_K_M)");
    println!();

    let t_load = Instant::now();
    let weights = load_gguf_weights(Path::new(model_path)).expect("Failed to load GGUF weights");
    let mut model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Load tokenizer from GGUF metadata (new API returns tuple)
    let (_config, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(
        Path::new(model_path), TokenizerBackend::MistralRs
    ).expect("Failed to load tokenizer from GGUF");
    model.tokenizer = Some(tokenizer);

    // Test at different context lengths
    let seq_lengths = [128, 256, 512];
    
    for &seq_len in &seq_lengths {
        println!("--- Sequence length: {} ---", seq_len);
        
        // Create synthetic long prompt (repeated pattern)
        let base_prompt = "The quick brown fox jumps over the lazy dog. ";
        let mut long_prompt = String::new();
        while long_prompt.len() < seq_len * 4 {
            long_prompt.push_str(base_prompt);
        }
        
        // Encode
        let t_encode = Instant::now();
        let prompt_tokens = {
            let tok = model.tokenizer.as_ref().expect("No tokenizer");
            tok.encode(&long_prompt).expect("Failed to encode")
                .into_iter()
                .take(seq_len)
                .collect::<Vec<u32>>()
        };
        println!("Encoded {} tokens in {:.3}s", prompt_tokens.len(), t_encode.elapsed().as_secs_f64());
        
        // Prefill (first forward pass with full sequence)
        let sampling = pesti_runner::transformer::SamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: Some(42),
        };
        
        let t_prefill = Instant::now();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // Generate just 1 token to measure prefill time
        let _result = model.generate(&prompt_tokens, 1, &sampling, &mut rng, &[151663])
            .expect("Generation failed");
        let prefill_time = t_prefill.elapsed().as_secs_f64();
        
        println!("Prefill time: {:.3}s ({:.0} tok/s effective)", 
            prefill_time, seq_len as f64 / prefill_time);
        println!();
    }

    println!("=== Benchmark Complete ===");
}
