//! CPU-only end-to-end text generation. Bypasses the GPU dispatch path (which
//! OOMs when the GPUs are occupied by other processes) and exercises the
//! CPU forward path — the one fixed by the Q5_0/Q5_1/Q6_K dequant and the
//! SwiGLU sigmoid corrections — all the way to generated text.
//!
//! Usage: cargo run -p pesti-runner --release --features cuda
//!   --example cpu_e2e_generate -- <model.gguf> "<prompt>" [max_tokens]
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: cpu_e2e_generate <model.gguf> \"<prompt>\" [max_tokens]");
        std::process::exit(2);
    }
    let model_path = args[1].clone();
    let prompt = args[2].clone();
    let max_tokens = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(48);

    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) =
        pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(&model_path), backend)?;

    let prompt_tokens = tokenizer.encode(&prompt)?;
    eprintln!("prompt: {} ({} tokens)", prompt, prompt_tokens.len());

    model.reset_cpu_kv_caches();

    // Prefill: run each prompt token at its position (fills the KV cache).
    let mut logits: Vec<f32> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        let h = model.forward_layers_with_cache(&hidden, i)?;
        logits = model.apply_output_head(&h)?;
    }

    let mut generated: Vec<u32> = Vec::new();
    let t0 = Instant::now();
    let mut pos = prompt_tokens.len();
    for _ in 0..max_tokens {
        let next = pesti_runner::transformer::LlamaModel::argmax_from_logits(&logits);
        if next == 151645 {
            break; // Qwen2.5 eos
        }
        generated.push(next);
        let hidden = model.embed(next, pos)?;
        let h = model.forward_layers_with_cache(&hidden, pos)?;
        logits = model.apply_output_head(&h)?;
        pos += 1;
    }
    let dt = t0.elapsed();

    let text = tokenizer.decode(&generated)?;
    println!("\n=== CPU E2E GENERATION ===");
    println!(
        "tokens: {}  time: {:.3}s  ({:.1} tok/s)",
        generated.len(),
        dt.as_secs_f32(),
        generated.len() as f64 / dt.as_secs_f64()
    );
    println!("text:   {}", text);
    Ok(())
}
