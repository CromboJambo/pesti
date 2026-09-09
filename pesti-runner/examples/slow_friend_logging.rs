//! Slow-Friend G5: Bootstrap log → routing-alignment training data
//!
//! Runs a high-quant session and logs per-step signals (hidden norms,
//! divergence scores, activation pattern hashes) as JSONL for downstream
//! routing-distillation training.
//!
//! Usage: cargo run -p pesti-runner --release --features cuda \
//!   --example slow_friend_logging -- <model.gguf> [seq_len]

use std::path::Path;
use std::time::Instant;

use pesti_runner::kernel::slow_friend::{
    divergence, DivergenceMetric, SessionLogger, SlowFriendConfig, SlowFriendState,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: slow_friend_logging <model.gguf> [seq_len]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let seq_len: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(512);

    eprintln!("model: {}", model_path);
    eprintln!("seq_len: {}", seq_len);

    // Load model and tokenizer
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path), backend)?;

    // Build deterministic prompt
    let seed = "The quick brown fox jumps over the lazy dog. ";
    let mut prompt_tokens = tokenizer.encode(seed)?;
    while prompt_tokens.len() < seq_len {
        let more = tokenizer.encode(seed)?;
        prompt_tokens.extend(more);
    }
    prompt_tokens.truncate(seq_len);

    eprintln!("Running forward pass with session logging...");
    model.reset_cpu_kv_caches();

    let hidden_dim = model.config.embed_dim;
    let sf_cfg = SlowFriendConfig { dim: hidden_dim, alpha: 0.95 };
    let mut slow = SlowFriendState::new(&sf_cfg);
    let mut logger = SessionLogger::new();

    let t_start = Instant::now();

    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        let h = model.forward_layers_with_cache(&hidden, i)?;

        // Update slow friend and compute divergence
        slow.update(&h);
        let score = divergence(DivergenceMetric::Cosine, slow.summary(), &h);

        // Log this step for routing-alignment training data
        logger.log_step(tok, &h, Some(score.value));

        // Only compute logits for last token
        if i == prompt_tokens.len() - 1 {
            let _logits = model.apply_output_head(&h)?;
        }
    }

    let elapsed = t_start.elapsed();
    eprintln!("Forward pass complete in {:.3}s", elapsed.as_secs_f64());

    // Export training data
    println!("\nSession log summary:");
    println!("{}", logger.summary());

    let jsonl = logger.to_jsonl();
    println!("\nJSONL output ({} bytes):", jsonl.len());
    println!("{}", jsonl);

    Ok(())
}
