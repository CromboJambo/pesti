//! Slow-Friend Expert Scoping Probe (G2)
//!
//! Tests whether using the slow friend's stable state to scope "expert"
//! activation reduces drift at high context. For a dense model, we partition
//! the hidden state into synthetic expert regions and track which are active.

use std::path::Path;
use std::time::Instant;

use pesti_runner::kernel::slow_friend::{
    apply_scoping, activation_pattern, ExpertPrior, SlowFriendConfig, SlowFriendState,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: slow_friend_scoping <model.gguf> [seq_lengths...]");
        std::process::exit(2);
    }
    let model_path = &args[1];

    // Parse sequence lengths from env or args
    let seq_lens_str = match args.get(2) {
        Some(s) => s.clone(),
        None => std::env::var("PESTI_SLOW_SEQS").unwrap_or_else(|_| "256,512,1024".into()),
    };
    let seq_lens: Vec<usize> = seq_lens_str.split(',').map(|s| s.trim().parse())
        .collect::<Result<Vec<usize>, _>>()?;

    if seq_lens.len() < 2 {
        eprintln!("need at least 2 sequence lengths (reference + long context)");
        std::process::exit(2);
    }

    let num_experts = 8; // Partition hidden state into 8 synthetic experts

    println!("\nSlow-Friend Expert Scoping Probe (G2)");
    println!("======================================");
    println!("model: {}", model_path);
    println!("seq_lengths: {:?}", seq_lens);
    println!("synthetic_experts: {} (hidden-state region partitioning)", num_experts);

    // Load model and tokenizer using the same pattern as G1 drift probe
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path), backend)?;

    // Seed sentence for deterministic long prompts
    let seed = "The quick brown fox jumps over the lazy dog. ";

    // Build prompt tokenization function (closure to avoid type annotation issues)
    let make_prompt = |seq_len: usize| {
        let mut tokens = tokenizer.encode(seed).unwrap();
        while tokens.len() < seq_len {
            let more = tokenizer.encode(seed).unwrap();
            tokens.extend(more);
        }
        tokens.truncate(seq_len);
        tokens
    };

    // Run at reference (short) context to get baseline activation pattern
    let ref_seq_len = seq_lens[0];
    println!("\n--- Reference context: {} tokens ---", ref_seq_len);
    let ref_pattern = {
        let prompt_tokens = make_prompt(ref_seq_len);
        model.reset_cpu_kv_caches();

        let hidden_dim = model.config.embed_dim;
        let sf_cfg = SlowFriendConfig { dim: hidden_dim, alpha: 0.95 };
        let mut slow = SlowFriendState::new(&sf_cfg);

        let t_start = Instant::now();
        let mut last_pattern = Vec::new();

        for (i, &tok) in prompt_tokens.iter().enumerate() {
            let hidden = model.embed(tok, i)?;
            let h = model.forward_layers_with_cache(&hidden, i)?;
            slow.update(&h);
            last_pattern = activation_pattern(&h, num_experts);

            if i == prompt_tokens.len() - 1 {
                let _logits = model.apply_output_head(&h)?;
            }
        }

        let elapsed = t_start.elapsed();
        eprintln!("[seq={},scoped=false] done in {:.3}s", ref_seq_len, elapsed.as_secs_f64());
        last_pattern
    };
    println!("Reference activation pattern: {:?}", ref_pattern);

    // Run at long contexts with and without scoping
    for seq_len in &seq_lens[1..] {
        println!("\n--- Long context: {} tokens ---", seq_len);

        // Free activation (no scoping)
        let free_pattern = {
            let prompt_tokens = make_prompt(*seq_len);
            model.reset_cpu_kv_caches();

            let hidden_dim = model.config.embed_dim;
            let sf_cfg = SlowFriendConfig { dim: hidden_dim, alpha: 0.95 };
            let mut slow = SlowFriendState::new(&sf_cfg);

            let t_start = Instant::now();
            let mut last_pattern = Vec::new();

            for (i, &tok) in prompt_tokens.iter().enumerate() {
                let hidden = model.embed(tok, i)?;
                let h = model.forward_layers_with_cache(&hidden, i)?;
                slow.update(&h);
                last_pattern = activation_pattern(&h, num_experts);

                if i == prompt_tokens.len() - 1 {
                    let _logits = model.apply_output_head(&h)?;
                }
            }

            let elapsed = t_start.elapsed();
            eprintln!("[seq={},scoped=false] done in {:.3}s", seq_len, elapsed.as_secs_f64());
            last_pattern
        };
        let free_jaccard = jaccard_similarity(&ref_pattern, &free_pattern);
        println!("Free activation pattern:  {:?}", free_pattern);
        println!("Free Jaccard similarity:  {:.4}", free_jaccard);

        // Scoped activation (using slow-friend prior)
        let scoped_pattern = {
            let prompt_tokens = make_prompt(*seq_len);
            model.reset_cpu_kv_caches();

            let hidden_dim = model.config.embed_dim;
            let sf_cfg = SlowFriendConfig { dim: hidden_dim, alpha: 0.95 };
            let mut slow = SlowFriendState::new(&sf_cfg);
            let prior = ExpertPrior::new(hidden_dim, num_experts);

            let t_start = Instant::now();
            let mut last_pattern = Vec::new();

            for (i, &tok) in prompt_tokens.iter().enumerate() {
                let hidden = model.embed(tok, i)?;
                let h = model.forward_layers_with_cache(&hidden, i)?;
                slow.update(&h);

                if slow.step() > 0 {
                    // Compute expert relevance prior from stable summary
                    let relevance = prior.compute(slow.summary());
                    // Apply scoping: attenuate regions with low prior relevance
                    let scoped_h = apply_scoping(&h, &relevance);
                    last_pattern = activation_pattern(&scoped_h, num_experts);
                } else {
                    last_pattern = activation_pattern(&h, num_experts);
                }

                if i == prompt_tokens.len() - 1 {
                    let _logits = model.apply_output_head(&h)?;
                }
            }

            let elapsed = t_start.elapsed();
            eprintln!("[seq={},scoped=true] done in {:.3}s", seq_len, elapsed.as_secs_f64());
            last_pattern
        };
        let scoped_jaccard = jaccard_similarity(&ref_pattern, &scoped_pattern);
        println!("Scoped activation pattern: {:?}", scoped_pattern);
        println!("Scoped Jaccard similarity: {:.4}", scoped_jaccard);

        // Verdict for this length
        let improvement = scoped_jaccard - free_jaccard;
        if improvement > 0.0 {
            println!(
                "→ Scoping improved similarity by {:.4} at seq_len={}",
                improvement, seq_len
            );
        } else {
            println!(
                "→ Scoping did NOT improve similarity at seq_len={} ({:.4})",
                seq_len, improvement
            );
        }
    }

    // Overall G2 verdict
    println!("\n======================================");
    println!("G2 VERDICT: (requires manual inspection of above output)");
    println!("PASS if scoped Jaccard > free Jaccard for all long contexts.");

    Ok(())
}

fn jaccard_similarity(a: &[bool], b: &[bool]) -> f32 {
    let mut intersection = 0;
    let mut union = 0;
    for i in 0..a.len() {
        if a[i] && b[i] {
            intersection += 1;
            union += 1;
        } else if a[i] || b[i] {
            union += 1;
        }
    }
    if union == 0 {
        return 1.0;
    }
    intersection as f32 / union as f32
}
