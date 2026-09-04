//! Profile ONE decode step on the GPU dispatch path.
//!
//! Times: model build, first forward (KV init), then per-step timing of
//! forward_with_dispatch at increasing positions. Prints per-step wall time
//! and the GPU fallback counter so we can see whether the step time is
//! dominated by weight re-upload/conversion (constant ~3.5s) or grows with
//! seq_len (attention/KV).
//!
//! Usage: cargo run -p pesti-runner --release --features cuda --example profile_decode_step
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    println!("[P] loaded weights in {:.2}s", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    println!("[P] built model in {:.2}s", t1.elapsed().as_secs_f32());

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_cfg, tokenizer) =
        pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path), backend)?;

    let prompt_tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.")?;
    println!("[P] prompt tokens: {}", prompt_tokens.len());

    // Prefill the prompt (first call initializes GPU KV caches).
    let mut pos = 0;
    let t2 = Instant::now();
    let mut logits: Vec<f32> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        logits = model.forward_with_dispatch(&hidden, i)?;
        pos = i + 1;
    }
    println!(
        "[P] prefill {} tokens in {:.2}s ({:.2} tok/s)",
        prompt_tokens.len(),
        t2.elapsed().as_secs_f32(),
        prompt_tokens.len() as f64 / t2.elapsed().as_secs_f64()
    );

    // Decode N steps, timing each.
    let n_steps = 8usize;
    let mut times = Vec::new();
    for s in 0..n_steps {
        let next = logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(pi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (pi, bv)
            }
        });
        let (tok_id, _) = next;
        let tok_id = tok_id as u32;
        let hidden = model.embed(tok_id, pos)?;
        let ts = Instant::now();
        logits = model.forward_with_dispatch(&hidden, pos)?;
        let dt = ts.elapsed().as_secs_f32();
        times.push(dt);
        println!(
            "[P] step {s} pos={} tok={tok_id}: {:.3}s ({:.2} tok/s)",
            pos,
            dt,
            1.0 / dt
        );
        pos += 1;
    }

    let fb = model
        .dispatch
        .as_ref()
        .map(|c| c.gpu_fallback_count())
        .unwrap_or(0);
    println!("[P] gpu_fallback_count = {fb}");
    let median = {
        let mut v = times.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!(
        "[P] median decode step: {:.3}s ({:.2} tok/s)",
        median,
        1.0 / median
    );
    let _ = rand::rngs::StdRng::seed_from_u64(42);
    Ok(())
}
