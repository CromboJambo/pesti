//! Parametric coherence check: load a GGUF, generate, print text.
//! Usage: cargo run -p pesti-runner --release --features cuda --example coherence_check -- <path> [max_tokens]

use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: coherence_check <model.gguf> [max_tokens]");
        std::process::exit(2);
    }
    let model_path = args[1].clone();
    let max_tokens = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);

    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    println!(
        "[load] {:.2}s tensors={}",
        t0.elapsed().as_secs_f32(),
        weights.tensors.len()
    );

    let t1 = Instant::now();
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    println!("[build] {:.2}s", t1.elapsed().as_secs_f32());

    // DEBUG: Check dispatch
    if model.dispatch.is_some() {
        println!("[DEBUG] Dispatch context IS initialized");
        let ctx = model.dispatch.as_ref().unwrap();
        println!("[DEBUG] GPU available: {}", ctx.gpu_available());
    } else {
        println!("[DEBUG] Dispatch context is NONE - CPU only!");
    }

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_cfg, tokenizer) =
        pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(&model_path), backend)?;
    println!("[tokenizer] ok");

    let prompt = "The quick brown fox jumps over the lazy dog.";
    let prompt_tokens = if let Ok(ids) = std::env::var("PESTI_PROMPT_TOKENS") {
        let parsed: Vec<u32> = ids
            .split(',')
            .map(|s| s.trim().parse().expect("bad token id"))
            .collect();
        println!(
            "[prompt from PESTI_PROMPT_TOKENS] {} tokens: {:?}",
            parsed.len(),
            parsed
        );
        parsed
    } else {
        let enc = tokenizer.encode(prompt)?;
        println!("[encode] {} tokens: {:?}", enc.len(), enc);
        enc
    };

    model.reset_cpu_kv_caches();
    let sampling_config = pesti_runner::transformer::SamplingConfig {
        temperature: 0.0,
        top_p: 0.9,
        top_k: 40,
        seed: Some(42),
    };

    let t3 = Instant::now();

    // DEBUG: Enable verbose tracing for this run
    unsafe {
        std::env::set_var("RUST_LOG", "debug,pesti_runner=trace");
    }

    let generated_tokens = model.generate(
        &prompt_tokens,
        max_tokens,
        &sampling_config,
        &mut rand::rngs::StdRng::seed_from_u64(42),
        &[0],
    )?;
    let gen_time = t3.elapsed();

    let decoded = tokenizer.decode(&generated_tokens)?;
    println!(
        "\n=== {} tokens in {:.3}s ({:.2} tok/s) ===",
        generated_tokens.len(),
        gen_time.as_secs_f32(),
        generated_tokens.len() as f64 / gen_time.as_secs_f64()
    );
    println!("TEXT: {}", decoded.chars().take(300).collect::<String>());
    println!("GEN_TOKEN_IDS: {:?}", generated_tokens);
    Ok(())
}
