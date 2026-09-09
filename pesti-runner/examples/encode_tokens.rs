//! Minimal tokenizer test - encode text to token IDs
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <text>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let text = &args[2];

    let (_config, tokenizer) = pesti_runner::load_tokenizer_from_gguf(
        std::path::Path::new(model_path),
        pesti_runner::transformer::TokenizerBackend::MistralRs,
    )?;

    // Use encode() directly (same as cpu_e2e_generate)
    let ids = tokenizer.encode(text)?;
    let id_strs: Vec<String> = ids.iter().map(|t| t.to_string()).collect();
    println!("{}", id_strs.join(","));

    Ok(())
}
