//! Minimal tokenizer test - encode text to token IDs
use std::env;
use pesti_runner::{load_tokenizer_from_gguf, TokenizerBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <text>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let text = &args[2];

    let (_config, tokenizer) = load_tokenizer_from_gguf(
        std::path::Path::new(model_path),
        TokenizerBackend::MistralRs,
    )?;

    // Match cpu_e2e_generate: encode_with_special(text, true, false)
    let ids = tokenizer.encode_with_special(text, true, false)?;
    println!("{}", ids.join(","));

    Ok(())
}
