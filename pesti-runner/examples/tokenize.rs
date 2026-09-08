//! Tokenize text and print token IDs.
//! Usage: tokenize <model.gguf> <text>

use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <text>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let text = &args[2];

    let (config, tokenizer) = pesti_runner::load_tokenizer_from_gguf(
        std::path::Path::new(model_path),
        pesti_runner::TokenizerBackend::MistralRs,
    )?;

    let tokens = tokenizer.encode(text, true)?;
    let ids: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    println!("{}", ids.join(","));

    Ok(())
}
