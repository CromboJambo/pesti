//! Decode token IDs using PESTI's tokenizer.
//! Usage: decode_tokens <model.gguf> <id1,id2,...>

use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <id1,id2,...>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ids_str = &args[2];
    let ids: Vec<u32> = ids_str
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();

    // Load tokenizer from GGUF
    let (_config, tokenizer) = pesti_runner::load_tokenizer_from_gguf(
        std::path::Path::new(model_path),
        pesti_runner::TokenizerBackend::MistralRs,
    )?;

    let text = tokenizer.decode(&ids, true).map_err(|e| format!("decode error: {}", e))?;
    println!("{}", text);

    Ok(())
}
