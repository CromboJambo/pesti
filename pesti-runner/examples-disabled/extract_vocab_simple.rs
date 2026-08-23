//! Extract vocabulary from GGUF using mistral.rs conversion
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("Loading GGUF: {}", gguf_path);
    
    // Use mistralrs_core's conversion function (it's public via the wrapper)
    let content = std::fs::File::open(gguf_path)?;
    let context = ggml_rs::Content::read(content)?;
    
    // Convert using mistralrs_core
    let conversion = mistralrs_core::gguf_tokenizer::convert_gguf_to_hf_tokenizer(&context)?;
    
    println!("Conversion successful!");
    
    // Get vocabulary as HashMap<String, u32>
    let vocab = conversion.tokenizer.get_vocab(true);  // with_added_tokens
    
    // Convert to Vec for sorting
    let mut items: Vec<(&String, &u32)> = vocab.iter().collect();
    items.sort_by_key(|(_, id)| *id);
    
    println!("\nVocabulary size: {}", items.len());
    println!("\nFirst 50 tokens (sorted by ID):");
    
    for (token, id) in items.iter().take(50) {
        let line = format!("{:6}: {:?}", *id, token);
        println!("{}", line);
    }
    
    // Save to file
    let json_output = serde_json::to_string_pretty(&items)?;
    fs::write("/tmp/qwen2_vocab_sorted.json", &json_output)?;
    println!("\nSaved sorted vocabulary to /tmp/qwen2_vocab_sorted.json");
    
    Ok(())
}
