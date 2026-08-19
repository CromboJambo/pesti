//! Load Qwen2 vocabulary and merge pairs from JSON files

use qwen2_bpe::Qwen2Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab_path = "/tmp/qwen2_vocab_dump.json";
    let merges_path = "/tmp/qwen2_merge_pairs.json";
    
    println!("Loading Qwen2 tokenizer with merges...");
    let tokenizer = Qwen2Tokenizer::load_with_merges(vocab_path, merges_path)?;
    
    println!("✅ Vocabulary loaded: {} tokens", tokenizer.vocab_size());
    
    // Test encoding a simple string
    let test_text = "Hello world";
    println!("\nEncoding: '{}'", test_text);
    let tokens = tokenizer.encode(test_text)?;
    println!("→ Token IDs: {:?}", tokens);
    
    // Test decoding back
    let decoded = tokenizer.decode(&tokens)?;
    println!("Decoded: '{}'", decoded);
    
    Ok(())
}
