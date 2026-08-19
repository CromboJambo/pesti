//! Example: Load Qwen2 tokenizer from GGUF and encode sample text

use qwen2_bpe::Qwen2Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load tokenizer from JSON files (extracted from GGUF)
    let vocab_path = "/tmp/qwen2_vocab_dump.json";
    let merges_path = "/tmp/qwen2_merge_pairs.json";
    
    println!("Loading Qwen2 tokenizer with merges...");
    let tokenizer = Qwen2Tokenizer::load_with_merges(vocab_path, merges_path)?;
    
    println!("✅ Vocabulary loaded: {} tokens", tokenizer.vocab_size());
    println!("✅ Merges loaded: {} pairs", tokenizer.merge_count());
    
    // Test encoding with special tokens
    let test_texts = vec!["Hello", "world", "Hello world", "The quick brown fox"];
    
    println!("\n=== Encoding Tests ===");
    for text in &test_texts {
        let tokens = tokenizer.encode_with_special(text, true, true)?;
        let decoded = tokenizer.decode(&tokens)?;
        
        println!("\nText: '{}'", text);
        println!("  Token IDs: {:?}", tokens);
        println!("  Decoded:   '{}'", decoded);
    }
    
    Ok(())
}
