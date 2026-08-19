//! Load Qwen2 vocabulary from JSON and test encoding

use qwen2_bpe::Qwen2Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab_path = "/tmp/qwen2_vocab_dump.json";
    
    println!("Loading Qwen2 tokenizer from: {}", vocab_path);
    
    let tokenizer = Qwen2Tokenizer::load_from_json(vocab_path)?;
    
    println!("✅ Tokenizer loaded!");
    println!("Vocabulary size: {} tokens", tokenizer.vocab_size());
    
    // Test encoding
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox",
        "Qwen2 tokenizer",
    ];
    
    for text in test_texts {
        println!("\n🔤 Encoding: '{}'", text);
        
        let tokens = tokenizer.encode(text)?;
        println!("  Token IDs: {:?}", tokens);
        println!("  Token count: {}", tokens.len());
        
        // Decode back (will be approximate since we don't have merges yet)
        let decoded = tokenizer.decode(&tokens)?;
        println!("  Decoded: '{}'", decoded);
    }
    
    Ok(())
}
