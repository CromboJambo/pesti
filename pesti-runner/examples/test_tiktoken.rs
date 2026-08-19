//! Test with tiktoken-rs (works for GPT-2 style BPE).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = "The quick brown fox jumps over the lazy dog.";
    
    println!("=== Testing tiktoken-rs ===");
    println!("Text: {}", text);
    
    // tiktoken-rs requires a model name (e.g., "cl100k_base" for GPT-3.5/4)
    // For Qwen2, we need to use the correct encoding
    
    match tiktoken_rs::get_bpe_from_model("qwen2") {
        Ok(tokenizer) => {
            let tokens = tokenizer.encode_with_special_tokens(text);
            println!("✅ tiktoken-rs encoding successful!");
            println!("Token count: {}", tokens.len());
            println!("First 10 token IDs: {:?}", &tokens[..tokens.len().min(10)]);
            
            // Decode back to text
            let decoded = tokenizer.decode(&tokens)?;
            println!("Decoded: {}", decoded);
        },
        Err(e) => {
            println!("❌ tiktoken-rs failed: {}", e);
            println!("Note: 'qwen2' might not be a built-in model in tiktoken-rs");
        }
    }

    Ok(())
}
