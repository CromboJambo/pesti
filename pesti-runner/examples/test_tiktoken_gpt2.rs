//! Test with tiktoken-rs using GPT-2 model (should work).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = "The quick brown fox jumps over the lazy dog.";

    println!("=== Testing tiktoken-rs with GPT-2 ===");
    println!("Text: {}", text);

    match tiktoken_rs::get_bpe_from_model("gpt2") {
        Ok(tokenizer) => {
            let tokens = tokenizer.encode_with_special_tokens(text);
            println!("✅ tiktoken-rs encoding successful!");
            println!("Token count: {}", tokens.len());
            println!("First 10 token IDs: {:?}", &tokens[..tokens.len().min(10)]);

            // Decode back to text
            let decoded = tokenizer.decode(&tokens)?;
            println!("Decoded: {}", decoded);
        }
        Err(e) => {
            println!("❌ tiktoken-rs failed: {}", e);
        }
    }

    Ok(())
}
