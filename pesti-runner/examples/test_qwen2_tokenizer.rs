//! Test Qwen2 tokenizer with real data

fn main() {
    let tokenizer_path = "/tmp/qwen2_tokenizer.json";

    println!("Loading Qwen2 tokenizer from {:?}", tokenizer_path);
    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

    // Test encoding
    let test_cases = vec![
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "ĠĠ ĠĠ", // Qwen2-specific merge test
    ];

    for text in test_cases {
        println!("\n📝 Input: '{}'", text);

        match tokenizer.encode(text, false) {
            Ok(encoding) => {
                let ids: Vec<u32> = encoding.get_ids().to_vec();
                println!("   Tokens: {:?}", ids);

                // Decode back
                match tokenizer.decode(&ids, false) {
                    Ok(decoded) => {
                        if decoded == text {
                            println!("   ✅ Round-trip successful");
                        } else {
                            println!("   ⚠️  Mismatch - decoded: '{}'", decoded);
                        }
                    }
                    Err(e) => println!("   ❌ Decode error: {}", e),
                }
            }
            Err(e) => println!("   ❌ Encode error: {}", e),
        }
    }
}
