fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pesti_gguf::parser::parse_gguf;
    use pesti_gguf::types::GgufKvValue;
    use std::path::Path;

    let path = Path::new("/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let header = parse_gguf(path)?;

    // Find the tokens KV pair
    let tokens_kv = header.kv_pairs.iter()
        .find(|kv| kv.key == "tokenizer.ggml.tokens")
        .ok_or("missing tokenizer.ggml.tokens")?;

    println!("Tokens KV: key={}, value_type={:?}", tokens_kv.key, tokens_kv.value_type);
    
    // Try to extract the array
    if let GgufKvValue::Array(arr) = &tokens_kv.value {
        println!("Array length: {}", arr.len());
        
        // Check first few elements
        for (i, elem) in arr.iter().take(5).enumerate() {
            println!("  Element {}: type={:?} value={:?}", i, elem.value_type(), elem);
            
            if let GgufKvValue::String(s) = elem {
                println!("    -> String: '{}'", s.chars().take(20).collect::<String>());
            }
        }
        
        // Try to extract all strings
        let strings: Vec<String> = arr.iter()
            .filter_map(|v| match v {
                GgufKvValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
            
        println!("\nExtracted {} strings", strings.len());
        if !strings.is_empty() {
            println!("First 3: {:?}", &strings[..3.min(strings.len())]);
        }
    } else {
        eprintln!("ERROR: Expected Array, got {:?}", tokens_kv.value);
    }

    Ok(())
}
