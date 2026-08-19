//! Performance benchmark: qwen2-bpe vs reference

use qwen2_bpe::Qwen2Tokenizer;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load tokenizer
    let vocab_path = "/tmp/qwen2_vocab_dump.json";
    let merges_path = "/tmp/qwen2_merge_pairs.json";
    
    println!("Loading Qwen2 tokenizer with merges...");
    let tokenizer = Qwen2Tokenizer::load_with_merges(vocab_path, merges_path)?;
    
    // Test texts of varying lengths
    let test_texts = vec![
        "Hello",
        "The quick brown fox jumps over the lazy dog",
        "Qwen2.5-0.5B is a small but powerful language model with 50k vocabulary and 151k merge pairs.",
        "Infor Syteline ERP system uses UTF-16 encoding which can cause newline corruption in Excel exports.",
    ];
    
    println!("\n=== Performance Benchmark ===\n");
    
    for text in &test_texts {
        // Warm-up run
        let _ = tokenizer.encode_with_special(text, false, false);
        
        // Measure encode time
        let start = Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let _ = tokenizer.encode_with_special(text, false, false);
        }
        let duration = start.elapsed();
        let avg_ns = duration.as_nanos() / iterations as u128;
        
        println!("Text: '{}'", text);
        println!("  Length: {} chars", text.len());
        println!("  Iterations: {}", iterations);
        println!("  Total time: {:.2} ms", duration.as_millis());
        println!("  Avg per encode: {:.2} μs", avg_ns as f64 / 1000.0);
        println!();
    }
    
    Ok(())
}
