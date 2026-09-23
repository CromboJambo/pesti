use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file>", args[0]);
        std::process::exit(1);
    }
    let path = &args[1];
    let src = fs::read_to_string(path).expect("failed to read file");
    
    println!("File: {}", path);
    println!("Bytes: {}", src.len());
    println!("Lines: {}", src.lines().count());
    
    let tokenizer = pesti_structural_tokenizer::StructuralTokenizer::new();
    
    let start = std::time::Instant::now();
    match tokenizer.tokenize(&src) {
        Ok(tokens) => {
            let elapsed = start.elapsed();
            println!("Tokens: {}", tokens.len());
            println!("Time: {:.2?}", elapsed);
            println!("Compression: {:.1}x", src.len() as f64 / tokens.len() as f64);
            
            // Count token kinds
            let mut kind_counts = std::collections::HashMap::new();
            for t in &tokens {
                *kind_counts.entry(format!("{:?}", t.kind)).or_insert(0) += 1;
            }
            println!("\nToken type distribution (top 15):");
            let mut sorted: Vec<_> = kind_counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            for (kind, count) in sorted.iter().take(15) {
                println!("  {:4} {}", count, kind);
            }
        }
        Err(e) => {
            let elapsed = start.elapsed();
            eprintln!("ERROR after {:.2?}: {}", elapsed, e);
            std::process::exit(1);
        }
    }
}
