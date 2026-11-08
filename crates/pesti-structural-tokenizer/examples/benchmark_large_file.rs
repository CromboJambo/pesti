use std::time::Instant;
use pesti_structural_tokenizer::StructuralTokenizer;

fn main() {
    // Read the large test file
    let src = std::fs::read_to_string("/tmp/large_test.rs").expect("Failed to read test file");
    println!("Source size: {} bytes, {} lines", src.len(), src.lines().count());

    let tokenizer = StructuralTokenizer::new();

    // Benchmark tokenize()
    let start = Instant::now();
    let tokens = tokenizer.tokenize(&src).expect("Tokenization failed");
    let elapsed = start.elapsed();
    println!("tokenize(): {} tokens in {:.3}s", tokens.len(), elapsed.as_secs_f64());

    // Benchmark with budget
    let budget = pesti_structural_tokenizer::Budget {
        max_tokens: 1000,
        min_body_stmts: 5,
    };
    let start = Instant::now();
    let emissions = tokenizer.tokenize_with_budget(&src, budget.clone()).expect("Budgeted tokenization failed");
    let elapsed = start.elapsed();
    println!("tokenize_with_budget(): {} emissions in {:.3}s", emissions.len(), elapsed.as_secs_f64());

    // Benchmark with semantics
    let start = Instant::now();
    let semantic_emissions = tokenizer.tokenize_with_semantics(&src, budget).expect("Semantic tokenization failed");
    let elapsed = start.elapsed();
    println!("tokenize_with_semantics(): {} emissions in {:.3}s", semantic_emissions.len(), elapsed.as_secs_f64());

    // Check if under 1 second
    if elapsed.as_secs_f64() < 1.0 {
        println!("\n✅ PASSED: Tokenization of 10K-line file completed in <1 second");
    } else {
        println!("\n❌ FAILED: Tokenization took {:.3}s (>1 second)", elapsed.as_secs_f64());
    }
}