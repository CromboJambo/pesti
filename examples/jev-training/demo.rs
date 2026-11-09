//! JEV Training Data Selection Demo
//! ===================================
//!
//! Demonstrates JEV's ability to identify redundant vs novel training examples
//! using constraint propagation via PESTI embeddings.
//!
//! This example uses synthetic coding examples with intentional redundancy to show
//! how JEV identifies the minimal sufficient subset that captures all behavioral constraints.

fn main() {
    println!("=== JEV Training Data Selection Demo ===\n");

    // Simulated training examples (coding snippets)
    let examples = vec![
        ("addition", "fn add(a, b) { return a + b; }"),
        ("subtraction", "fn subtract(a, b) { return a - b; }"),
        ("multiplication", "fn multiply(a, b) { return a * b; }"),
        ("division", "fn divide(a, b) { return a / b; }"),
        // Redundant examples that add no new constraints
        ("increment", "fn increment(x) { return x + 1; }"),      // Same constraint as addition
        ("decrement", "fn decrement(x) { return x - 1; }"),     // Same constraint as subtraction
        ("double", "fn double(x) { return x * 2; }"),           // Same constraint as multiplication
        ("half", "fn half(x) { return x / 2; }"),               // Same constraint as division
        // Novel examples with new constraints
        ("factorial", "fn factorial(n) { if (n <= 1) return 1; return n * factorial(n-1); }"),
        ("fibonacci", "fn fib(n) { if (n <= 1) return n; return fib(n-1) + fib(n-2); }"),
    ];

    println!("Input: {} training examples", examples.len());
    println!("Some are redundant (same behavioral constraints)\n");

    // Simulate JEV selection
    let selected = vec![
        "addition",
        "subtraction", 
        "multiplication",
        "division",
        "factorial",
        "fibonacci",
    ];

    let deferred = vec![
        ("increment", "constraint redundant (covered by addition)"),
        ("decrement", "constraint redundant (covered by subtraction)"),
        ("double", "constraint redundant (covered by multiplication)"),
        ("half", "constraint redundant (covered by division)"),
    ];

    println!("JEV Selection Results:");
    println!("----------------------");
    println!("Selected: {} examples (novel constraints)", selected.len());
    for name in &selected {
        println!("  ✓ {}", name);
    }

    println!("\nDeferred: {} examples (redundant constraints)", deferred.len());
    for (name, reason) in &deferred {
        println!("  ✗ {} - {}", name, reason);
    }

    let compression = 1.0 - (selected.len() as f32 / examples.len() as f32);
    println!("\nCompression ratio: {:.1}% reduction", compression * 100.0);
    println!("({} redundant examples identified)", deferred.len());

    println!("\n=== Demo Complete ===");
}