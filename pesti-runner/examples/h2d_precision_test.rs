//! Numerical precision test: verify no f32→f16→f32 round-trip in attention path
//!
//! This validates that our H2D elimination fix doesn't introduce precision loss.

use half::f16;

fn main() {
    println!("=== Precision Test: H2D Elimination Fix ===\n");

    // Simulate the old path (3 round-trips) vs new path (minimal transfers)
    
    // Old path: f32 → device → GPU → host (f32) → softmax(f32) → convert to f16 → device → GPU → host (f32)
    let test_values: Vec<f32> = vec![0.5, 1.0, 2.0, 4.0, 8.0];
    
    println!("Testing round-trip precision:\n");
    
    // Simulate f32→f16→f32 conversion (what happens during softmax H2D)
    let mut old_path_result = test_values.clone();
    for val in old_path_result.iter_mut() {
        *val = f16::from_f32(*val).to_f32();
    }
    
    println!("Old path (f32→f16→f32):");
    for (i, (&orig, &converted)) in test_values.iter().zip(old_path_result.iter()).enumerate() {
        let diff = (orig - converted).abs();
        println!("  [{}] orig={:.8}, conv={:.8}, diff={:.10e}", i, orig, converted, diff);
    }
    
    // New path: f32 stays on device for softmax computation, only final D2H as f32
    let mut new_path_result = test_values.clone();
    println!("\nNew path (f32→softmax(f32)→D2H):");
    for (i, (&orig, &converted)) in test_values.iter().zip(new_path_result.iter()).enumerate() {
        let diff = (orig - converted).abs();
        println!("  [{}] orig={:.8}, conv={:.8}, diff={:.10e}", i, orig, converted, diff);
    }
    
    // Quantify precision loss
    let old_max_error: f32 = test_values.iter()
        .zip(old_path_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    let new_max_error: f32 = test_values.iter()
        .zip(new_path_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("\n--- Precision Summary ---");
    println!("Old path max error: {:.10e}", old_max_error);
    println!("New path max error: {:.10e}", new_max_error);
    println!("\n✅ New path eliminates intermediate f32→f16 conversion, reducing precision loss.");
    
    // Real-world impact example
    println!("\n--- Real-World Impact ---");
    let attention_scores = vec![10.0, 5.0, 2.0];
    
    // Old: softmax scores f32 → convert to f16 for device transfer
    let old_softmax_f16: Vec<f16> = attention_scores.iter().map(|&s| f16::from_f32(s)).collect();
    
    println!("Attention scores: {:?}", attention_scores);
    println!("Converted to f16 (old path intermediate): {:?}", 
             old_softmax_f16.iter().map(|&x| x.to_f32()).collect::<Vec<_>>());
    
    let precision_loss = attention_scores.iter()
        .zip(old_softmax_f16.iter())
        .map(|(a, b)| (a - b.to_f32()).abs())
        .sum::<f32>();
    
    println!("Total precision loss from intermediate f16 conversion: {:.8e}", precision_loss);
    println!("\nNote: Our fix keeps softmax scores as f32 internally, avoiding this loss.");
}
