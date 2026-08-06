//! Minimal test for QuantizedLinear dequantization correctness.

use pesti_runner::quantized_linear::QuantizedLinear;
use pesti_runner::tile_dequant::QuantDtype;

fn main() {
    println!("=== QuantizedLinear Smoke Test ===\n");

    // Test 1: Q4_0 small matrix
    let in_f = 32; // one block
    let out_f = 2;
    // Q4_0: 18 bytes per 32 elements. For 2 rows × 32 cols = 2 × 18 = 36 bytes
    let raw = vec![0u8; 36];
    let ql = QuantizedLinear::new(raw, QuantDtype::Q4_0, in_f, out_f, None);
    println!("Q4_0: data_len={}, row_bytes={}", ql.data.len(), ql.data.len() / out_f);
    let x = vec![1.0f32; in_f];
    let result = ql.forward(&x, 1);
    println!("Q4_0 output: len={}, all_zero={}", result.len(), result.iter().all(|&v| v == 0.0));

    // Test 2: Q4_K small matrix
    let in_f = 16; // one block
    let out_f = 2;
    // Q4_K: 28 bytes per 16 elements. For 2 rows × 16 cols = 2 × 28 = 56 bytes
    let raw = vec![0u8; 56];
    let ql = QuantizedLinear::new(raw, QuantDtype::Q4_K, in_f, out_f, None);
    println!("\nQ4_K: data_len={}, row_bytes={}", ql.data.len(), ql.data.len() / out_f);
    let x = vec![1.0f32; in_f];
    let result = ql.forward(&x, 1);
    println!("Q4_K output: len={}, all_zero={}", result.len(), result.iter().all(|&v| v == 0.0));

    // Test 3: Medium Q4_K matrix
    let in_f = 300;
    let out_f = 300;
    let row_bytes = ((in_f + 15) / 16) * 28; // Q4_K: 28 bytes per 16 elements
    let raw = vec![0u8; out_f * row_bytes];
    let ql = QuantizedLinear::new(raw, QuantDtype::Q4_K, in_f, out_f, None);
    println!("\nQ4_K large: data_len={}, row_bytes={}", ql.data.len(), row_bytes);
    let x = vec![1.0f32; in_f];
    let result = ql.forward(&x, 1);
    println!("Q4_K large output: len={}, all_zero={}", result.len(), result.iter().all(|&v| v == 0.0));

    println!("\n=== Smoke Test Complete ===");
}
