//! Minimal forward-only test — no bench, no comparison.
use pesti_runner::quantized_linear::QuantizedLinear;
use pesti_runner::tile_dequant::QuantDtype;
use std::io::Write;

fn main() {
    eprintln!("Starting...\n");
    std::io::stderr().flush().unwrap();

    let in_f = 500;
    let out_f = 500;
    let row_bytes = ((in_f + 31) / 32) * 18; // Q4_0: 16 blocks * 18 = 288

    let raw = vec![0u8; out_f * row_bytes];
    let ql = QuantizedLinear::new(raw, QuantDtype::Q4_0, in_f, out_f, None);
    let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.01).sin()).collect();

    eprintln!("Calling ql.forward()...");
    std::io::stderr().flush().unwrap();

    let result = ql.forward(&x, 1);
    eprintln!("Done! result.len={}", result.len());
    std::io::stderr().flush().unwrap();

    // Verify output is all zeros (input * zero weights = zero)
    let all_zero = result.iter().all(|&v| v == 0.0);
    eprintln!("All zero: {}", all_zero);
    std::io::stderr().flush().unwrap();
}
