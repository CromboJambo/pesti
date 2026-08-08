//! Minimal dequant-only test — no gemm, no Linear.

use pesti_runner::tile_dequant::{QuantDtype, dequantize_q4_k_tile};

fn main() {
    println!("=== Dequant-only Diagnostic ===\n");

    // Q4_K: 28 bytes per block, 16 elements per block
    let in_f = 300;
    let out_f = 300;
    let row_bytes = ((in_f + 15) / 16) * 28; // 532 for Q4_K with 300 elements
    let total_bytes = out_f * row_bytes; // 159600

    println!("Total bytes: {}, row_bytes: {}", total_bytes, row_bytes);

    let raw = vec![0u8; total_bytes];

    // Tile 0: col_start=0, tile_cols=256
    let col_start: usize = 0;
    let tile_cols: usize = 256;
    let elements_per_block: usize = 16;
    let bytes_per_block: usize = 28;

    let block_idx = col_start / elements_per_block; // 0
    let block_byte_offset = block_idx * bytes_per_block; // 0
    let elements_needed = (col_start % elements_per_block) + tile_cols; // 0 + 256 = 256
    let blocks_needed = (elements_needed + elements_per_block - 1) / elements_per_block; // 17
    let bytes_needed = blocks_needed * bytes_per_block; // 17 * 28 = 476

    println!(
        "Tile 0: block_idx={}, blocks_needed={}, bytes_needed={}",
        block_idx, blocks_needed, bytes_needed
    );

    // Test on first row
    let row_data = &raw[0..row_bytes]; // 525 bytes
    println!(
        "row_data.len()={}, block_byte_offset={}",
        row_data.len(),
        block_byte_offset
    );

    if block_byte_offset < row_data.len() {
        let block_data = &row_data[block_byte_offset..];
        println!("block_data.len()={}", block_data.len());

        // This should NOT crash - all zeros
        match dequantize_q4_k_tile(block_data, 0, blocks_needed * elements_per_block) {
            Ok(result) => println!("dequantize OK: len={}", result.len()),
            Err(e) => println!("dequantize ERROR: {}", e),
        }
    }

    // Now test the actual QuantizedLinear::dequantize_tile
    println!("\n--- Testing QuantizedLinear::forward (no gemm) ---");
    let ql = pesti_runner::quantized_linear::QuantizedLinear::new(
        raw,
        QuantDtype::Q4_K,
        in_f,
        out_f,
        None,
    );
    let x = vec![1.0f32; in_f];

    // Catch any panic
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ql.forward(&x, 1))) {
        Ok(result) => println!("forward OK: len={}", result.len()),
        Err(_) => println!("forward PANICKED"),
    }

    println!("\n=== Done ===");
}
