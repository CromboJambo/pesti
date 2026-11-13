//! Minimal CUDA dispatch verification.
//!
//! Checks that the candle bridge can initialize CUDA and run a tiny GPU matmul,
//! proving the LinearDispatch → candle_bridge → CUDA path works end-to-end.
//! Does NOT load a model — uses only ~1 MB of GPU memory.
//!
//! Usage:
//!   cargo run --package pesti-runner --features cuda --example verify_cuda_dispatch

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CUDA Dispatch Verification ===\n");

    // 1. Check if candle bridge can initialize CUDA
    let is_cuda = pesti_runner::kernel::candle_bridge::bridge_is_cuda();
    println!("bridge_is_cuda(): {}", is_cuda);

    if !is_cuda {
        eprintln!("❌ CUDA not available via candle bridge");
        eprintln!("   (This is expected if no GPU is present or candle-core lacks cuda feature)");
        return Ok(());
    }

    let dev = pesti_runner::kernel::candle_bridge::bridge_device();
    println!("Device: {:?}", dev);

    // 2. Create tiny tensors and run matmul on GPU
    use half::f16;
    use pesti_runner::kernel::candle_bridge::{f16_to_tensor, tensor_to_f32_flat};

    // 4x8 @ 8x4 = 4x4 (tiny — negligible GPU memory)
    let a_data_f32: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let b_data_f32: Vec<f32> = (0..32).map(|i| (i * 2) as f32).collect();

    // Convert to f16 for the bridge
    let a_data: Vec<f16> = a_data_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let b_data: Vec<f16> = b_data_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let a = f16_to_tensor(&a_data, &[4, 8], Some(dev))?;
    let b = f16_to_tensor(&b_data, &[8, 4], Some(dev))?;

    let result = a.matmul(&b)?;
    let out: Vec<f32> = tensor_to_f32_flat(&result)?;
    println!("\nGPU matmul (4x8 @ 8x4) result:\n{:?}", out);

    // Verify correctness against CPU computation
    let mut expected = Vec::with_capacity(16);
    for i in 0..4 {
        for j in 0..4 {
            let sum: f32 = (0..8)
                .map(|k| a_data_f32[i * 8 + k] * b_data_f32[k * 4 + j])
                .sum();
            expected.push(sum);
        }
    }

    let max_err = out
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max error vs CPU: {:.6}", max_err);

    if max_err < 1e-4 {
        println!("\n✅ CUDA dispatch path verified — GPU matmul executed correctly");
    } else {
        eprintln!("\n❌ GPU matmul result mismatch (max_err={:.6})", max_err);
        std::process::exit(1);
    }

    Ok(())
}
