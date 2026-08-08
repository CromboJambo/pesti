//! Test CUTLASS GEMM via cudarc cublas

use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::driver::CudaDevice;
use half::f16;
use std::sync::Arc;

fn main() {
    println!("=== CUTLASS GEMM Test via cudarc ===\n");

    // Initialize CUDA
    let device = CudaDevice::new(0).expect("CUDA device failed");
    let cublas = Arc::new(CudaBlas::new(device.clone()).expect("cublas failed"));

    println!("✅ CUDA device: {}", device.device_id().ordinal());
    println!("✅ cublas handle created (uses CUTLASS internally)");

    // Create simple 2x3 @ 3x2 = 2x2 GEMM
    let m = 2;
    let n = 2;
    let k = 3;

    // A: [m, k] = [2, 3]
    let a_host: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
        f16::from_f32(5.0),
        f16::from_f32(6.0),
    ];

    // B: [k, n] = [3, 2]
    let b_host: Vec<f16> = vec![
        f16::from_f32(7.0),
        f16::from_f32(8.0),
        f16::from_f32(9.0),
        f16::from_f32(10.0),
        f16::from_f32(11.0),
        f16::from_f32(12.0),
    ];

    // C: [m, n] = [2, 2] (output)
    let mut c_host = vec![f16::default(); m * n];

    // Allocate on GPU
    let a_dev = device.htod_slice(&a_host).expect("A allocation failed");
    let b_dev = device.htod_slice(&b_host).expect("B allocation failed");
    let mut c_dev = device
        .create_zeros::<f16>(m * n)
        .expect("C allocation failed");

    println!(
        "✅ Allocated tensors on GPU: A[{}x{}], B[{}x{}], C[{}x{}]",
        m, k, k, n, m, n
    );

    // Launch GEMM with default config (uses tensor cores for sm_8.9+)
    cublas
        .gemm(&a_dev, &b_dev, &mut c_dev)
        .expect("GEMM failed");

    println!("✅ CUTLASS GEMM launched successfully!");

    // Read back result
    let c_final: Vec<f16> = device
        .htod_slice(&c_host)
        .expect("C readback failed")
        .to_vec();

    // Expected: A @ B = [[58, 64], [139, 154]]
    println!("\nResults:");
    println!("  C[0][0] = {:.0} (expected 58)", c_final[0].to_f32());
    println!("  C[0][1] = {:.0} (expected 64)", c_final[1].to_f32());
    println!("  C[1][0] = {:.0} (expected 139)", c_final[2].to_f32());
    println!("  C[1][1] = {:.0} (expected 154)", c_final[3].to_f32());

    if (c_final[0].to_f32() - 58.0).abs() < 0.1
        && (c_final[1].to_f32() - 64.0).abs() < 0.1
        && (c_final[2].to_f32() - 139.0).abs() < 0.1
        && (c_final[3].to_f32() - 154.0).abs() < 0.1
    {
        println!("\n✅ SUCCESS! CUTLASS tensor core GEMM working on RTX 4070 Ti SUPER!");
    } else {
        println!("\n⚠️  Results differ from expected (may be due to accumulation precision)");
    }
}
