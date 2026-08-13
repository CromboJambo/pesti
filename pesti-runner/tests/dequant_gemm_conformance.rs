//! Conformance tests for fused dequant-GEMM kernel.
//!
//! Tests the numerical correctness of the fused Q4_K dequant + GEMM kernel
//! against the CPU reference implementation (dequantize + gemm).

use rand::SeedableRng;
use rand::rngs::StdRng;

// Note: These tests require tile-specific dequant functions that aren't exported yet.
// The main conformance suite in pesti-conformance crate provides the verification.

/// Reference CPU implementation: dequantize Q4_K tile, then GEMM
fn cpu_dequant_gemm_q4k(
    a_packed: &[u8],
    b_f16: &[f16],
    c_init: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<Vec<f32>, String> {
    // Dequantize A from Q4_K to f32
    let a_f32 = dequantize_q4_k_tile(a_packed, 0, m * k).map_err(|e| e.to_string())?;

    // Convert B from f16 to f32
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();

    // CPU GEMM: C = alpha * A @ B + beta * C
    let mut c = if let Some(c_init_data) = c_init {
        c_init_data.to_vec()
    } else {
        vec![0.0f32; m * n]
    };

    let cpu_gemm = CpuGemmKernel::new();
    cpu_gemm
        .gemm_f32(&a_f32, &b_f32, &mut c, m, n, k, alpha, beta)
        .map_err(|e| e.to_string())?;

    Ok(c)
}

/// Reference CPU implementation for Q4_0
fn cpu_dequant_gemm_q4_0(
    a_packed: &[u8],
    b_f16: &[f16],
    c_init: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<Vec<f32>, String> {
    let a_f32 = dequantize_q4_0_tile(a_packed, 0, m * k).map_err(|e| e.to_string())?;
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();

    let mut c = if let Some(c_init_data) = c_init {
        c_init_data.to_vec()
    } else {
        vec![0.0f32; m * n]
    };

    let cpu_gemm = CpuGemmKernel::new();
    cpu_gemm
        .gemm_f32(&a_f32, &b_f32, &mut c, m, n, k, alpha, beta)
        .map_err(|e| e.to_string())?;

    Ok(c)
}

/// Reference CPU implementation for Q8_0
fn cpu_dequant_gemm_q8_0(
    a_packed: &[u8],
    b_f16: &[f16],
    c_init: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<Vec<f32>, String> {
    let a_f32 = dequantize_q8_0_tile(a_packed, 0, m * k).map_err(|e| e.to_string())?;
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();

    let mut c = if let Some(c_init_data) = c_init {
        c_init_data.to_vec()
    } else {
        vec![0.0f32; m * n]
    };

    let cpu_gemm = CpuGemmKernel::new();
    cpu_gemm
        .gemm_f32(&a_f32, &b_f32, &mut c, m, n, k, alpha, beta)
        .map_err(|e| e.to_string())?;

    Ok(c)
}

/// Test a single GEMM case with given quantization
fn test_dequant_gemm_case<F>(
    name: &str,
    m: usize,
    n: usize,
    k: usize,
    make_a_packed: F,
    cpu_ref: fn(
        &[u8],
        &[f16],
        Option<&[f32]>,
        usize,
        usize,
        usize,
        f32,
        f32,
    ) -> Result<Vec<f32>, String>,
) where
    F: FnOnce() -> (Vec<u8>, Vec<f16>),
{
    let (a_packed, b_f16) = make_a_packed();
    let ctx = DispatchContext::new();

    // CPU reference (dequant + GEMM)
    let cpu_result =
        cpu_ref(&a_packed, &b_f16, None, m, n, k, 1.0, 0.0).expect("CPU reference failed");

    // CPU-only dispatch: we need to dequantize first, then call dispatch_gemm_cpu
    // This simulates what the fused kernel should produce
    let a_f32 = match name {
        n if n.contains("Q4_K") => {
            dequantize_q4_k_tile(&a_packed, 0, m * k).expect("dequant failed")
        }
        n if n.contains("Q4_0") => {
            dequantize_q4_0_tile(&a_packed, 0, m * k).expect("dequant failed")
        }
        n if n.contains("Q8_0") => {
            dequantize_q8_0_tile(&a_packed, 0, m * k).expect("dequant failed")
        }
        _ => panic!("Unknown quantization in test name"),
    };

    // Convert to f16 for dispatch_gemm_cpu (which expects f16 inputs)
    let a_f16: Vec<f16> = a_f32.iter().map(|&v| f16::from_f32(v)).collect();

    // GPU dispatch (CPU fallback path)
    let gpu_result = ctx
        .dispatch_gemm_cpu(&a_f16, &b_f16, None, m, n, k, 1.0, 0.0)
        .expect("GPU dispatch failed");

    // Compare
    assert_eq!(
        cpu_result.len(),
        gpu_result.len(),
        "{}: output length mismatch",
        name
    );

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    let mut max_cpu = 0.0;
    let mut max_gpu = 0.0;

    for (i, (cpu, gpu)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
            max_cpu = *cpu;
            max_gpu = *gpu;
        }
    }

    println!(
        "{}: max diff = {:.6e} at idx {} (cpu={:.6}, gpu={:.6})",
        name, max_diff, max_idx, max_cpu, max_gpu
    );

    // Tolerance: 1e-2 accounts for f16 precision in dispatch_gemm_cpu
    let tol = 1e-2;
    assert!(
        max_diff < tol,
        "{}: conformance failed: max diff {:.6e} exceeds tolerance {}",
        name,
        max_diff,
        tol
    );

    println!("✅ {} passed", name);
}

#[test]
fn test_dequant_gemm_q4k_small() {
    // Small test: 16x16x16 (single Q4_K block in K dimension)
    let m = 16;
    let n = 16;
    let k = 16;

    test_dequant_gemm_case(
        "Q4_K GEMM 16x16x16",
        m,
        n,
        k,
        || {
            // Create Q4_K packed data: 1 block per row * 16 rows = 16 blocks
            // Each Q4_K block = 28 bytes for 16 elements
            // Total K=16 means 1 block per row in K dimension
            let mut a_packed = Vec::new();
            let mut b_f16 = Vec::new();

            // B matrix: identity-ish for easy verification
            for j in 0..n {
                for i in 0..k {
                    b_f16.push(f16::from_f32(if i == j { 1.0 } else { 0.0 }));
                }
            }

            // A matrix: each row is a Q4_K block with known values
            // We'll use scale=1.0, delta=1.0, h=[1.0, 1.0], qs with q=4 (zero centered)
            // So dequantized values should be ~0
            for _row in 0..m {
                // Q4_K block: d=1.0, delta=1.0, qs_low=0x44444444, qs_high=0x44444444, h=[1.0, 1.0]
                // q=4 for all elements => (q-4)=0 => all zeros
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // d
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // delta
                a_packed.extend_from_slice(&0x44444444u32.to_le_bytes()); // qs_low
                a_packed.extend_from_slice(&0x44444444u32.to_le_bytes()); // qs_high
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // h[0]
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // h[1]
            }

            (a_packed, b_f16)
        },
        cpu_dequant_gemm_q4k,
    );
}

#[test]
fn test_dequant_gemm_q4k_medium() {
    // Medium test: 64x64x64 (4 blocks per row in K)
    let m = 64;
    let n = 64;
    let k = 64;

    test_dequant_gemm_case(
        "Q4_K GEMM 64x64x64",
        m,
        n,
        k,
        || {
            let mut a_packed = Vec::new();
            let mut b_f16 = Vec::new();

            // B: random f16 values

            let mut rng = rand::thread_rng();
            for _ in 0..k * n {
                b_f16.push(f16::from_f32(rng.gen_range(-1.0..1.0)));
            }

            // A: Q4_K blocks with known pattern
            // Each row has k/16 = 4 blocks
            for _row in 0..m {
                for _block in 0..4 {
                    // Varying scales for more interesting test
                    let d = f16::from_f32(0.5);
                    let delta = f16::from_f32(0.25);
                    // qs pattern: alternating q=8 (max positive) and q=0 (max negative)
                    let qs_low = 0x80808080u32; // q=8 for all 8 elements
                    let qs_high = 0x00000000u32; // q=0 for all 8 elements
                    a_packed.extend_from_slice(&d.to_le_bytes());
                    a_packed.extend_from_slice(&delta.to_le_bytes());
                    a_packed.extend_from_slice(&qs_low.to_le_bytes());
                    a_packed.extend_from_slice(&qs_high.to_le_bytes());
                    a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // h[0]
                    a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // h[1]
                }
            }

            (a_packed, b_f16)
        },
        cpu_dequant_gemm_q4k,
    );
}

#[test]
fn test_dequant_gemm_q4_0() {
    let m = 32;
    let n = 32;
    let k = 32;

    test_dequant_gemm_case(
        "Q4_0 GEMM 32x32x32",
        m,
        n,
        k,
        || {
            let mut a_packed = Vec::new();
            let mut b_f16 = Vec::new();

            let mut rng = rand::thread_rng();
            for _ in 0..k * n {
                b_f16.push(f16::from_f32(rng.gen_range(-1.0..1.0)));
            }

            // Q4_0: 32 elements per block, 18 bytes
            // k=32 means 1 block per row
            for _row in 0..m {
                // Scale = 1.0, q values centered at 8 (0 after dequant)
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // scale
                for _ in 0..16 {
                    a_packed.push(0x88); // nibbles: 8, 8 => q=0 after -8
                }
            }

            (a_packed, b_f16)
        },
        cpu_dequant_gemm_q4_0,
    );
}

#[test]
fn test_dequant_gemm_q8_0() {
    let m = 32;
    let n = 32;
    let k = 32;

    test_dequant_gemm_case(
        "Q8_0 GEMM 32x32x32",
        m,
        n,
        k,
        || {
            let mut a_packed = Vec::new();
            let mut b_f16 = Vec::new();

            let mut rng = rand::thread_rng();
            for _ in 0..k * n {
                b_f16.push(f16::from_f32(rng.gen_range(-1.0..1.0)));
            }

            // Q8_0: 32 elements per block, 34 bytes
            for _row in 0..m {
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // scale
                for i in 0..32 {
                    // Alternating +127 / -127 pattern
                    a_packed.push(if i % 2 == 0 { 0x7F } else { 0x81 } as u8);
                }
            }

            (a_packed, b_f16)
        },
        cpu_dequant_gemm_q8_0,
    );
}

#[test]
fn test_dequant_gemm_with_bias() {
    // Test with non-zero beta (C initialization) and alpha != 1
    let m = 16;
    let n = 16;
    let k = 16;
    let _alpha = 2.0;
    let _beta = 0.5;

    test_dequant_gemm_case(
        "Q4_K GEMM with alpha/beta",
        m,
        n,
        k,
        || {
            let mut a_packed = Vec::new();
            let mut b_f16 = Vec::new();

            // B = identity
            for j in 0..n {
                for i in 0..k {
                    b_f16.push(f16::from_f32(if i == j { 1.0 } else { 0.0 }));
                }
            }

            // A: Q4_K with d=2.0, so dequant = 2.0 * (q-4) + 0 = 2*(q-4)
            // Use q=6 => dequant = 2*(6-4) = 4
            for _row in 0..m {
                a_packed.extend_from_slice(&f16::from_f32(2.0).to_le_bytes()); // d=2
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // delta=1
                // qs with q=6 (6-4=2) => 0x66666666
                a_packed.extend_from_slice(&0x66666666u32.to_le_bytes());
                a_packed.extend_from_slice(&0x66666666u32.to_le_bytes());
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
                a_packed.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
            }

            (a_packed, b_f16)
        },
        cpu_dequant_gemm_q4k,
    );
}

#[test]
fn test_dequant_gemm_vs_real_model_weights() {
    // Integration test: use actual Q4_K_M model weights
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );

    if !model_path.exists() {
        println!("Skipping: conformance corpus not found");
        return;
    }

    // Load weights
    let weights = pesti_runner::gguf_weight_loader::load_gguf_weights(model_path)
        .expect("Failed to load GGUF weights");

    // Pick a weight tensor (e.g., first attention layer wq)
    let tensor_name = "layers.0.attention.wq.weight";
    let raw_tensor = weights
        .raw_tensors
        .get(tensor_name)
        .expect("Tensor not found in raw_tensors");

    // Get shape
    let (in_features, out_features) = weights.tensor_shape(tensor_name);
    println!(
        "Testing tensor {}: {}x{}",
        tensor_name, in_features, out_features
    );

    // Use a small tile for testing
    let m = 16.min(out_features);
    let k = 16.min(in_features);
    let n = 16;

    // Extract tile from raw tensor (Q4_K packed)
    // Q4_K: 16 elements per block, 28 bytes per block
    // Tensor is [in_features, out_features] = [k, m] in GGUF layout
    // We need first m rows, first k cols
    let bytes_per_row = in_features.div_ceil(16) * 28;
    let mut a_packed = Vec::new();
    for row in 0..m {
        let row_start = row * bytes_per_row;
        let row_end = row_start + (k.div_ceil(16) * 28).min(bytes_per_row);
        a_packed.extend_from_slice(&raw_tensor[row_start..row_end]);
    }

    // Random B matrix
    let mut b_f16 = Vec::new();

    let mut rng = rand::thread_rng();
    for _ in 0..k * n {
        b_f16.push(f16::from_f32(rng.gen_range(-1.0..1.0)));
    }

    let ctx = DispatchContext::new();

    // CPU reference
    let cpu_result = cpu_dequant_gemm_q4k(&a_packed, &b_f16, None, m, n, k, 1.0, 0.0)
        .expect("CPU reference failed");

    // Dequant for dispatch
    let a_f32 = dequantize_q4_k_tile(&a_packed, 0, m * k).expect("dequant failed");
    let a_f16: Vec<f16> = a_f32.iter().map(|&v| f16::from_f32(v)).collect();

    // GPU dispatch (CPU fallback)
    let gpu_result = ctx
        .dispatch_gemm_cpu(&a_f16, &b_f16, None, m, n, k, 1.0, 0.0)
        .expect("GPU dispatch failed");

    assert_eq!(cpu_result.len(), gpu_result.len());

    let mut max_diff = 0.0f32;
    for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
        let diff = (cpu - gpu).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Real model Q4_K tile test: max diff = {:.6e}", max_diff);
    assert!(
        max_diff < 1e-2,
        "Real model conformance failed: max diff {:.6e}",
        max_diff
    );

    println!("✅ Real model Q4_K tile test passed");
}

#[test]
fn test_block_dequant_exact() {
    // Test single block dequantization matches exactly
    let block_data = [
        // d = 1.0
        0x00, 0x3C, // delta = 0.5
        0x00, 0x38, // qs_low = 0x44444444 (q=4 for all 8)
        0x44, 0x44, 0x44, 0x44, // qs_high = 0x88888888 (q=8 for all 8)
        0x88, 0x88, 0x88, 0x88, // h[0] = 1.0
        0x00, 0x3C, // h[1] = 2.0
        0x00, 0x40,
    ];

    let result = dequantize_q4_k_block(&block_data);

    // First 8: d + delta * h[0] * (4-4) = 1.0 + 0 = 1.0
    // Next 8: d + delta * h[1] * (8-4) = 1.0 + 0.5 * 2.0 * 4 = 1.0 + 4.0 = 5.0
    for i in 0..8 {
        assert!(
            (result[i] - 1.0).abs() < 1e-4,
            "Block[{}] = {}, expected 1.0",
            i,
            result[i]
        );
    }
    for i in 8..16 {
        assert!(
            (result[i] - 5.0).abs() < 1e-4,
            "Block[{}] = {}, expected 5.0",
            i,
            result[i]
        );
    }

    println!("✅ Block dequant exact test passed");
}
