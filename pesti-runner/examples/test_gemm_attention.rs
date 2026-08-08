//! Test GEMM-based attention implementation on consumer GPUs
//!
//! This demonstrates Option A: using existing mma.sync GEMM for attention
//! instead of writing new WGMMA/tcgen05 PTX kernels.
//!
//! Usage:
//!   cargo run --package pesti-runner --features cuda --example test_gemm_attention

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::kernel::Kvcache;
use pesti_runner::kernel::attention::{AttentionConfig, AttentionKernel, GemmBasedAttentionKernel};
use pesti_runner::kernel::device_buf::DeviceBuffer;
use pesti_runner::kernel::gemm::{CudaGemmKernelBuilder, GemmArch};
use pesti_runner::kernel::memory::{CudaMemoryBackend, MemoryBackend};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GEMM-Based Attention Test (Option A) ===\n");

    // Use device 1 (RTX 5060 Ti, sm_12.0)
    let ordinal = 1;
    println!("Using device 1: RTX 5060 Ti (sm_12.0)");

    // Initialize CUDA runtime
    let rt = CudaRuntime::new(ordinal)?;
    let stream = rt.new_stream()?;
    let info = rt.device_info();

    println!(
        "Device: {} (compute capability sm_{}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    );

    // Build GEMM kernel (uses mma.sync for consumer Blackwell)
    let gemm_arch = GemmArch::Mma;
    let gemm_kernel = CudaGemmKernelBuilder::new(
        gemm_arch,
        rt.context().clone(),
        stream.clone(),
        info.clone(),
    )
    .build()?;

    println!("✅ GEMM kernel loaded (arch {:?})", gemm_arch);

    // Create backend for device allocation
    let backend = Arc::new(CudaMemoryBackend::with_device_info(
        stream.clone(),
        info.clone(),
    ));

    // Create GEMM-based attention kernel
    let attn_kernel = GemmBasedAttentionKernel::new(gemm_kernel, backend.clone());

    // Attention config: Qwen2.5-0.5B style — 8 heads, 64 dim, 256 seq
    let num_heads = 8;
    let head_dim = 64;
    let seq_len = 256;

    let query_len = 1;
    let q_size = query_len * num_heads * head_dim;

    println!("\n📦 Allocating tensors:");
    println!(
        "   Q: [{} x {} x {}] = {} f16 elements",
        query_len, num_heads, head_dim, q_size
    );

    // Create sample Q data on GPU
    let q_host: Vec<f16> = (0..q_size)
        .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
        .collect();
    let q_buf = DeviceBuffer::from_host_device(&*backend, &q_host)?;
    println!("✅ Q allocated on GPU");

    // Create K and V test data on host first
    let k_test: Vec<f16> = (0..num_heads * head_dim * seq_len)
        .map(|i| f16::from_f32(((i % 13) as f32) * 0.05 - 0.4))
        .collect();

    // Create device-backed Kvcache and write data position by position
    // Since Kvcache::append only works with host-backed buffers, we create
    // a host cache, populate it, then copy to device.
    let mut k_host_cache = Kvcache::new(num_heads, num_heads, head_dim, seq_len, false);
    for pos in 0..seq_len {
        let offset = pos * num_heads * head_dim;
        let row = &k_test[offset..offset + num_heads * head_dim];
        k_host_cache.append(row, row)?;
    }

    // Copy host buffer to device
    let host_slice = k_host_cache
        .buffer()
        .as_slice()
        .expect("host Kvcache must have data");
    let _kv_device_buf = DeviceBuffer::from_host_device(&*backend, host_slice)?;

    // Create device-backed Kvcache from the device pointer
    let kv_ptr = _kv_device_buf.device_ptr();
    let mut k_cache =
        unsafe { Kvcache::from_device(kv_ptr, num_heads, num_heads, head_dim, seq_len) };
    k_cache.set_seq_len(seq_len);

    println!(
        "✅ K/V caches on GPU (ptr={:#x}, seq_len={})",
        kv_ptr, seq_len
    );

    // Configure attention
    let config = AttentionConfig::new(num_heads, head_dim).with_max_seq(seq_len);

    let scale = 1.0 / (head_dim as f32).sqrt();
    println!(
        "\n⚙️  Config: {} heads, {} dim, scale={:.4}",
        config.num_heads, config.head_dim, scale
    );

    // Execute attention: Q @ K^T -> softmax -> S @ V
    println!("\n--- Running GEMM-based attention ---");
    let output = attn_kernel.forward(
        &q_buf, &k_cache, &k_cache, // V = K for this test
        None, &config,
    )?;

    // Retrieve and verify results
    let output_host = output.to_host_vec(&*backend)?;
    println!(
        "✅ Attention completed: {} output elements",
        output_host.len()
    );

    // Compute expected CPU result for comparison
    let mut expected = vec![0.0f32; query_len * num_heads * head_dim];
    for q_idx in 0..query_len {
        for head in 0..num_heads {
            // Compute scores: Q @ K^T
            let mut scores = vec![0.0f32; seq_len];
            for s in 0..seq_len {
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q_host[q_idx * num_heads * head_dim + head * head_dim + d].to_f32();
                    let k_val = k_test[s * num_heads * head_dim + head * head_dim + d].to_f32();
                    sum += q_val * k_val;
                }
                scores[s] = sum * scale;
            }

            // Softmax
            let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
            let sum_exp: f32 = exp_vals.iter().sum();

            // Apply to V (= K for test)
            for d in 0..head_dim {
                let mut out_sum = 0.0f32;
                for s in 0..seq_len {
                    let weight = exp_vals[s] / sum_exp;
                    let v_val = k_test[s * num_heads * head_dim + head * head_dim + d].to_f32();
                    out_sum += weight * v_val;
                }
                expected[q_idx * num_heads * head_dim + head * head_dim + d] = out_sum;
            }
        }
    }

    // Compare
    let mut max_err = 0.0f32;
    for (&gpu, &cpu) in output_host.iter().zip(expected.iter()) {
        let err = (gpu - cpu).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("\n--- Results ---");
    println!("Max error vs CPU reference: {:.3e}", max_err);

    if max_err < 1e-2 {
        println!("✅ CORRECT: GPU attention output matches CPU reference within tolerance");
    } else {
        println!("⚠️  WARNING: Error {:.3e} exceeds tolerance", max_err);
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
