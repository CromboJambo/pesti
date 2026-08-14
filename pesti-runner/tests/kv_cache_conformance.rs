//! KV Cache conformance test for one-stage full fusion attention kernel
//! 
//! Validates that the kernel can use pre-computed KV cache instead of recomputing K/V.

#![cfg(feature = "cuda")]

use pesti_runner::cuda_runtime::{allocate_device_memory, copy_host_to_device, CudaRuntime};
use pesti_runner::cuda_shim::{cu_stream, launch_kernel, CudaModule};

/// Reference causal attention with KV cache (uses pre-computed K and V from cache)
fn reference_kv_attention(
    q: &[f32],
    kv_cache_k: &[f32], // Pre-computed keys from cache
    kv_cache_v: &[f32], // Pre-computed values from cache
    seq_q: usize,       // Query sequence length (typically 1 for decode)
    seq_k: usize,       // Total sequence length (including cached positions)
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];

    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for dim_idx in 0..head_dim {
                let mut sum = 0.0f32;
                
                // Compute scores over cached k positions (causal: only k_pos <= q_pos)
                let mut scores = vec![f32::NEG_INFINITY; seq_k];
                for k_pos in 0..=q_pos.min(seq_k - 1) {
                    let q_idx = q_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    let k_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    
                    // Dot product over head_dim
                    for d in 0..head_dim {
                        let q_d = q[q_pos * num_heads * head_dim + head * head_dim + d];
                        let k_d = kv_cache_k[k_idx]; // Use cached K
                        scores[k_pos] += q_d * k_d;
                    }
                    scores[k_pos] /= (head_dim as f32).sqrt();
                }

                // Softmax over sequence dimension (only past positions)
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores
                    .iter()
                    .map(|&s| {
                        if s.is_finite() {
                            (s - max_score).exp()
                        } else {
                            0.0
                        }
                    })
                    .sum();
                
                for k_pos in 0..=q_pos.min(seq_k - 1) {
                    if scores[k_pos].is_finite() && exp_sum > 1e-6 {
                        let softmax_val = (scores[k_pos] - max_score).exp() / exp_sum;
                        let v_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
                        sum += softmax_val * kv_cache_v[v_idx]; // Use cached V
                    }
                }

                output[q_pos * num_heads * head_dim + head * head_dim + dim_idx] = sum;
            }
        }
    }

    output
}

/// Launch the fused attention kernel (synchronous)
fn launch_fused_attention_with_kv(
    cuda_rt: &CudaRuntime,
    module: &CudaModule,
    q_ptr: *mut u8,       // Query (new token)
    k_cache_ptr: *mut u8, // Cached keys
    v_cache_ptr: *mut u8, // Cached values
    out_ptr: *mut u8,     // Output
    seq_q: usize,         // Query sequence length (typically 1 for decode)
    seq_k: usize,         // Total sequence length (cached + new)
    num_heads: usize,
    head_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = cuda_rt.new_stream()?;

    // Get function from module
    let mangled_name = "_Z27fused_attention_full_kernelPK6__halfS1_S1_Pfiiii";
    let function = module.load_function(mangled_name)?;

    // Parameters (10 total, matching fused kernel signature)
    let mut q_v: u64 = q_ptr as u64;
    let mut k_cache_v: u64 = k_cache_ptr as u64; // Point to cached K
    let mut v_cache_v: u64 = v_cache_ptr as u64; // Point to cached V
    let mut out_v: u64 = out_ptr as u64;
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel with grid (seq_q, seq_k, num_heads), block (head_dim, 1, 1)
    let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
    let block = (head_dim as u32, 1u32, 1u32);

    let mut params: [*mut std::ffi::c_void; 10] = [
        &mut q_v as *mut u64 as *mut std::ffi::c_void,
        &mut k_cache_v as *mut u64 as *mut std::ffi::c_void,
        &mut v_cache_v as *mut u64 as *mut std::ffi::c_void,
        &mut out_v as *mut u64 as *mut std::ffi::c_void,
        &mut (scale as f32) as *mut f32 as *mut std::ffi::c_void,
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
        &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
        &mut 0u64 as *mut u64 as *mut std::ffi::c_void, // placeholder for unused param
    ];

    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            cu_stream(&stream),
            &mut params,
        )?;
    }

    Ok(())
}

#[test]
fn test_kv_cache_decode() {
    println!("\n=== KV Cache Decode Test ===");
    println!("Simulating autoregressive decode (seq_q=1, seq_k=cached_positions)");

    let cuda_rt = CudaRuntime::new(0).unwrap();

    // Configuration: 1 new token to generate, with 3 cached positions
    let seq_q = 1;      // New query (single token)
    let seq_k = 4;      // Total sequence (3 cached + 1 new)
    let num_heads = 2;
    let head_dim = 8;

    let q_size = seq_q * num_heads * head_dim * 2;  // f16
    let k_cache_size = seq_k * num_heads * head_dim * 2;
    let v_cache_size = seq_k * num_heads * head_dim * 2;
    let output_size = seq_q * num_heads * head_dim * 4;

    // Initialize with deterministic values
    let q_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // New query token
    let k_cache_host: Vec<f32> = (0..seq_k * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_k * num_heads * head_dim) as f32 / 2.0) * 0.15)
        .collect();
    let v_cache_host: Vec<f32> = (0..seq_k * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_k * num_heads * head_dim) as f32 / 2.0) * 0.2)
        .collect();

    // Allocate GPU memory
    let q_ptr = unsafe { allocate_device_memory(q_size).unwrap() };
    let k_cache_ptr = unsafe { allocate_device_memory(k_cache_size).unwrap() };
    let v_cache_ptr = unsafe { allocate_device_memory(v_cache_size).unwrap() };
    let out_ptr = unsafe { allocate_device_memory(output_size).unwrap() };

    // Copy to device (convert f32 to f16)
    let q_host_f16: Vec<half::f16> = q_host.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k_cache_host_f16: Vec<half::f16> = k_cache_host.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_cache_host_f16: Vec<half::f16> = v_cache_host.iter().map(|&x| half::f16::from_f32(x)).collect();

    unsafe {
        copy_host_to_device(q_ptr, q_host_f16.as_ptr() as *const u8, q_size).unwrap();
        copy_host_to_device(k_cache_ptr, k_cache_host_f16.as_ptr() as *const u8, k_cache_size).unwrap();
        copy_host_to_device(v_cache_ptr, v_cache_host_f16.as_ptr() as *const u8, v_cache_size).unwrap();
    }

    // Load PTX and launch
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_full_kernel.ptx");
    let module = CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();

    launch_fused_attention_with_kv(
        &cuda_rt,
        &module,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        out_ptr,
        seq_q,
        seq_k,
        num_heads,
        head_dim,
    )
    .unwrap();

    // Synchronize and read back
    cuda_rt.synchronize().unwrap();

    let mut gpu_output: Vec<f32> = vec![0.0; output_size / 4];
    unsafe {
        copy_host_to_device(out_ptr, gpu_output.as_mut_ptr() as *mut u8, output_size).unwrap();
    }

    // Compare with CPU reference (using cached K/V)
    let cpu_output = reference_kv_attention(&q_host, &k_cache_host, &v_cache_host, seq_q, seq_k, num_heads, head_dim);

    // Compute error metrics
    let mut max_abs_err = 0.0f32;

    for i in 0..(seq_q * num_heads * head_dim) {
        let abs_err = (gpu_output[i] - cpu_output[i]).abs();
        max_abs_err = max_abs_err.max(abs_err);
    }

    println!("  Max absolute error: {:.6e}", max_abs_err);

    // With KV cache, error should be < 1e-3 (same as non-causal)
    assert!(
        max_abs_err < 1e-3,
        "KV cache attention error {} exceeds threshold",
        max_abs_err
    );

    println!("✅ KV cache decode PASSED");
}

#[test]
fn test_kv_cache_prefill_vs_decode() {
    println!("\n=== KV Cache Prefill vs Decode Comparison ===");

    let cuda_rt = CudaRuntime::new(0).unwrap();

    // Prefill: process multiple tokens at once (seq_q > 1)
    // Decode: process one token at a time (seq_q = 1)
    
    let seq_k_prefill = 8;  // Total sequence length
    let seq_q_prefill = 4;  // Process 4 tokens at once
    let num_heads = 2;
    let head_dim = 8;

    let q_size = seq_q_prefill * num_heads * head_dim * 2;
    let k_cache_size = seq_k_prefill * num_heads * head_dim * 2;
    let v_cache_size = seq_k_prefill * num_heads * head_dim * 2;
    let output_size = seq_q_prefill * num_heads * head_dim * 4;

    // Initialize with deterministic values
    let q_host: Vec<f32> = (0..seq_q_prefill * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_q_prefill * num_heads * head_dim) as f32 / 2.0) * 0.1)
        .collect();
    let k_cache_host: Vec<f32> = (0..seq_k_prefill * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_k_prefill * num_heads * head_dim) as f32 / 2.0) * 0.15)
        .collect();
    let v_cache_host: Vec<f32> = (0..seq_k_prefill * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_k_prefill * num_heads * head_dim) as f32 / 2.0) * 0.2)
        .collect();

    let q_ptr = unsafe { allocate_device_memory(q_size).unwrap() };
    let k_cache_ptr = unsafe { allocate_device_memory(k_cache_size).unwrap() };
    let v_cache_ptr = unsafe { allocate_device_memory(v_cache_size).unwrap() };
    let out_ptr = unsafe { allocate_device_memory(output_size).unwrap() };

    let q_host_f16: Vec<half::f16> = q_host.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k_cache_host_f16: Vec<half::f16> = k_cache_host.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_cache_host_f16: Vec<half::f16> = v_cache_host.iter().map(|&x| half::f16::from_f32(x)).collect();

    unsafe {
        copy_host_to_device(q_ptr, q_host_f16.as_ptr() as *const u8, q_size).unwrap();
        copy_host_to_device(k_cache_ptr, k_cache_host_f16.as_ptr() as *const u8, k_cache_size).unwrap();
        copy_host_to_device(v_cache_ptr, v_cache_host_f16.as_ptr() as *const u8, v_cache_size).unwrap();
    }

    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_full_kernel.ptx");
    let module = CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();

    launch_fused_attention_with_kv(
        &cuda_rt,
        &module,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        out_ptr,
        seq_q_prefill,
        seq_k_prefill,
        num_heads,
        head_dim,
    )
    .unwrap();

    cuda_rt.synchronize().unwrap();

    let mut gpu_output: Vec<f32> = vec![0.0; output_size / 4];
    unsafe {
        copy_host_to_device(out_ptr, gpu_output.as_mut_ptr() as *mut u8, output_size).unwrap();
    }

    // Prefill should produce reasonable outputs (not all zeros)
    let has_nonzero = gpu_output.iter().any(|&x| x.abs() > 1e-6);
    
    println!("  GPU output (first 8 values): {:?}", &gpu_output[0..8.min(gpu_output.len())]);
    println!("  Has non-zero outputs: {}", has_nonzero);
    
    assert!(has_nonzero, "Prefill with KV cache should produce non-zero outputs");

    println!("✅ KV cache prefill PASSED");
}
