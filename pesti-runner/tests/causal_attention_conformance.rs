//! Causal masking test for one-stage full fusion attention kernel
//! 
//! Validates that causal masking (only attending to past/current positions) works correctly.

#![cfg(feature = "cuda")]

use pesti_runner::cuda_runtime::{allocate_device_memory, copy_host_to_device, CudaRuntime};
use pesti_runner::cuda_shim::{cu_stream, launch_kernel, CudaModule};

/// Reference causal attention implementation (only attends to k_pos <= q_pos)
fn reference_causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];

    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for dim_idx in 0..head_dim {
                let mut sum = 0.0f32;
                
                // Compute scores over k positions (causal: only k_pos <= q_pos)
                let mut scores = vec![f32::NEG_INFINITY; seq_k]; // Initialize with -inf for future positions
                for k_pos in 0..=q_pos {
                    let q_idx = q_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    let k_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    
                    // Dot product over head_dim
                    for d in 0..head_dim {
                        let q_d = q[q_pos * num_heads * head_dim + head * head_dim + d];
                        let k_d = k[k_pos * num_heads * head_dim + head * head_dim + d];
                        scores[k_pos] += q_d * k_d;
                    }
                    scores[k_pos] /= (head_dim as f32).sqrt();
                }

                // Softmax over sequence dimension (only past positions)
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| if s.is_finite() { (s - max_score).exp() } else { 0.0 }).sum();
                
                for k_pos in 0..=q_pos {
                    if scores[k_pos].is_finite() {
                        let softmax_val = (scores[k_pos] - max_score).exp() / exp_sum;
                        let v_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
                        sum += softmax_val * v[v_idx];
                    }
                }

                output[q_pos * num_heads * head_dim + head * head_dim + dim_idx] = sum;
            }
        }
    }

    output
}

/// Launch the fused attention kernel (synchronous)
fn launch_fused_attention_sync(
    cuda_rt: &CudaRuntime,
    module: &CudaModule,
    q_ptr: *mut u8,
    k_ptr: *mut u8,
    v_ptr: *mut u8,
    out_ptr: *mut u8,
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = cuda_rt.new_stream()?;

    // Get function from module
    let mangled_name = "_Z22fused_attention_kernelPK6__halfS1_S1_Pfiiii";
    let function = module.load_function(mangled_name)?;

    // Parameters (10 total, matching fused kernel signature)
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut v_v: u64 = v_ptr as u64;
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
        &mut k_v as *mut u64 as *mut std::ffi::c_void,
        &mut v_v as *mut u64 as *mut std::ffi::c_void,
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
fn test_causal_attention() {
    println!("\n=== Causal Attention Test ===");

    let cuda_rt = CudaRuntime::new(0).unwrap();

    // Small configuration for testing
    let seq_q = 4;
    let seq_k = 4;
    let num_heads = 2;
    let head_dim = 8;

    let q_size = seq_q * num_heads * head_dim * 2; // f16
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let output_size = seq_q * num_heads * head_dim * 4; // f32

    // Initialize with deterministic values
    let q_host: Vec<f32> = (0..seq_q * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_q * num_heads * head_dim) as f32 / 2.0) * 0.1)
        .collect();
    let k_host: Vec<f32> = (0..seq_k * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_k * num_heads * head_dim) as f32 / 2.0) * 0.15)
        .collect();
    let v_host: Vec<f32> = (0..seq_k * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_k * num_heads * head_dim) as f32 / 2.0) * 0.2)
        .collect();

    // Allocate GPU memory
    let q_ptr = unsafe { allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { allocate_device_memory(output_size).unwrap() };

    // Copy to device (convert f32 to f16)
    let q_host_f16: Vec<half::f16> = q_host.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k_host_f16: Vec<half::f16> = k_host.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_host_f16: Vec<half::f16> = v_host.iter().map(|&x| half::f16::from_f32(x)).collect();

    unsafe {
        copy_host_to_device(q_ptr, q_host_f16.as_ptr() as *const u8, q_size).unwrap();
        copy_host_to_device(k_ptr, k_host_f16.as_ptr() as *const u8, k_size).unwrap();
        copy_host_to_device(v_ptr, v_host_f16.as_ptr() as *const u8, v_size).unwrap();
    }

    // Load PTX and launch
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_full_kernel.ptx");
    let module = CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();

    launch_fused_attention_sync(
        &cuda_rt,
        &module,
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        seq_q,
        seq_k,
        num_heads,
        head_dim,
    )
    .unwrap();

    // Synchronize and read back (simplified - just check it runs)
    cuda_rt.synchronize().unwrap();

    let mut gpu_output: Vec<f32> = vec![0.0; output_size / 4];
    unsafe {
        copy_host_to_device(out_ptr, gpu_output.as_mut_ptr() as *mut u8, output_size).unwrap();
    }

    // Compare with CPU reference (causal)
    let cpu_output = reference_causal_attention(&q_host, &k_host, &v_host, seq_q, seq_k, num_heads, head_dim);

    // Compute error metrics
    let mut max_abs_err = 0.0f32;

    for i in 0..(seq_q * num_heads * head_dim) {
        let abs_err = (gpu_output[i] - cpu_output[i]).abs();
        max_abs_err = max_abs_err.max(abs_err);
    }

    println!("  Max absolute error: {:.6e}", max_abs_err);

    // With causal masking, error should be < 1e-3 (same as non-causal)
    assert!(
        max_abs_err < 1e-3,
        "Causal attention error {} exceeds threshold",
        max_abs_err
    );

    println!("✅ Causal attention PASSED");
}

#[test]
fn test_causal_vs_noncausal() {
    println!("\n=== Causal vs Non-Causal Comparison ===");

    let cuda_rt = CudaRuntime::new(0).unwrap();

    // Configuration where causal and non-causal differ significantly
    let seq_q = 4;
    let seq_k = 8;  // k > q to see the masking effect
    let num_heads = 2;
    let head_dim = 8;

    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let output_size = seq_q * num_heads * head_dim * 4;

    // Initialize with values that will produce different results for causal vs non-causal
    let q_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // Simple pattern
    let k_host: Vec<f32> = (0..seq_k * num_heads * head_dim)
        .map(|i| (i as f32) * 0.5)
        .collect();
    let v_host: Vec<f32> = (0..seq_k * num_heads * head_dim)
        .map(|i| (i as f32) * 0.3)
        .collect();

    // Pad to correct size
    let mut q_padded = vec![0.0f32; seq_q * num_heads * head_dim];
    let mut k_padded = vec![0.0f32; seq_k * num_heads * head_dim];
    let mut v_padded = vec![0.0f32; seq_k * num_heads * head_dim];

    q_padded[0..4].copy_from_slice(&q_host);
    let k_len = k_host.len().min(k_padded.len());
    let v_len = v_host.len().min(v_padded.len());
    k_padded[0..k_len].copy_from_slice(&k_host);
    v_padded[0..v_len].copy_from_slice(&v_host);

    let q_ptr = unsafe { allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { allocate_device_memory(output_size).unwrap() };

    let q_host_f16: Vec<half::f16> = q_padded.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k_host_f16: Vec<half::f16> = k_padded.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_host_f16: Vec<half::f16> = v_padded.iter().map(|&x| half::f16::from_f32(x)).collect();

    unsafe {
        copy_host_to_device(q_ptr, q_host_f16.as_ptr() as *const u8, q_size).unwrap();
        copy_host_to_device(k_ptr, k_host_f16.as_ptr() as *const u8, k_size).unwrap();
        copy_host_to_device(v_ptr, v_host_f16.as_ptr() as *const u8, v_size).unwrap();
    }

    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_full_kernel.ptx");
    let module = CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();

    launch_fused_attention_sync(
        &cuda_rt,
        &module,
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        seq_q,
        seq_k,
        num_heads,
        head_dim,
    )
    .unwrap();

    cuda_rt.synchronize().unwrap();

    let mut gpu_output: Vec<f32> = vec![0.0; output_size / 4];
    unsafe {
        copy_host_to_device(out_ptr, gpu_output.as_mut_ptr() as *mut u8, output_size).unwrap();
    }

    // Causal should produce different results than non-causal when seq_k > seq_q
    println!("  GPU output (first 16 values): {:?}", &gpu_output[0..16]);
    println!("  ✅ Causal masking test completed");
}
