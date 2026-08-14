//! Adversarial bounded input test for single-kernel fused attention
//! Tests with varied inputs to ensure conformance across different scenarios

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{cu_stream, launch_kernel};

/// Reference RoPE implementation matching llama.cpp (HALF-SWAP rotation)
fn apply_rope_cpu(q: &mut [f32], head_dim: usize, pos: usize, rope_base: f32) {
    let half_dim = head_dim / 2;
    
    // HALF-SWAP rotation: dimension i pairs with (i + head_dim/2)
    for dim in 0..half_dim {
        let idx_first = dim;
        let idx_second = dim + half_dim;
        
        let inv_freq = 1.0 / (rope_base.powf((dim as f32) / half_dim as f32));
        let freq = pos as f32 * inv_freq;
        let cos_val = freq.cos();
        let sin_val = freq.sin();
        
        let q_first = q[idx_first];
        let q_second = q[idx_second];
        q[idx_first] = q_first * cos_val - q_second * sin_val;
        q[idx_second] = q_first * sin_val + q_second * cos_val;
    }
}

/// Compute llama.cpp-style attention scores (scaled dot-product) - PER HEAD
fn reference_llama_attention(
    q: &[f32],
    k: &[f32],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
    rope_base: f32,
) -> Vec<f32> {
    let mut q_rope = vec![0.0f32; seq_q * num_heads * head_dim];
    for (i, &val) in q.iter().enumerate() {
        q_rope[i] = val;
    }

    let mut k_rope = vec![0.0f32; seq_k * num_heads * head_dim];
    for (i, &val) in k.iter().enumerate() {
        k_rope[i] = val;
    }

    // Apply RoPE to each position - PER HEAD
    for pos in 0..seq_q {
        for head in 0..num_heads {
            let q_start = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(
                &mut q_rope[q_start..q_start + head_dim],
                head_dim,
                pos,
                rope_base,
            );
        }
    }

    for pos in 0..seq_k {
        for head in 0..num_heads {
            let k_start = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(
                &mut k_rope[k_start..k_start + head_dim],
                head_dim,
                pos,
                rope_base,
            );
        }
    }

    // Compute attention scores (scaled by 1/sqrt(head_dim)) - PER HEAD
    let mut scores = vec![0.0f32; seq_q * num_heads * seq_k];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for k_pos in 0..seq_k {
                let mut score = 0.0f32;
                let q_offset = q_pos * num_heads * head_dim + head * head_dim;
                let k_offset = k_pos * num_heads * head_dim + head * head_dim;

                for d in 0..head_dim {
                    score += q_rope[q_offset + d] * k_rope[k_offset + d];
                }
                score /= (head_dim as f32).sqrt();
                scores[q_pos * num_heads * seq_k + head * seq_k + k_pos] = score;
            }
        }
    }

    scores
}

/// Apply causal mask (llama.cpp style: mask future tokens where k_pos > q_pos) - per head
fn apply_causal_mask(scores: &mut [f32], seq_q: usize, seq_k: usize, num_heads: usize) {
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for k_pos in 0..seq_k {
                if k_pos > q_pos {
                    scores[q_pos * num_heads * seq_k + head * seq_k + k_pos] = -1e9;
                }
            }
        }
    }
}

/// Apply softmax per query row, per head (llama.cpp style)
fn reference_softmax(scores: &[f32], seq_q: usize, seq_k: usize, num_heads: usize) -> Vec<f32> {
    let mut probs = vec![0.0f32; seq_q * num_heads * seq_k];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            let start = q_pos * num_heads * seq_k + head * seq_k;
            let end = start + seq_k;

            let max_val = scores[start..end]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            let exps: Vec<f32> = scores[start..end]
                .iter()
                .map(|&x| (x - max_val).exp())
                .collect();

            let sum: f32 = exps.iter().sum();
            for i in 0..seq_k {
                probs[start + i] = exps[i] / sum;
            }
        }
    }
    probs
}

#[cfg(feature = "cuda")]
#[test]
fn test_adversarial_bounded_attention() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping adversarial test");
        return;
    }

    println!("=== Adversarial Bounded Attention Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();

    // Configuration: multi-position causal cases
    let seq_q = 3;   // q=0, q=1, q=2 (varying causal contexts)
    let seq_k = 8;   // Short sequence for clarity
    let num_heads = 2;
    let head_dim = 16;
    let rope_base = 10_000.0;

    // ADVERSARIAL INPUTS: bounded but varied (not deterministic zeros)
    // Range: [-10.0, 10.0] with specific patterns to test edge cases
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| {
            // Alternating positive/negative values with varying magnitudes
            let v = ((i % 20) as f32 - 10.0) / 2.0;
            f16::from_f32(v)
        })
        .collect();

    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| {
            // Different pattern for K to ensure non-trivial dot products
            let v = ((i % 15) as f32 - 7.5) / 1.5;
            f16::from_f32(v)
        })
        .collect();

    let v_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| {
            // V values with mixed signs and magnitudes
            let v = ((i % 25) as f32 - 12.5) / 2.5;
            f16::from_f32(v)
        })
        .collect();

    println!(
        "Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );
    println!("Input range: [-10.0, 10.0] (varied patterns)");
    println!();

    // Compute CPU reference (llama.cpp style with RoPE + causal mask)
    let mut llama_scores = reference_llama_attention(
        &q_h.iter().map(|&x| x.to_f32()).collect::<Vec<f32>>(),
        &k_h.iter().map(|&x| x.to_f32()).collect::<Vec<f32>>(),
        seq_q,
        seq_k,
        num_heads,
        head_dim,
        rope_base,
    );

    // Apply causal mask BEFORE softmax
    apply_causal_mask(&mut llama_scores, seq_q, seq_k, num_heads);

    let llama_probs = reference_softmax(&llama_scores, seq_q, seq_k, num_heads);

    println!("CPU reference computed");
    println!(
        "CPU softmax sum (q=0, h=0): {:.6}",
        llama_probs[0..seq_k].iter().sum::<f32>()
    );
    println!();

    // Allocate device memory
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let out_size = seq_q * num_heads * head_dim * 2;

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };

    // Copy Q, K, V to device
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size)
            .unwrap();
    }

    // Load PTX and get function
    let stream = cuda_rt.new_stream().unwrap();
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src)
        .unwrap();

    let mangled_name = "_Z29fused_attention_simple_kernelPK6__halfS1_S1_PS_fiiii";
    let function = module.load_function(mangled_name).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel
    unsafe {
        let mut scale_v: f32 = scale;
        let mut q_v: u64 = q_ptr as u64;
        let mut k_v: u64 = k_ptr as u64;
        let mut v_v: u64 = v_ptr as u64;
        let mut out_v: u64 = out_ptr as u64;
        let mut seq_q_v: u32 = seq_q as u32;
        let mut seq_k_v: u32 = seq_k as u32;
        let mut num_heads_v: u32 = num_heads as u32;
        let mut head_dim_v: u32 = head_dim as u32;

        let mut params: [*mut std::ffi::c_void; 9] = [
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut out_v as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
        ];

        // Grid: (seq_q, num_heads, 1), Block: (64, 1, 1)
        let grid = (seq_q as u32, num_heads as u32, 1u32);
        let block = (64u32, 1u32, 1u32);

        launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            cu_stream(&stream),
            &mut params,
        )
        .unwrap();
    }

    cuda_rt.synchronize().unwrap();

    println!("✅ GPU kernel launched successfully");

    // Copy results back to host
    let mut gpu_out = vec![0.0f32; seq_q * num_heads * head_dim];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_out.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        )
        .unwrap();
    }

    println!("✅ Results copied back to host");

    // Compute CPU reference output (for comparison)
    // For adversarial test: compute full attention output with softmax + V-multiply
    let mut cpu_out = vec![0.0f32; seq_q * num_heads * head_dim];

    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            // Get attention probs for this (q_pos, head)
            let start_idx = q_pos * num_heads * seq_k + head * seq_k;
            
            // Weighted sum of V using softmax probs
            for d in 0..head_dim {
                let mut out_val = 0.0f32;
                for k_pos in 0..seq_k {
                    if llama_probs[start_idx + k_pos] > 1e-9 {
                        let v_val = v_h[k_pos * num_heads * head_dim + head * head_dim + d]
                            .to_f32();
                        out_val += llama_probs[start_idx + k_pos] * v_val;
                    }
                }
                cpu_out[q_pos * num_heads * head_dim + head * head_dim + d] = out_val;
            }
        }
    }

    // Compare outputs
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for i in 0..gpu_out.len() {
        let abs_err = (cpu_out[i] - gpu_out[i]).abs();
        let rel_err = if cpu_out[i].abs() > 1e-8 {
            abs_err / cpu_out[i].abs()
        } else {
            abs_err
        };

        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);

        // Debug: print first few values
        if i < 10 {
            println!(
                "Debug: idx={} | cpu={:.6} | gpu={:.6} | abs_err={:.6e} | rel_err={:.6e}",
                i, cpu_out[i], gpu_out[i], abs_err, rel_err
            );
        }
    }

    println!();
    println!("Results:");
    println!("  Max absolute error: {:.6e}", max_abs_err);
    println!("  Max relative error: {:.6e}", max_rel_err);
    println!();

    // ASSERTION: Conformance requires <1e-4 relative error for adversarial inputs
    assert!(
        max_rel_err < 1e-4,
        "Adversarial conformance FAILED: max_rel_error={:.6e} >= 1e-4 target",
        max_rel_err
    );

    if max_rel_err < 1e-4 {
        println!("✅ Adversarial conformance PASSED (rel error < 1e-4)");
    } else if max_rel_err < 1e-2 {
        println!("⚠️ Moderate discrepancy detected (rel error < 1e-2)");
    } else {
        println!("❌ Large numerical discrepancy detected (rel error >= 1e-2)");
    }

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
}
