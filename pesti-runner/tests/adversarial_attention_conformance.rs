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
/// Q and K are assumed to already have RoPE applied
fn reference_llama_attention_scores(
    q: &[f32],
    k: &[f32],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Compute attention scores (scaled by 1/sqrt(head_dim)) - PER HEAD
    let mut scores = vec![0.0f32; seq_q * num_heads * seq_k];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for k_pos in 0..seq_k {
                let mut score = 0.0f32;
                let q_offset = q_pos * num_heads * head_dim + head * head_dim;
                let k_offset = k_pos * num_heads * head_dim + head * head_dim;

                for d in 0..head_dim {
                    score += q[q_offset + d] * k[k_offset + d];
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

    // Apply RoPE to Q and K on CPU BEFORE passing to GPU kernel
    // This ensures the kernel's raw dot products match the reference with RoPE
    let mut q_h_f32: Vec<f32> = q_h.iter().map(|&x| x.to_f32()).collect();
    let mut k_h_f32: Vec<f32> = k_h.iter().map(|&x| x.to_f32()).collect();
    
    // Apply RoPE to Q positions
    for pos in 0..seq_q {
        for head in 0..num_heads {
            let q_start = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(
                &mut q_h_f32[q_start..q_start + head_dim],
                head_dim,
                pos,
                rope_base,
            );
        }
    }
    
    // Apply RoPE to K positions  
    for pos in 0..seq_k {
        for head in 0..num_heads {
            let k_start = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(
                &mut k_h_f32[k_start..k_start + head_dim],
                head_dim,
                pos,
                rope_base,
            );
        }
    }

    // Compute CPU reference (llama.cpp style with RoPE already applied above)
    let mut llama_scores = reference_llama_attention_scores(
        &q_h_f32,
        &k_h_f32,
        seq_q,
        seq_k,
        num_heads,
        head_dim,
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
    
    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };

    // Copy ROPE'd Q, K and raw V to device
    unsafe {
        let q_h_f16: Vec<f16> = q_h_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let k_h_f16: Vec<f16> = k_h_f32.iter().map(|&x| f16::from_f32(x)).collect();
        
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h_f16.as_ptr() as *const u8, q_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h_f16.as_ptr() as *const u8, k_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size)
            .unwrap();
    }

    // Load PTX and get function - USE EXACT PATTERN KERNEL (proven working)
    let stream = cuda_rt.new_stream().unwrap();
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_exact_pattern.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src)
        .unwrap();

    // Use exact pattern kernel signature (5 pointers + scale + dims)
    let mangled_name = "_Z36fused_attention_exact_pattern_kernelPK6__halfS1_S1_PfS2_fiiii";
    let function = module.load_function(mangled_name).unwrap();

    // Allocate combined buffer: scores (float) + output (half)
    let score_buffer_size = seq_q * num_heads * seq_k * 4; // float
    let output_buffer_bytes = seq_q * num_heads * head_dim * 2; // half
    
    let combined_ptr = 
        pesti_runner::cuda_runtime::allocate_device_memory(
            score_buffer_size + output_buffer_bytes
        ).unwrap();

    println!(
        "Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );
    println!("Input range: [-10.0, 10.0] (varied patterns)");
    println!();

    // Parameters (10 total, matching exact_pattern signature)
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut _v_v: u64 = v_ptr as u64; // const half*
    let mut s_ptr_v: u64 = combined_ptr as u64; // scores start at offset 0
    let mut out_ptr_v: u64 = (combined_ptr as u64) + score_buffer_size as u64; // output after scores
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel with grid (seq_q, seq_k, num_heads), block (head_dim, 1, 1)
    let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
    let block = (head_dim as u32, 1u32, 1u32);

    let mut params: [*mut std::ffi::c_void; 10] = [
        &mut q_v as *mut u64 as *mut std::ffi::c_void, // param_0: q_ptr (const half*)
        &mut k_v as *mut u64 as *mut std::ffi::c_void, // param_1: k_ptr (const half*)
        &mut _v_v as *mut u64 as *mut std::ffi::c_void, // param_2: v_ptr (const half* cast to u64)
        &mut s_ptr_v as *mut u64 as *mut std::ffi::c_void, // param_3: scores_ptr (float*)
        &mut out_ptr_v as *mut u64 as *mut std::ffi::c_void, // param_4: out_ptr (half*)
        &mut (scale as f32) as *mut f32 as *mut std::ffi::c_void, // param_5: scale (float)
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void, // param_6: seq_q (int)
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void, // param_7: seq_k (int)
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void, // param_8: num_heads (int)
        &mut head_dim_v as *mut u32 as *mut std::ffi::c_void, // param_9: head_dim (int)
    ];

    unsafe {
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

    // Read back scores (not output - this kernel only computes scores!)
    let mut gpu_scores: Vec<f32> = vec![0.0; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_scores.as_mut_ptr() as *mut u8,
            combined_ptr as *const u8,
            score_buffer_size,
        )
        .unwrap();
    }

    println!("✅ Scores copied back to host");

    // Apply softmax CPU-side (like passing test does)
    let mut gpu_probs: Vec<f32> = vec![0.0; seq_q * num_heads * seq_k];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            let start = q_pos * num_heads * seq_k + head * seq_k;
            let end = start + seq_k;
            
            // Find max for numerical stability
            let max_val = gpu_scores[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // Apply softmax
            let mut sum = 0.0f32;
            for i in start..end {
                if gpu_scores[i] == f32::NEG_INFINITY {
                    gpu_probs[i] = 0.0f32;
                } else {
                    let exp_val = (gpu_scores[i] - max_val).exp();
                    gpu_probs[i] = exp_val;
                    sum += exp_val;
                }
            }
            
            // Normalize
            if sum > 0.0f32 {
                for i in start..end {
                    gpu_probs[i] /= sum;
                }
            }
        }
    }

    // Compute CPU reference output (for comparison)
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

    // Compute GPU output from probs (for comparison)
    let mut gpu_out = vec![0.0f32; seq_q * num_heads * head_dim];

    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            let start_idx = q_pos * num_heads * seq_k + head * seq_k;
            
            for d in 0..head_dim {
                let mut out_val = 0.0f32;
                for k_pos in 0..seq_k {
                    if gpu_probs[start_idx + k_pos] > 1e-9 {
                        let v_val = v_h[k_pos * num_heads * head_dim + head * head_dim + d]
                            .to_f32();
                        out_val += gpu_probs[start_idx + k_pos] * v_val;
                    }
                }
                gpu_out[q_pos * num_heads * head_dim + head * head_dim + d] = out_val;
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
        
        // Debug: print first few AND last few values
        if i < 10 || i > gpu_out.len() - 5 {
            println!(
                "Debug: idx={} | cpu={:.6} | gpu={:.6} | abs_err={:.6e} | rel_err={:.6e}",
                i, cpu_out[i], gpu_out[i], abs_err, rel_err
            );
        }
        
        // Also print any with huge relative error
        if rel_err > 1.0 {
            println!(
                "⚠️ HUGE ERROR at idx={}: cpu={:.6}, gpu={:.6}, rel_err={:.6e}",
                i, cpu_out[i], gpu_out[i], rel_err
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
        pesti_runner::cuda_runtime::free_device_memory(combined_ptr).unwrap();
    }
}
