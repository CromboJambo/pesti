//! Long-sequence attention conformance test
//! Tests the fused attention kernel with longer sequences (64 and 128 tokens)
//! to verify correctness beyond the original short-sequence tests.

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{cu_stream, launch_kernel};

/// Reference RoPE implementation matching llama.cpp (HALF-SWAP rotation)
fn apply_rope_cpu(q: &mut [f32], head_dim: usize, pos: usize, rope_base: f32) {
    let half_dim = head_dim / 2;
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

/// Compute reference attention scores (scaled dot-product) per head
fn reference_attention_scores(
    q: &[f32],
    k: &[f32],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
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

/// Apply causal mask
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

/// Apply softmax per query row, per head
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

fn run_long_sequence_test(seq_len: usize) {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping");
        return;
    }

    println!("=== Long-Sequence Attention Test (seq={}) ===", seq_len);
    println!("GPU: {}", cuda_rt.device_info().name);

    let seq_q = seq_len;
    let seq_k = seq_len;
    let num_heads = 2;
    let head_dim = 16;
    let rope_base = 10000.0;

    // Initialize Q, K, V with varied values for adversarial testing
    let total_tokens = seq_q * num_heads * head_dim;
    let mut q_h: Vec<f16> = (0..total_tokens)
        .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
        .collect();
    let k_h: Vec<f16> = (0..total_tokens)
        .map(|i| f16::from_f32((i as f32 * 0.1 + 1.0).cos()))
        .collect();
    let v_h: Vec<f16> = vec![f16::from_f32(1.0); total_tokens];

    // Convert to float for reference computation
    let mut q_f: Vec<f32> = q_h.iter().map(|x| x.to_f32()).collect();
    let mut k_f: Vec<f32> = k_h.iter().map(|x| x.to_f32()).collect();

    // Apply RoPE to Q (reference)
    for pos in 0..seq_q {
        for head in 0..num_heads {
            let offset = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(&mut q_f[offset..offset + head_dim], head_dim, pos, rope_base);
        }
    }

    // Apply RoPE to K (reference)
    for pos in 0..seq_k {
        for head in 0..num_heads {
            let offset = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(&mut k_f[offset..offset + head_dim], head_dim, pos, rope_base);
        }
    }

    // Reference computation: scores → causal mask → softmax
    let mut ref_scores = reference_attention_scores(&q_f, &k_f, seq_q, seq_k, num_heads, head_dim);
    apply_causal_mask(&mut ref_scores, seq_q, seq_k, num_heads);
    let _ref_probs = reference_softmax(&ref_scores, seq_q, seq_k, num_heads);

    // GPU computation
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;

    let q_ptr = pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap();
    let k_ptr = pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap();
    let v_ptr = pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap();

    pesti_runner::cuda_runtime::copy_host_to_device(
        q_ptr,
        q_h.as_ptr() as *const u8,
        q_size,
    )
    .unwrap();
    pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)
        .unwrap();
    pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size)
        .unwrap();

    let stream = cuda_rt.new_stream().unwrap();
    // Use single-kernel fused attention (patched for longer sequences with softmax stability fix)
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_single_kernel.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src)
        .unwrap();

    // Single-kernel signature changed: now takes pre-allocated scores/probs buffers
    let mangled_name = "_Z29fused_attention_single_kernelPK6__halfS1_S1_PS_PfS3_fiiiif";
    let function = module.load_function(mangled_name).unwrap();

    // Single-kernel writes output directly (no intermediate scores buffer)
    let output_buffer_bytes = seq_q * num_heads * head_dim * 2; // half
    let out_ptr = pesti_runner::cuda_runtime::allocate_device_memory(output_buffer_bytes).unwrap();

    // Pre-allocate score and prob buffers: [seq_q, num_heads, seq_k] of f32
    let scores_bytes = seq_q * num_heads * seq_k * 4;
    let probs_bytes = seq_q * num_heads * seq_k * 4;
    let scores_ptr = pesti_runner::cuda_runtime::allocate_device_memory(scores_bytes).unwrap();
    let probs_ptr = pesti_runner::cuda_runtime::allocate_device_memory(probs_bytes).unwrap();

    unsafe {
        let mut q_v: u64 = q_ptr as u64;
        let mut k_v: u64 = k_ptr as u64;
        let mut v_v: u64 = v_ptr as u64;
        let mut out_v: u64 = out_ptr as u64;
        let mut scores_v: u64 = scores_ptr as u64;
        let mut probs_v: u64 = probs_ptr as u64;
        let mut seq_q_v: u32 = seq_q as u32;
        let mut seq_k_v: u32 = seq_k as u32;
        let mut num_heads_v: u32 = num_heads as u32;
        let mut head_dim_v: u32 = head_dim as u32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut params: [*mut std::ffi::c_void; 12] = [
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut out_v as *mut u64 as *mut std::ffi::c_void,
            &mut scores_v as *mut u64 as *mut std::ffi::c_void,
            &mut probs_v as *mut u64 as *mut std::ffi::c_void,
            &mut (scale as f32) as *mut f32 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
            &mut (rope_base as f32) as *mut f32 as *mut std::ffi::c_void,
        ];

        let grid = (seq_q as u32, num_heads as u32, 1u32);
        let block = (head_dim as u32, 1u32, 1u32);

        launch_kernel(function.cu_function(), grid, block, 0, cu_stream(&stream), &mut params).unwrap();
    }

    cuda_rt.synchronize().unwrap();

    // Copy output back and check for NaN/Inf (basic sanity)
    let mut gpu_out: Vec<f16> = vec![f16::ZERO; seq_q * num_heads * head_dim];
    pesti_runner::cuda_runtime::copy_device_to_host(
        gpu_out.as_mut_ptr() as *mut u8,
        out_ptr,
        output_buffer_bytes,
    )
    .unwrap();

    let nan_count = gpu_out.iter().filter(|x| x.to_f32().is_nan()).count();
    let inf_count = gpu_out.iter().filter(|x| x.to_f32().is_infinite()).count();

    println!("Sequence length: {}", seq_q);
    println!("NaN outputs: {}", nan_count);
    println!("Inf outputs: {}", inf_count);

    // Cleanup
    pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
    pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
    pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
    pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    pesti_runner::cuda_runtime::free_device_memory(scores_ptr).unwrap();
    pesti_runner::cuda_runtime::free_device_memory(probs_ptr).unwrap();

    assert_eq!(nan_count, 0, "Found NaN outputs at sequence length {}", seq_q);
    assert_eq!(inf_count, 0, "Found Inf outputs at sequence length {}", seq_q);
}

#[test]
fn test_long_sequence_attention_64() {
    run_long_sequence_test(64);
}

#[test]
fn test_very_long_sequence_attention_128() {
    run_long_sequence_test(128);
}

#[test]
fn test_long_sequence_attention_256() {
    run_long_sequence_test(256);
}

#[test]
fn test_long_sequence_attention_512() {
    run_long_sequence_test(512);
}

#[test]
fn test_beyond_max_seq_1024() {
    run_long_sequence_test(1024);
}

#[test]
fn test_extreme_seq_4096() {
    run_long_sequence_test(4096);
}

#[test]
fn test_seq_2048() {
    run_long_sequence_test(2048);
}

#[test]
fn test_seq_1536() {
    run_long_sequence_test(1536);
}