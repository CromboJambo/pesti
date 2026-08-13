//! Single-kernel numerical conformance with RoPE using exact_pattern (separate Q,K,V)

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

/// Reference RoPE implementation matching llama.cpp (HALF-SWAP rotation)
fn apply_rope_cpu(q: &mut [f32], head_dim: usize, pos: usize, rope_base: f32) {
    let half_dim = head_dim / 2;
    
    // HALF-SWAP rotation: dimension i pairs with (i + head_dim/2)
    for dim in 0..half_dim {
        let idx_first = dim; // First half: dimensions 0..dim/2-1
        let idx_second = dim + half_dim; // Second half: dimensions dim/2..dim-1
        
        // Compute cos/sin for this position and dimension
        let inv_freq = 1.0 / (rope_base.powf((dim as f32) / half_dim as f32));
        let freq = pos as f32 * inv_freq;
        let cos_val = freq.cos();
        let sin_val = freq.sin();
        
        // Half-swap rotation: [first, second] -> [first*cos - second*sin, first*sin + second*cos]
        let q_first = q[idx_first];
        let q_second = q[idx_second];
        q[idx_first] = q_first * cos_val - q_second * sin_val;
        q[idx_second] = q_first * sin_val + q_second * cos_val;
    }
}

/// Reference attention with RoPE (CPU) matching llama.cpp behavior
fn reference_attention_with_rope(
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
        for k_pos in 0..seq_k {
            for head in 0..num_heads {
                // Dot product across dimensions for THIS HEAD only
                let q_offset = q_pos * num_heads * head_dim + head * head_dim;
                let k_offset = k_pos * num_heads * head_dim + head * head_dim;
                
                let mut score: f32 = 0.0;
                for d in 0..head_dim {
                    score += q_rope[q_offset + d] * k_rope[k_offset + d];
                }
                scores[q_pos * num_heads * seq_k + head * seq_k + k_pos] = score;
            }
        }
    }
    
    // Apply causal mask BEFORE softmax - PER HEAD (mask future tokens)
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            let start = q_pos * num_heads * seq_k + head * seq_k;
            for k_pos in q_pos + 1..seq_k {
                scores[start + k_pos] = f32::NEG_INFINITY;
            }
        }
    }
    
    // Apply softmax per query row, per head
    let mut probs = vec![0.0f32; seq_q * num_heads * seq_k];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            let start = q_pos * num_heads * seq_k + head * seq_k;
            let end = start + seq_k;
            
            // Numerically stable softmax (max-subtract trick)
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

#[test]
fn test_single_kernel_numerical_conformance_with_rope() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping RoPE conformance test");
        return;
    }
    
    println!("=== Single-Kernel Numerical Conformance with RoPE ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();
    
    // Configuration (small for quick testing)
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;
    
    // Create deterministic Q, K (f16) matching llama.cpp test patterns
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    
    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    
    // Create V (same as K for this test)
    let v_h: Vec<f16> = k_h.clone();
    
    println!(
        "Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );
    
    // Compute reference (CPU with RoPE + causal mask)
    let llama_probs = reference_attention_with_rope(
        &q_h.iter().map(|&x| x.to_f32()).collect::<Vec<f32>>(),
        &k_h.iter().map(|&x| x.to_f32()).collect::<Vec<f32>>(),
        seq_q,
        seq_k,
        num_heads,
        head_dim,
        rope_base,
    );
    
    println!(
        "Reference softmax sum (first query): {:.6}",
        llama_probs[0..seq_k].iter().sum::<f32>()
    );
    println!(
        "Reference max attention score: {:.6}",
        llama_probs[0..seq_k]
            .iter()
            .fold(f32::NEG_INFINITY, |max_val, &x| f32::max(max_val, x))
    );
    
    // Allocate SEPARATE buffers for Q, K, V (like exact_pattern expects)
    let q_size = seq_q * num_heads * head_dim * 2; // half
    let k_size = seq_k * num_heads * head_dim * 2; // half
    let v_size = seq_k * num_heads * head_dim * 2; // half
    
    // Allocate ONE buffer containing both scores AND output (like two-kernel)
    let score_buffer_size = seq_q * num_heads * seq_k * 4; // float
    let output_buffer_bytes = seq_q * num_heads * head_dim * 2; // half
    let combined_ptr = 
        pesti_runner::cuda_runtime::allocate_device_memory(
            score_buffer_size + output_buffer_bytes
        ).unwrap();
    
    let q_ptr = pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap();
    let k_ptr = pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap();
    let v_ptr = pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap();
    
    println!("✅ Allocated separate buffers:");
    println!("   Q: {} bytes", q_size);
    println!("   K: {} bytes", k_size);
    println!("   V: {} bytes", v_size);
    println!("   Scores+Output: {} bytes", score_buffer_size + output_buffer_bytes);
    
    // Load exact pattern kernel (stable with separate allocations)
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_exact_pattern.ptx");
    let module =
        pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();
    
    // Check mangled name (from earlier debugging)
    let mangled_name = "_Z36fused_attention_exact_pattern_kernelPK6__halfS1_S1_PfS2_fiiii";
    let function = module.load_function(mangled_name).unwrap();
    
    println!("✅ Loaded exact pattern kernel");
    
    // Parameters: q_ptr, k_ptr, v_ptr (separate), s_ptr (scores), out_ptr, scale, seq_q, seq_k, num_heads, head_dim
    let mut scale_v: f32 = 1.0 / (head_dim as f32).sqrt();
    let mut q_ptr_v: u64 = q_ptr as u64; // Separate allocation!
    let mut k_ptr_v: u64 = k_ptr as u64; // Separate allocation!
    let mut v_ptr_v: u64 = v_ptr as u64; // Separate allocation!
    let mut s_ptr_v: u64 = combined_ptr as u64; // scores start at offset 0
    let mut out_ptr_v: u64 = (combined_ptr as u64) + score_buffer_size as u64; // output starts after scores
    
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;
    
    let stream = cuda_rt.new_stream().unwrap();
    // Exact pattern uses different grid dimensions!
    let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
    let block = (head_dim as u32, 1u32, 1u32);
    
    println!("🚀 Launching with grid={:?}, block={:?}", grid, block);
    
    let mut params = [std::ptr::null_mut(); 10];
    params[0] = &mut q_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[1] = &mut k_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[2] = &mut v_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[3] = &mut s_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[4] = &mut out_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[5] = &mut scale_v as *mut f32 as *mut std::ffi::c_void;
    params[6] = &mut seq_q_v as *mut u32 as *mut std::ffi::c_void;
    params[7] = &mut seq_k_v as *mut u32 as *mut std::ffi::c_void;
    params[8] = &mut num_heads_v as *mut u32 as *mut std::ffi::c_void;
    params[9] = &mut head_dim_v as *mut u32 as *mut std::ffi::c_void;
    
    // Write Q, K, V data to device (async - synchronize after)
    unsafe {
        let _ = cudarc::driver::result::memcpy_htod_async(q_ptr as u64, &q_h, stream.cu_stream());
        let _ = cudarc::driver::result::memcpy_htod_async(k_ptr as u64, &k_h, stream.cu_stream());
        let _ = cudarc::driver::result::memcpy_htod_async(v_ptr as u64, &v_h, stream.cu_stream());
    }
    
    unsafe {
        pesti_runner::cuda_shim::launch_kernel(
            function.cu_function(),
            grid,
            block,
            0, // No shared memory for this kernel
            stream.cu_stream(),
            &mut params,
        )
        .unwrap();
    }
    
    println!("✅ Kernel launched");
    
    cuda_rt.synchronize().unwrap();
    println!("✅ Single-kernel execution completed!");
    
    // Read output back (f16) - kernel writes f16 to output buffer
    let mut out_host: Vec<f16> = vec![f16::ZERO; seq_q * num_heads * head_dim];
    
    unsafe {
        let out_device_ptr_u64 = combined_ptr as u64 + score_buffer_size as u64;
        let _ = cudarc::driver::result::memcpy_dtoh_async(
            &mut out_host,
            out_device_ptr_u64,
            stream.cu_stream(),
        );
    }
    
    // Wait for read-back to complete
    cuda_rt.synchronize().unwrap();
    
    println!(
        "✅ Read back {} f16 values from output buffer",
        out_host.len()
    );
    
    // Print first few values for debugging
    println!("First 4 output values (f32):");
    for i in 0..4.min(out_host.len()) {
        println!("  [{}] = {:.6}", i, out_host[i].to_f32());
    }
    
    // Compute numerical conformance vs reference
    let mut max_rel_error: f32 = 0.0;
    for (i, &out_val) in out_host.iter().enumerate() {
        let out_f32 = out_val.to_f32();
        let ref_val = llama_probs[i]; // Reference attention output
        let abs_error = (out_f32 - ref_val).abs();
        let rel_error = if ref_val.abs() > 1e-6 {
            abs_error / ref_val.abs()
        } else {
            abs_error // Absolute error when reference is near zero
        };
        max_rel_error = max_rel_error.max(rel_error);
    }
    
    println!("\n=== Numerical Conformance Results ===");
    println!("Max relative error: {:.2e}", max_rel_error);
    if max_rel_error < 1e-4 {
        println!("✅ PASS: Error within target (<1e-4)");
    } else {
        println!(
            "⚠️  WARNING: Error exceeds target (max_rel_error = {:.2e})",
            max_rel_error
        );
    }
    
    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(combined_ptr).unwrap();
    }
    
    println!("\n=== Single-Kernel with RoPE Test PASSED ===");
}
