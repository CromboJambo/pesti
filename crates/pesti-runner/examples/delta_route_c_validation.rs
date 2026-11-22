//! Route C validation: delta-encoded attention PTX kernel vs CPU reference.
//!
//! Tests the fused_attention_delta PTX kernel against a pure-Rust CPU reference
//! implementation to verify numerical correctness before integration.
//!
//! Usage: cargo run --package pesti-runner --example delta_route_c_validation --features cuda

use std::sync::Arc;
use std::hash::{Hasher, Hash};

#[cfg(feature = "cuda")]

/// Naive full attention: reconstruct all K/V, compute standard attention.
fn naive_attention_reference(
    q: &[f32],
    k_base: &[f32],
    v_base: &[f32],
    k_deltas: &[i8],
    v_deltas: &[i8],
    k_step: f32,
    v_step: f32,
    scale: f32,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Reconstruct all K vectors
    let mut k_all = vec![vec![0.0f32; head_dim]; seq_len];
    for i in 0..head_dim {
        k_all[0][i] = k_base[i];
    }
    for pos in 1..seq_len {
        for i in 0..head_dim {
            let delta_idx = (pos - 1) * head_dim + i;
            k_all[pos][i] = k_all[pos-1][i] + (k_deltas[delta_idx] as f32) * k_step;
        }
    }

    // Reconstruct all V vectors
    let mut v_all = vec![vec![0.0f32; head_dim]; seq_len];
    for i in 0..head_dim {
        v_all[0][i] = v_base[i];
    }
    for pos in 1..seq_len {
        for i in 0..head_dim {
            let delta_idx = (pos - 1) * head_dim + i;
            v_all[pos][i] = v_all[pos-1][i] + (v_deltas[delta_idx] as f32) * v_step;
        }
    }

    // Standard attention: scores = q . K[pos] * scale
    let mut scores: Vec<f32> = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let mut dot = 0.0f32;
        for i in 0..head_dim {
            dot += q[i] * k_all[pos][i];
        }
        scores.push(dot * scale);
    }

    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let total: f32 = exps.iter().sum();
    for e in exps.iter_mut() { *e /= total; }

    // Weighted V sum
    let mut output = vec![0.0f32; head_dim];
    for pos in 0..seq_len {
        for i in 0..head_dim {
            output[i] += exps[pos] * v_all[pos][i];
        }
    }
    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pesti_runner::kernel::delta_kvcache::{DeltaKvCache, DeltaQuantizer};
    use pesti_runner::kernel::device_buf::DeviceBuffer;
    use pesti_runner::kernel::fused_decode_attention_delta::build_delta_fused_decode_attention_kernel;
    use pesti_runner::kernel::memory::CudaMemoryBackend;
    use cudarc::driver::CudaContext;

    println!("=== Route C: Delta Attention PTX Kernel Validation ===\n");

    // Test configurations: (head_dim, seq_len, label)
    let configs = vec![
        (64, 8, "Tiny (smoke test)"),
        (64, 32, "Small"),
        (64, 128, "Medium"),
        (64, 512, "Large"),
    ];

    for (head_dim, seq_len, label) in configs {
        println!("--- {} (head_dim={}, seq_len={}) ---", label, head_dim, seq_len);

        // Generate test data: smooth RoPE-like K and V sequences
        let k_data = generate_smooth_sequence(head_dim, seq_len, 0.1);
        let v_data = generate_smooth_sequence(head_dim, seq_len, 0.05);
        let q = generate_query(head_dim, 42);

        // Quantize to delta format
        let scale = 1.0 / (head_dim as f32).sqrt();
        let k_step = DeltaQuantizer::step_size_prevent_clipping(
            head_dim, 1.5, 2.0, seq_len, 8.0,
        ).unwrap_or(0.01);
        let v_step = k_step;

        let mut delta_cache = DeltaKvCache::new(1, head_dim, seq_len, k_step, v_step);
        for pos in 0..seq_len {
            delta_cache.append(&k_data[pos], &v_data[pos])?;
        }

        println!("  K step: {:.6}, V step: {:.6}", k_step, v_step);
        println!("  Clipped deltas: {}", delta_cache.total_clipped());

        // Get base vectors and delta arrays for head 0
        let k_base = delta_cache.k_base_head(0).to_vec();
        let v_base = delta_cache.v_base_head(0).to_vec();
        let k_deltas = delta_cache.k_deltas_head_flat(0);
        let v_deltas = delta_cache.v_deltas_head_flat(0);

        // === CPU Reference Implementation ===
        let ref_output = cpu_reference_attention(
            &q, &k_base, &v_base, &k_deltas, &v_deltas,
            k_step, v_step, scale, seq_len, head_dim,
        );
        
        // Cross-validate: naive full-attention vs delta-based reference
        let naive_ref = naive_attention_reference(
            &q, &k_base, &v_base, &k_deltas, &v_deltas,
            k_step, v_step, scale, seq_len, head_dim,
        );
        if head_dim == 64 && seq_len == 8 {
            let (naive_l2, _) = compare_vectors(&ref_output, &naive_ref);
            println!("  Delta-ref vs Naive-ref L2: {:.10}", naive_l2);
            println!("  Naive[0..8]: {:?}", &naive_ref[..8]);
        }

        // === GPU Kernel Execution ===
        let gpu_output = run_gpu_kernel(
            &q, &k_base, &v_base, &k_deltas, &v_deltas,
            k_step, v_step, scale, seq_len, head_dim,
        )?;

        // Debug for tiny config
        if head_dim == 64 && seq_len == 8 {
            println!("  Q[0..4]: {:?}", &q[..4]);
            println!("  K_base[0..4]: {:?}", &k_base[..4]);
            println!("  V_base[0..4]: {:?}", &v_base[..4]);
            println!("  Ref[0..8]: {:?}", &ref_output[..8]);
            println!("  GPU[0..8]: {:?}", &gpu_output[..8]);
        }

        // === Compare ===
        let (l2_error, max_abs_diff) = compare_vectors(&ref_output, &gpu_output);
        let ref_norm: f32 = ref_output.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel_error = if ref_norm > 0.0 { l2_error / ref_norm } else { 0.0 };

        println!("  L2 error (ref vs GPU): {:.10}", l2_error);
        println!("  Max abs diff:          {:.10}", max_abs_diff);
        println!("  Relative L2 error:     {:.8} ({:.6}%)", rel_error, rel_error * 100.0);

        if rel_error < 0.001 {
            println!("  ✅ PASS — relative error < 0.1%");
        } else if rel_error < 0.01 {
            println!("  ⚠️  MARGINAL — relative error 0.1-1% (check FP accumulation)");
        } else {
            println!("  ❌ FAIL — relative error > 1%, kernel likely incorrect");
        }
        println!();
    }

    println!("=== Route C Validation Complete ===");
    Ok(())
}

/// CPU reference: computes the same math as the PTX kernel in f32.
fn cpu_reference_attention(
    q: &[f32],
    k_base: &[f32],
    v_base: &[f32],
    k_deltas: &[i8],
    v_deltas: &[i8],
    k_step: f32,
    v_step: f32,
    scale: f32,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Reconstruct K at each position and compute scores
    let mut scores = Vec::with_capacity(seq_len);
    let mut k_recon = k_base.to_vec();

    for pos in 0..seq_len {
        if pos > 0 {
            // Apply delta to get K[pos]
            let row_start = (pos - 1) * head_dim;
            for i in 0..head_dim {
                let delta_val = k_deltas[row_start + i] as f32 * k_step;
                k_recon[i] += delta_val;
            }
        }
        // score[pos] = q . K[pos] * scale
        let dot: f32 = q.iter().zip(k_recon.iter()).map(|(a, b)| a * b).sum();
        scores.push(dot * scale);
    }

    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = scores
        .iter()
        .map(|&s| ((s - max_score) * 1.44269504f32).exp()) // exp via log2(e)
        .collect();
    let sum_w: f32 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum_w;
    }

    // Weighted V accumulation using suffix sums (Route C math)
    // output[i] = v_base[i]*1.0 + sum_{j=1}^{N-1} delta_v[j][i]*v_step*suffix_w[j]
    // where suffix_w[j] = sum_{pos=j}^{N-1} w[pos]
    let mut suffix_w = vec![0.0f32; seq_len];
    if seq_len > 0 {
        suffix_w[seq_len - 1] = weights[seq_len - 1];
        for j in (0..seq_len - 1).rev() {
            suffix_w[j] = suffix_w[j + 1] + weights[j];
        }
    }

    let mut output = vec![0.0f32; head_dim];
    for i in 0..head_dim {
        output[i] = v_base[i]; // * 1.0 (sum of all normalized weights)
        for j in 1..seq_len {
            let row_start = (j - 1) * head_dim;
            let delta_val = v_deltas[row_start + i] as f32 * v_step;
            output[i] += delta_val * suffix_w[j];
        }
    }

    output
}

/// Run the GPU kernel and return the output.
fn run_gpu_kernel(
    q: &[f32],
    k_base: &[f32],
    v_base: &[f32],
    k_deltas: &[i8],
    v_deltas: &[i8],
    k_step: f32,
    v_step: f32,
    scale: f32,
    seq_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use pesti_runner::kernel::fused_decode_attention_delta::build_delta_fused_decode_attention_kernel;
    use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();

    // Allocate device memory using cudarc safe API
    let mut q_dev = unsafe { stream.alloc(q.len()) }?;
    let mut k_base_dev = unsafe { stream.alloc(k_base.len()) }?;
    let mut v_base_dev = unsafe { stream.alloc(v_base.len()) }?;
    let mut k_deltas_dev = unsafe { stream.alloc(k_deltas.len()) }?;
    let mut v_deltas_dev = unsafe { stream.alloc(v_deltas.len()) }?;
    let out_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(head_dim) }?;

    // Copy to device
    stream.memcpy_htod(q, &mut q_dev)?;
    stream.memcpy_htod(k_base, &mut k_base_dev)?;
    stream.memcpy_htod(v_base, &mut v_base_dev)?;
    stream.memcpy_htod(k_deltas, &mut k_deltas_dev)?;
    stream.memcpy_htod(v_deltas, &mut v_deltas_dev)?;

    // Build and launch kernel (need Arc-wrapped context/stream for the wrapper)
    let kernel = build_delta_fused_decode_attention_kernel(context, stream.clone())?;

    // Get raw device pointers via DevicePtr trait
    let (q_ptr, _sq) = q_dev.device_ptr(&stream);
    let (k_base_ptr, _sk) = k_base_dev.device_ptr(&stream);
    let (v_base_ptr, _sv) = v_base_dev.device_ptr(&stream);
    let (k_deltas_ptr, _sdk) = k_deltas_dev.device_ptr(&stream);
    let (v_deltas_ptr, _sdv) = v_deltas_dev.device_ptr(&stream);
    let (out_ptr, _so) = out_dev.device_ptr(&stream);

    kernel.launch(
        q_ptr,
        k_base_ptr,
        v_base_ptr,
        k_deltas_ptr,
        v_deltas_ptr,
        k_step,
        v_step,
        scale,
        seq_len,
        head_dim,
        out_ptr,
    )?;

    // Synchronize and read back
    stream.synchronize()?;
    
    let mut output = vec![0.0f32; head_dim];
    stream.memcpy_dtoh(&out_dev, &mut output)?;

    Ok(output)
}

fn compare_vectors(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut l2 = 0.0f32;
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        l2 += diff * diff;
        if diff.abs() > max_abs {
            max_abs = diff.abs();
        }
    }
    (l2.sqrt(), max_abs)
}

fn generate_smooth_sequence(head_dim: usize, seq_len: usize, rotation_rate: f32) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let t = pos as f32;
        let mut vec = Vec::with_capacity(head_dim);
        for i in 0..head_dim {
            let theta = 10000.0f32.powf(-2.0 * i as f32 / head_dim as f32);
            let angle = t * theta * rotation_rate + i as f32 * 0.7;
            vec.push(angle.sin());
        }
        result.push(vec);
    }
    result
}

fn generate_query(head_dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = std::collections::hash_map::DefaultHasher::new();
    rng.write_u64(seed);
    (0..head_dim)
        .map(|_| {
            let h = rng.finish() as f32 / u64::MAX as f32;
            (h - 0.5) * 2.0 // [-1, 1]
        })
        .collect()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("⚠️  Requires --features cuda");
}
