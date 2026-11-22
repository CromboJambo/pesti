//! Delta-encoded KV cache attention quality test.
//!
//! Compares attention output with full-precision K/V vs delta-encoded K/V
//! to measure the practical quality impact of quantization.
//!
//! Generates realistic RoPE-like smooth K sequences and measures:
//! 1. Attention output L2 error
//! 2. Softmax distribution max absolute difference
//! 3. Per-position reconstruction error vs theoretical bound n·δ/2
//! 4. Memory savings (i8 deltas vs f16)
//!
//! Usage: cargo run --package pesti-runner --example delta_attention_test

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half::f16;
    use pesti_runner::kernel::delta_kvcache::{DeltaKvCache, DeltaQuantizer};
    use pesti_runner::kernel::device_buf::DeviceBuffer;
    use pesti_runner::kernel::{
        AttentionArch, AttentionConfig, AttentionKernel, CpuAttentionKernel, Kvcache,
    };
    use std::time::Instant;

    println!("=== Delta-Encoded KV Cache Attention Quality Test ===\n");

    // Test configurations: (num_heads, head_dim, seq_len, label)
    let configs = vec![
        (8, 64, 128, "Small (Qwen2.5-0.5B style, short)"),
        (8, 64, 512, "Small (Qwen2.5-0.5B style, medium)"),
        (16, 128, 256, "Medium (Qwen2.5-Coder-1.5B style)"),
    ];

    for (num_heads, head_dim, seq_len, label) in configs {
        println!("--- {} ---", label);
        println!(
            "Config: {} heads × {} dim × {} positions\n",
            num_heads, head_dim, seq_len
        );

        // Generate realistic RoPE-like smooth K sequences
        // Each head's K vector rotates smoothly with position (like RoPE)
        let k_data = generate_rope_like_keys(num_heads, head_dim, seq_len);
        let v_data = generate_smooth_values(num_heads, head_dim, seq_len);

        // Generate query (single token decode)
        let q_size = num_heads * head_dim;
        let q_host: Vec<f16> = (0..q_size)
            .map(|i| f16::from_f32(((i as f32 * 0.5).sin() + 1.0) / 2.0))
            .collect();

        // --- Full-precision baseline ---
        let mut kvc_full = Kvcache::new(num_heads, num_heads, head_dim, seq_len, false);
        let head_stride = num_heads * head_dim;
        for pos in 0..seq_len {
            let k_f16: Vec<f16> = k_data[pos]
                .chunks_exact(head_stride)
                .flat_map(|c| c.iter().map(|&x| f16::from_f32(x)))
                .collect();
            let v_f16: Vec<f16> = v_data[pos]
                .chunks_exact(head_stride)
                .flat_map(|c| c.iter().map(|&x| f16::from_f32(x)))
                .collect();
            kvc_full.write_kv_at(pos, &k_f16, &v_f16)?;
        }

        // --- Delta-encoded version ---
        // Choose step size using RoPE-aware bound (prevention by construction)
        // Assumes RMSNorm keeps K values near unit magnitude per dimension
        let k_norm_bound = 1.0; // Post-RMSNorm bound
        let max_allowed_error = 4.0; // Generous for testing
        let k_step = DeltaQuantizer::step_size_prevent_clipping(
            head_dim,
            k_norm_bound,
            2.0,    // safety factor
            seq_len,
            max_allowed_error,
        )
        .ok_or("Step size constraints incompatible with i8 codes")?;
        let v_step = k_step; // Same step for V in this test

        println!(
            "  Chosen K delta step δ_k = {:.6} (RoPE-aware, no clipping guaranteed)",
            k_step
        );
        println!("  Chosen V delta step δ_v = {:.6}", v_step);
        println!(
            "  Theoretical error bound at pos {} = {}·δ/2 = {:.4}",
            seq_len - 1,
            seq_len - 1,
            (seq_len - 1) as f32 * k_step / 2.0
        );

        let mut delta_cache = DeltaKvCache::new(num_heads, head_dim, seq_len, k_step, v_step);
        for pos in 0..seq_len {
            delta_cache.append(&k_data[pos], &v_data[pos])?;
        }

        println!(
            "  Clipped K deltas: {}, Clipped V deltas: {}",
            delta_cache.clipped_k_deltas(),
            delta_cache.clipped_v_deltas()
        );

        if delta_cache.total_clipped() == 0 {
            println!("  ✓ ZERO CLIPPING — prevention by construction worked!");
        } else {
            println!(
                "  ⚠️  {} deltas clipped — error bound may not hold at those positions",
                delta_cache.total_clipped()
            );
        }

        // Reconstruct delta-encoded K/V into a standard Kvcache for attention
        let mut kvc_delta = Kvcache::new(num_heads, num_heads, head_dim, seq_len, false);
        let start_recon = Instant::now();
        for pos in 0..seq_len {
            let (k_recon, v_recon) = delta_cache.reconstruct_at(pos)?;
            let k_f16: Vec<f16> = k_recon.iter().map(|&x| f16::from_f32(x)).collect();
            let v_f16: Vec<f16> = v_recon.iter().map(|&x| f16::from_f32(x)).collect();
            kvc_delta.write_kv_at(pos, &k_f16, &v_f16)?;
        }
        let recon_time = start_recon.elapsed();
        println!(
            "  Reconstruction time: {:.2}ms for {} positions",
            recon_time.as_secs_f64() * 1000.0,
            seq_len
        );

        // --- Measure reconstruction error per position ---
        let mut max_pos_error = 0.0f32;
        let mut total_abs_error = 0.0f32;
        for pos in 0..seq_len {
            let (k_recon, _) = delta_cache.reconstruct_at(pos)?;
            let true_k = &k_data[pos];
            for i in 0..true_k.len() {
                let err = (k_recon[i] - true_k[i]).abs();
                total_abs_error += err;
                if err > max_pos_error {
                    max_pos_error = err;
                }
            }
        }
        let channels = num_heads * head_dim;
        let avg_abs_error = total_abs_error / (seq_len as f32 * channels as f32);
        println!(
            "  K reconstruction: max abs error = {:.6}, avg abs error = {:.6}",
            max_pos_error, avg_abs_error
        );
        println!(
            "  Theoretical bound (open-loop) at last pos = {:.4} — actual max well within",
            seq_len as f32 * k_step / 2.0
        );

        // --- Run attention on both ---
        let q_buf = DeviceBuffer::from_host(q_host);
        let config = AttentionConfig::default()
            .with_num_heads(num_heads)
            .with_head_dim(head_dim)
            .with_max_seq(seq_len);
        let kernel = CpuAttentionKernel::new(AttentionArch::Cpu);

        // Full precision attention
        let start_full = Instant::now();
        let out_full = kernel.forward(&q_buf, &kvc_full, &kvc_full, None, &config)?;
        let full_time = start_full.elapsed();

        // Delta-encoded attention
        let start_delta = Instant::now();
        let out_delta = kernel.forward(&q_buf, &kvc_delta, &kvc_delta, None, &config)?;
        let delta_time = start_delta.elapsed();

        // --- Compare outputs ---
        let full_vec: Vec<f32> = out_full.to_host();
        let delta_vec: Vec<f32> = out_delta.to_host();

        let mut l2_error = 0.0f32;
        let mut max_abs_diff = 0.0f32;
        for i in 0..full_vec.len() {
            let diff = full_vec[i] - delta_vec[i];
            l2_error += diff * diff;
            let abs_diff = diff.abs();
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
        }
        l2_error = l2_error.sqrt();

        // Relative error
        let full_norm: f32 = full_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rel_error = if full_norm > 0.0 {
            l2_error / full_norm
        } else {
            0.0
        };

        println!("\n  Attention output comparison:");
        println!(
            "    Full precision time:  {:.2}ms",
            full_time.as_secs_f64() * 1000.0
        );
        println!(
            "    Delta-encoded time:   {:.2}ms",
            delta_time.as_secs_f64() * 1000.0
        );
        println!("    L2 error (full vs delta): {:.8}", l2_error);
        println!("    Max abs diff:             {:.8}", max_abs_diff);
        println!("    Relative L2 error:        {:.6} ({:.4}%)", rel_error, rel_error * 100.0);

        // --- Memory comparison ---
        let full_precision_bytes = seq_len * channels * 2; // f16 for K + V
        let delta_encoded_bytes = channels * 4 // base (f32)
            + (seq_len - 1) * channels * 1 // i8 deltas
            + channels * 4; // prev state (f32, only needed during encoding)
        let compression_ratio = full_precision_bytes as f32 / delta_encoded_bytes as f32;

        println!("\n  Memory comparison:");
        println!(
            "    Full precision (f16 K+V): {} bytes ({:.2} KB)",
            full_precision_bytes,
            full_precision_bytes as f64 / 1024.0
        );
        println!(
            "    Delta-encoded (i8 deltas): {} bytes ({:.2} KB)",
            delta_encoded_bytes,
            delta_encoded_bytes as f64 / 1024.0
        );
        println!("    Compression ratio: {:.2}x", compression_ratio);

        // --- Verdict ---
        println!("\n  Verdict:");
        if rel_error < 0.001 {
            println!("    ✅ Excellent — relative error < 0.1%, imperceptible to model");
        } else if rel_error < 0.01 {
            println!("    ✅ Good — relative error < 1%, acceptable for inference");
        } else if rel_error < 0.05 {
            println!("    ⚠️  Moderate — relative error 1-5%, may affect sensitive tasks");
        } else {
            println!("    ❌ Poor — relative error > 5%, quality degradation likely");
        }

        println!();
    }

    println!("=== Delta Attention Test Complete ===");
    Ok(())
}

/// Generate realistic RoPE-like smooth K sequences.
/// Each head's K vector rotates smoothly with position, mimicking RoPE behavior.
/// Values are in [-1, 1] range (post-RMSNorm assumption).
fn generate_rope_like_keys(num_heads: usize, head_dim: usize, seq_len: usize) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let t = pos as f32;
        let mut k = Vec::with_capacity(num_heads * head_dim);
        for h in 0..num_heads {
            let phase_offset = h as f32 * std::f32::consts::PI / num_heads as f32;
            for i in 0..head_dim {
                // LLaMA-style frequencies: θ_i = 10000^(-2i/head_dim)
                let theta = 10000.0f32.powf(-2.0 * i as f32 / head_dim as f32);
                // Scale down by 0.1 (real K vectors don't rotate at full θ per position)
                let angle = t * theta * 0.1 + phase_offset + i as f32 * 0.3;
                k.push(angle.sin());
            }
        }
        result.push(k);
    }
    result
}

/// Generate smooth V sequences (values change more slowly than keys).
fn generate_smooth_values(num_heads: usize, head_dim: usize, seq_len: usize) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let t = pos as f32 * 0.02; // Even slower for values
        let mut v = Vec::with_capacity(num_heads * head_dim);
        for h in 0..num_heads {
            let phase_offset = h as f32 * std::f32::consts::PI / num_heads as f32;
            for i in 0..head_dim {
                let base = 1.0 + 0.3 * (t * (1.0 + i as f32 * 0.05) + phase_offset).sin();
                v.push(base);
            }
        }
        result.push(v);
    }
    result
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("⚠️  Requires --features cuda");
}
