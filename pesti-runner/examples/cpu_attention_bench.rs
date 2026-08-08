//! Simple CPU attention benchmark (no GPU required)
//!
//! Measures performance of SIMD-optimized CpuAttentionKernel
//! Usage: cargo run --package pesti-runner --example cpu_attention_bench

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half::f16;
    use pesti_runner::kernel::device_buf::DeviceBuffer;
    use pesti_runner::kernel::{AttentionArch, AttentionConfig, AttentionKernel, CpuAttentionKernel};
    use std::time::Instant;

    println!("=== CPU Attention SIMD Benchmark ===\n");

    // Test configs matching the optimization plan
    let configs = vec![
        (8, 64, 256, "Qwen2.5-0.5B style"),
        (16, 128, 512, "Medium config"),
        (32, 128, 1024, "Large config"),
    ];

    for (num_heads, head_dim, seq_len, label) in configs {
        println!(
            "Testing: {} ({} heads × {} dim × {} seq)",
            label, num_heads, head_dim, seq_len
        );

        let query_len = 1; // Single-token decode

        // Generate deterministic data
        let q_size = query_len * num_heads * head_dim;
        let kv_size = num_heads * head_dim * seq_len;

        let q_host: Vec<f16> = (0..q_size)
            .map(|i| f16::from_f32(((i as f32 * 0.1).sin() + 1.0) / 2.0))
            .collect();
        let k_host: Vec<f16> = (0..kv_size)
            .map(|i| f16::from_f32(((i as f32 * 0.07).sin() + 1.0) / 2.0))
            .collect();
        let v_host: Vec<f16> = (0..kv_size)
            .map(|i| f16::from_f32(((i as f32 * 0.03).sin() + 1.0) / 2.0))
            .collect();

        // Create Kvcache
        let mut kvc =
            pesti_runner::kernel::Kvcache::new(num_heads, num_heads, head_dim, seq_len, false);
        let head_stride = num_heads * head_dim;
        for pos in 0..seq_len {
            kvc.write_kv_at(
                pos,
                &k_host[pos * head_stride..(pos + 1) * head_stride],
                &v_host[pos * head_stride..(pos + 1) * head_stride],
            )?;
        }

        let q_buf = DeviceBuffer::from_host(q_host);
        let config = AttentionConfig::default()
            .with_num_heads(num_heads)
            .with_head_dim(head_dim)
            .with_max_seq(seq_len);
        let kernel = CpuAttentionKernel::new(AttentionArch::Cpu);

        // Warmup
        let _ = kernel.forward(&q_buf, &kvc, &kvc, None, &config)?;

        // Benchmark
        let loops = if seq_len <= 128 { 20 } else { 5 };
        let start = Instant::now();
        let output = kernel.forward(&q_buf, &kvc, &kvc, None, &config)?;
        for _ in 0..loops {
            let _ = kernel.forward(&q_buf, &kvc, &kvc, None, &config)?;
        }
        let elapsed = start.elapsed() / loops;

        let tok_s = query_len as f32 / elapsed.as_secs_f32();
        println!(
            "  Time: {:.2}ms/iter, {:.1} tok/s",
            elapsed.as_secs_f64() * 1000.0,
            tok_s
        );

        // Verify output is reasonable (not NaN/Inf)
        let out_vec: Vec<f32> = output.to_host();
        let has_nan = out_vec.iter().any(|&x| x.is_nan());
        let has_inf = out_vec.iter().any(|&x| x.is_infinite());

        if has_nan || has_inf {
            println!("  ⚠️  Warning: Output contains NaN/Inf");
        } else {
            println!("  ✅ Output valid (no NaN/Inf)");
        }

        println!();
    }

    println!("=== Benchmark Complete ===");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("⚠️  Requires --features cuda");
}
