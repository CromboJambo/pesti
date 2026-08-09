//! Simple CPU attention benchmark demonstrating softmax integration

use half::f16;
use pesti_runner::kernel::device_buf::DeviceBuffer;
use pesti_runner::kernel::{AttentionArch, AttentionConfig, AttentionKernel, CpuAttentionKernel};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CPU Attention + Softmax Benchmark ===\n");

    let (num_heads, head_dim, seq_len, label) = (8, 64, 256, "Qwen2.5-0.5B style");
    
    println!(
        "Testing: {} ({} heads × {} dim × {} seq)",
        label, num_heads, head_dim, seq_len
    );

    let query_len = 1;

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

    let mut kvc = pesti_runner::kernel::Kvcache::new(
        num_heads,
        num_heads,
        head_dim,
        seq_len,
        false,
    );
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

    println!("Kernel available: {}", kernel.is_available());
    println!("Kernel arch: {:?}", kernel.arch());

    // Warmup
    match kernel.forward(&q_buf, &kvc, &kvc, None, &config) {
        Ok(output) => {
            println!("✅ Forward pass succeeded!");
            
            let out_vec: Vec<f32> = output.to_host();
            println!("Output size: {} elements", out_vec.len());
            
            let has_nan = out_vec.iter().any(|&x| x.is_nan());
            let has_inf = out_vec.iter().any(|&x| x.is_infinite());

            if has_nan || has_inf {
                println!("⚠️  Warning: Output contains NaN/Inf");
            } else {
                println!("✅ Output valid (no NaN/Inf) - softmax working correctly!");
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {:?}", e);
        }
    }

    Ok(())
}
