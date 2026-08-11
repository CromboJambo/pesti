//! Performance profiling for fused attention kernel
//!
//! Measures actual execution time, occupancy, and memory bandwidth.

use pesti_runner::cuda_runtime::CudaRuntime;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_rt = CudaRuntime::new(0)?;

    println!("=== Fused Attention Kernel Performance Profile ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();

    // Configuration for profiling
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;

    println!("Test Configuration:");
    println!(
        "  seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );
    println!();

    // Allocate host memory (f16)
    let q_h: Vec<half::f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| half::f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let k_h: Vec<half::f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| half::f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let q_size = seq_q * num_heads * head_dim * 2; // f16 = 2 bytes
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * seq_k * 4; // f32 scores

    println!("Memory Requirements:");
    println!("  Q: {} bytes ({} MB)", q_size, q_size as f64 / 1e6);
    println!("  K: {} bytes ({} MB)", k_size, k_size as f64 / 1e6);
    println!("  V: {} bytes ({} MB)", v_size, v_size as f64 / 1e6);
    println!("  S: {} bytes ({} MB)", s_size, s_size as f64 / 1e6);
    println!();

    // Allocate device memory
    let q_ptr = unsafe { cuda_rt.allocate_device_memory(q_size)? };
    let k_ptr = unsafe { cuda_rt.allocate_device_memory(k_size)? };
    let v_ptr = unsafe { cuda_rt.allocate_device_memory(v_size)? };
    let s_ptr = unsafe { cuda_rt.allocate_device_memory(s_size)? };

    // Copy to device (measure H2D time)
    let start = std::time::Instant::now();
    unsafe {
        cuda_rt.copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)?;
        cuda_rt.copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)?;
    }
    let h2d_time = start.elapsed();
    println!(
        "Host→Device Copy: {} ({} GB/s)",
        h2d_time.as_micros(),
        ((q_size + k_size) as f64 / h2d_time.as_secs_f64() / 1e6).round()
    );

    // Build kernel
    let stream = cuda_rt.new_stream()?;
    let kernel =
        pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
            pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
            cuda_rt.context().clone(),
            stream.clone(),
        )?;

    println!("Kernel Build Time: {} ms", h2d_time.as_millis());
    println!();

    // Warmup run
    let scale = 1.0 / (head_dim as f32).sqrt();
    unsafe {
        kernel.launch(
            scale,
            q_ptr as u64,
            k_ptr as u64,
            v_ptr as u64,
            s_ptr as u64,
            seq_q,
            seq_k,
            num_heads,
            head_dim,
            rope_base,
            seq_k,
        )?;
    }
    cuda_rt.synchronize()?;
    println!("Warmup: ✅");

    // Benchmark runs
    let num_runs = 100;
    let mut total_time = std::time::Duration::new(0, 0);

    for _ in 0..num_runs {
        let start = std::time::Instant::now();

        unsafe {
            kernel.launch(
                scale,
                q_ptr as u64,
                k_ptr as u64,
                v_ptr as u64,
                s_ptr as u64,
                seq_q,
                seq_k,
                num_heads,
                head_dim,
                rope_base,
                seq_k,
            )?;
        }

        cuda_rt.synchronize()?;
        total_time += start.elapsed();
    }

    let avg_kernel_time = total_time / num_runs as u32;
    println!("Kernel Execution:");
    println!("  Avg time per run: {} µs", avg_kernel_time.as_micros());
    println!(
        "  Throughput: {:.1} runs/sec",
        num_runs as f64 / avg_kernel_time.as_secs_f64()
    );

    // Calculate theoretical peak performance
    let ops = seq_q * seq_k * num_heads * head_dim;
    let tflops = (ops as f64 * 2.0) / (avg_kernel_time.as_secs_f64() * 1e12); // 2 FLOPs per element
    println!("  Theoretical TFLOPS: {:.2}", tflops);

    // Copy results back (measure D2H time)
    let mut gpu_probs = vec![0.0f32; seq_q * seq_k];
    let start = std::time::Instant::now();
    unsafe {
        cuda_rt.copy_device_to_host(
            gpu_probs.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        )?;
    }
    let d2h_time = start.elapsed();
    println!(
        "Device→Host Copy: {} ({} GB/s)",
        d2h_time.as_micros(),
        ((s_size as f64) / d2h_time.as_secs_f6() / 1e6).round()
    );

    // Cleanup
    unsafe {
        cuda_rt.free_device_memory(q_ptr)?;
        cuda_rt.free_device_memory(k_ptr)?;
        cuda_rt.free_device_memory(v_ptr)?;
        cuda_rt.free_device_memory(s_ptr)?;
    }

    println!();
    println!("=== Performance Summary ===");
    println!("Total kernel time: {} µs", avg_kernel_time.as_micros());
    println!(
        "H2D + D2H overhead: {} µs",
        (h2d_time + d2h_time).as_micros()
    );
    println!(
        "Compute fraction: {:.1}%",
        100.0 * avg_kernel_time.as_micros() as f64
            / ((avg_kernel_time + h2d_time + d2h_time).as_micros()) as f64
    );

    Ok(())
}
