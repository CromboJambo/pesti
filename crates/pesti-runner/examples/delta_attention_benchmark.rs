//! Route C Delta Attention Benchmark
//!
//! Compares throughput of standard fused decode attention (full f32 K/V)
//! vs delta-aware decode attention (base + i8 deltas) across sequence lengths.
//!
//! Usage: cargo run --package pesti-runner --example delta_attention_benchmark --features cuda

use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pesti_runner::kernel::fused_decode_attention::build_fused_decode_attention_kernel;
    use pesti_runner::kernel::fused_decode_attention_delta::build_delta_fused_decode_attention_kernel;
    use cudarc::driver::{CudaContext, CudaStream};

    println!("=== Route C Delta Attention Benchmark ===");
    println!();

    // Setup CUDA
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();

    // Build both kernels
    println!("Building standard fused decode kernel...");
    let std_kernel = build_fused_decode_attention_kernel(context.clone(), stream.clone())?;
    println!("  OK");

    println!("Building delta-aware decode kernel (Route C)...");
    let delta_kernel = build_delta_fused_decode_attention_kernel(context, stream.clone())?;
    println!("  OK");
    println!();

    // Benchmark configurations
    let head_dim = 64;
    let seq_lens = vec![16, 32, 64, 128, 256, 512]; // PTX shared mem limit: 512
    let warmup_iters = 10;
    let measure_iters = 100;

    println!("{:<8} {:<20} {:<20} {:<15}", "Seq Len", "Standard (us)", "Delta (us)", "Speedup");
    println!("{}", "-".repeat(65));

    for seq_len in seq_lens {
        // Generate synthetic data
        let q: Vec<f32> = (0..head_dim).map(|i| 0.1 * ((i as f32) % 7.0 - 3.5)).collect();
        let k_base: Vec<f32> = (0..head_dim).map(|i| 0.5 * ((i as f32) % 5.0 - 2.5)).collect();
        let v_base: Vec<f32> = (0..head_dim).map(|i| 0.3 * ((i as f32) % 9.0 - 4.5)).collect();

        // Standard path: full K/V tensors [seq_len x head_dim]
        let k_full: Vec<f32> = (0..seq_len * head_dim).map(|i| {
            let base_val = k_base[i % head_dim];
            base_val + 0.01 * ((i as f32) % 11.0 - 5.5)
        }).collect();
        let v_full: Vec<f32> = (0..seq_len * head_dim).map(|i| {
            let base_val = v_base[i % head_dim];
            base_val + 0.01 * ((i as f32) % 13.0 - 6.5)
        }).collect();

        // Delta path: base + i8 deltas
        let k_step = 0.05f32;
        let v_step = 0.05f32;
        let mut k_deltas: Vec<i8> = Vec::with_capacity((seq_len - 1) * head_dim);
        let mut v_deltas: Vec<i8> = Vec::with_capacity((seq_len - 1) * head_dim);

        for j in 1..seq_len {
            for i in 0..head_dim {
                let k_prev = k_full[(j - 1) * head_dim + i];
                let k_cur = k_full[j * head_dim + i];
                let delta = (k_cur - k_prev) / k_step;
                k_deltas.push(delta.round().clamp(-127.0, 127.0) as i8);

                let v_prev = v_full[(j - 1) * head_dim + i];
                let v_cur = v_full[j * head_dim + i];
                let delta = (v_cur - v_prev) / v_step;
                v_deltas.push(delta.round().clamp(-127.0, 127.0) as i8);
            }
        }

        // Allocate device memory
        let mut q_dev = unsafe { stream.alloc(q.len()) }?;
        let mut k_full_dev = unsafe { stream.alloc(k_full.len()) }?;
        let mut v_full_dev = unsafe { stream.alloc(v_full.len()) }?;
        let mut out_std_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(head_dim) }?;

        let mut k_base_dev = unsafe { stream.alloc(k_base.len()) }?;
        let mut v_base_dev = unsafe { stream.alloc(v_base.len()) }?;
        let mut k_deltas_dev = unsafe { stream.alloc(k_deltas.len()) }?;
        let mut v_deltas_dev = unsafe { stream.alloc(v_deltas.len()) }?;
        let mut out_delta_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(head_dim) }?;

        // Transfer data to device
        stream.memcpy_htod(&q, &mut q_dev)?;
        stream.memcpy_htod(&k_full, &mut k_full_dev)?;
        stream.memcpy_htod(&v_full, &mut v_full_dev)?;
        stream.memcpy_htod(&k_base, &mut k_base_dev)?;
        stream.memcpy_htod(&v_base, &mut v_base_dev)?;
        stream.memcpy_htod(&k_deltas, &mut k_deltas_dev)?;
        stream.memcpy_htod(&v_deltas, &mut v_deltas_dev)?;

        // Get raw device pointers
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let (q_ptr, _) = q_dev.device_ptr(&stream);
        let (k_full_ptr, _) = k_full_dev.device_ptr(&stream);
        let (v_full_ptr, _) = v_full_dev.device_ptr(&stream);
        let (out_std_ptr, _) = out_std_dev.device_ptr(&stream);
        let (k_base_ptr, _) = k_base_dev.device_ptr(&stream);
        let (v_base_ptr, _) = v_base_dev.device_ptr(&stream);
        let (k_deltas_ptr, _) = k_deltas_dev.device_ptr(&stream);
        let (v_deltas_ptr, _) = v_deltas_dev.device_ptr(&stream);
        let (out_delta_ptr, _) = out_delta_dev.device_ptr(&stream);

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Warmup
        for _ in 0..warmup_iters {
            std_kernel.launch(q_ptr, k_full_ptr, v_full_ptr, scale, seq_len, head_dim, out_std_ptr)?;
            delta_kernel.launch(
                q_ptr, k_base_ptr, v_base_ptr, k_deltas_ptr, v_deltas_ptr,
                k_step, v_step, scale, seq_len, head_dim, out_delta_ptr,
            )?;
        }

        // Measure standard path
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..measure_iters {
            std_kernel.launch(q_ptr, k_full_ptr, v_full_ptr, scale, seq_len, head_dim, out_std_ptr)?;
        }
        stream.synchronize()?;
        let std_time = start.elapsed().as_micros() as f64 / measure_iters as f64;

        // Measure delta path
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..measure_iters {
            delta_kernel.launch(
                q_ptr, k_base_ptr, v_base_ptr, k_deltas_ptr, v_deltas_ptr,
                k_step, v_step, scale, seq_len, head_dim, out_delta_ptr,
            )?;
        }
        stream.synchronize()?;
        let delta_time = start.elapsed().as_micros() as f64 / measure_iters as f64;

        // Report
        let speedup = std_time / delta_time;
        println!("{:<8} {:<20.1} {:<20.1} {:<15.2}x", seq_len, std_time, delta_time, speedup);
    }

    println!();
    println!("=== Benchmark Complete ===");
    println!();
    println!("Note: Kernel execution only (data pre-loaded on device).");
    println!("Real-world also benefits from reduced H2D transfer bandwidth.");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("CUDA feature required for this benchmark");
    std::process::exit(1);
}
