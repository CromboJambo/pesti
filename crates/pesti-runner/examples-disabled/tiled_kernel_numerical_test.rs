//! Numerical conformance test for tiled kernel vs CPU reference (raw dot products)

use pesti_runner::cuda_runtime::CudaRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_rt = CudaRuntime::new(0)?;
    let stream = cuda_rt.new_stream()?;

    println!("=== Tiled Kernel Numerical Conformance (Raw Dot Products) ===");
    println!("GPU: {}", cuda_rt.device_info().name);

    // Configuration (same as CPU reference)
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base: f32 = 10_000.0;

    // Allocate host memory (f16)
    let q_h: Vec<half::f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| half::f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let k_h: Vec<half::f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| half::f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    // CPU reference (with RoPE applied)
    let mut cpu_scores = vec![0.0f32; seq_q * num_heads * seq_k];

    let half_dim = head_dim as f32 / 2.0;

    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for k_pos in 0..seq_k {
                let mut sum_dot = 0.0f32;

                for d in (0..head_dim).step_by(2) {
                    let q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
                    let k_idx = k_pos * num_heads * head_dim + head * head_dim + d;

                    let q0 = q_h[q_idx].to_f32();
                    let q1 = q_h[q_idx + 1].to_f32();
                    let k0 = k_h[k_idx].to_f32();
                    let k1 = k_h[k_idx + 1].to_f32();

                    // Apply RoPE to Q pair
                    let inv_freq = 1.0f32 / (rope_base.powf((d as f32 / 2.0) / half_dim));
                    let freq_q = q_pos as f32 * inv_freq;
                    let c_q = freq_q.cos();
                    let s_q = freq_q.sin();

                    let new_q0 = q0 * c_q - q1 * s_q;
                    let new_q1 = q0 * s_q + q1 * c_q;

                    // Apply RoPE to K pair
                    let freq_k = k_pos as f32 * inv_freq;
                    let c_k = freq_k.cos();
                    let s_k = freq_k.sin();

                    let new_k0 = k0 * c_k - k1 * s_k;
                    let new_k1 = k0 * s_k + k1 * c_k;

                    sum_dot += new_q0 * new_k0 + new_q1 * new_k1;
                }

                let cpu_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
                cpu_scores[cpu_idx] = sum_dot;
            }
        }
    }

    // Allocate device memory (matching original kernel output size)
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * num_heads * seq_k * 4; // Same as original

    let mut q_d: Vec<u8> = vec![0u8; q_size];
    let mut k_d: Vec<u8> = vec![0u8; k_size];
    let mut s_d: Vec<u8> = vec![0u8; s_size];

    // Copy to device
    unsafe {
        std::ptr::copy_nonoverlapping(q_h.as_ptr() as *const u8, q_d.as_mut_ptr(), q_size);
        std::ptr::copy_nonoverlapping(k_h.as_ptr() as *const u8, k_d.as_mut_ptr(), k_size);
    }

    // Load tiled kernel
    let ptx_path =
        "/home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/attention_rope_softmax_tiled.ptx";

    if std::path::Path::new(ptx_path).exists() {
        println!("✅ Loading tiled kernel from {}", ptx_path);

        let kernel =
            pesti_runner::kernel::fused_attention_conformant::FusedAttentionKernelBuilder::new(
                pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
                cuda_rt.context().clone(),
                stream.clone(),
            )
            .build_from_ptx_file(
                ptx_path,
                "_Z28fused_attention_kernel_tiledfPK6__halfS1_S1_Pfiiiifi",
            )?;

        // Launch kernel (no softmax yet - raw dot products only)
        let scale = 1.0 / (head_dim as f32).sqrt();
        unsafe {
            kernel.launch(
                scale,
                q_d.as_ptr() as u64,
                k_d.as_ptr() as u64,
                0u64, // v_ptr not used in this version
                s_d.as_ptr() as u64,
                seq_q,
                seq_k,
                num_heads,
                head_dim,
                rope_base,
                seq_k,
            )?;
        }

        cuda_rt.synchronize()?;

        // Copy results back
        let mut gpu_scores = vec![0.0f32; seq_q * num_heads * seq_k];
        unsafe {
            std::ptr::copy_nonoverlapping(s_d.as_ptr(), gpu_scores.as_mut_ptr() as *mut u8, s_size);
        }

        // Compare with CPU reference (raw dot products)
        let mut max_error = 0.0;
        let total_elements = seq_q * num_heads * seq_k;

        for i in 0..total_elements {
            let diff = (gpu_scores[i] - cpu_scores[i]).abs();
            if diff > max_error {
                max_error = diff;
            }

            // Debug: print first few mismatches
            if diff > 1.0 && i < 20 {
                println!(
                    "Mismatch at idx {}: GPU={}, CPU={}",
                    i, gpu_scores[i], cpu_scores[i]
                );
            }
        }

        // Also check last few elements (might be uninitialized)
        for i in (total_elements - 10..total_elements).filter(|&x| x > 9) {
            let diff = (gpu_scores[i] - cpu_scores[i]).abs();
            if diff > max_error {
                max_error = diff;
            }
            println!(
                "Last idx {}: GPU={}, CPU={}, diff={}",
                i, gpu_scores[i], cpu_scores[i], diff
            );
        }

        println!("✅ Tiled kernel executed successfully");
        println!(
            "First 10 CPU scores: {:?}",
            &cpu_scores[..10.min(cpu_scores.len())]
        );
        println!(
            "First 10 GPU scores: {:?}",
            &gpu_scores[..10.min(gpu_scores.len())]
        );
        println!("Max absolute error: {:.2e}", max_error);

        if max_error < 1e-4 {
            println!("✅ PASS: Numerical conformance achieved (error < 1e-4)");
        } else {
            println!(
                "⚠️  WARNING: Error exceeds threshold ({} > 1e-4)",
                max_error
            );
        }
    } else {
        println!("⚠️  PTX file not found at {}", ptx_path);
    }

    Ok(())
}
