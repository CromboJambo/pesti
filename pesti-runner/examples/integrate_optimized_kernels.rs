//! Integration test for optimized fused attention kernels (vectorized/tiled)

use pesti_runner::cuda_runtime::CudaRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_rt = CudaRuntime::new(0)?;
    let stream = cuda_rt.new_stream()?;

    println!("=== Optimized Kernel Integration Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);

    // Configuration
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;

    // Allocate host memory (f16)
    let q_h: Vec<half::f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| half::f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let k_h: Vec<half::f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| half::f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    // Allocate device memory
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * seq_k * 4;

    let mut q_d: Vec<u8> = vec![0u8; q_size];
    let mut k_d: Vec<u8> = vec![0u8; k_size];
    let mut s_d: Vec<u8> = vec![0u8; s_size];

    // Copy to device
    unsafe {
        std::ptr::copy_nonoverlapping(q_h.as_ptr() as *const u8, q_d.as_mut_ptr(), q_size);
        std::ptr::copy_nonoverlapping(k_h.as_ptr() as *const u8, k_d.as_mut_ptr(), k_size);
    }

    // Try loading tiled kernel first (uses vectorized loads internally)
    let ptx_path = "/home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/attention_rope_softmax_tiled.ptx";
    
    if std::path::Path::new(ptx_path).exists() {
        println!("✅ Tiled PTX found: {}", ptx_path);

        // Build kernel from module using builder
        let kernel = pesti_runner::kernel::fused_attention_conformant::FusedAttentionKernelBuilder::new(
            pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
            cuda_rt.context().clone(),
            stream.clone(),
        )
        .build_from_ptx_file(
            ptx_path,
            "_Z28fused_attention_kernel_tiledfPK6__halfS1_S1_Pfiiiifi", // mangled name for tiled version
        )?;

        // Launch kernel
        let scale = 1.0 / (head_dim as f32).sqrt();
        unsafe {
            kernel.launch(
                scale,
                q_d.as_ptr() as u64,
                k_d.as_ptr() as u64,
                0u64, // v_ptr not used in this version
                s_d.as_ptr() as u64,
                seq_q, seq_k, num_heads, head_dim, rope_base, seq_k
            )?;
        }

        cuda_rt.synchronize()?;

        // Copy results back
        let mut gpu_scores = vec![0.0f32; seq_q * seq_k];
        unsafe {
            std::ptr::copy_nonoverlapping(s_d.as_ptr(), gpu_scores.as_mut_ptr() as *mut u8, s_size);
        }

        println!("✅ Tiled kernel executed successfully");
        println!(
            "First 5 scores: {:?}",
            &gpu_scores[..5.min(gpu_scores.len())]
        );
    } else {
        println!("⚠️  PTX file not found at {}", ptx_path);
        println!(
            "  Compile first with: nvcc -arch=sm_89 -ptx attention_rope_softmax_tiled.cu"
        );
    }

    Ok(())
}
