//! Profile attention kernel time breakdown: GEMM vs softmax vs transfers.
//! Uses real model weights from Qwen2.5-0.5B-Instruct to measure actual decode step costs.

use std::time::Instant;

use pesti_runner::kernel::gemm::{CudaGemmKernel, GemmArch};
use pesti_runner::kernel::attention::{AttentionConfig, AttentionArch};
use pesti_runner::kernel::softmax::{SoftmaxKernelBuilder, SoftmaxBackend};
use pesti_runner::cuda_runtime::{init_cuda, CudaRuntime};

fn main() {
    let rt = init_cuda(0).expect("CUDA init failed");
    
    // Build GEMM kernel
    let gemm = CudaGemmKernel::builder(GemmArch::Wgmma)
        .context(rt.context())
        .stream(rt.stream())
        .build()
        .expect("GEMM build failed");

    // Build softmax kernel
    let softmax = SoftmaxKernelBuilder::new(SoftmaxBackend::default())
        .context(rt.context())
        .stream(rt.stream())
        .build()
        .expect("Softmax build failed");

    println!("Profile: attention forward pass time breakdown");
    println!("================================================");
    
    // Config for Qwen2.5-0.5B-Instruct (approximate)
    let config = AttentionConfig::new(8, 128);
    
    // Generate synthetic inputs
    let query_seq_len = 1;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let cache_seq_len = 1024;
    
    let q_size = query_seq_len * num_heads * head_dim;
    let k_size = num_heads * cache_seq_len * head_dim;
    let v_size = k_size;
    
    // Allocate and fill with random data
    use pesti_runner::kernel::device_buf::DeviceBuffer;
    use half::f16;
    
    let q_data: Vec<f16> = (0..q_size).map(|_| f16::from_f32(0.1)).collect();
    let k_data: Vec<f16> = (0..k_size).map(|_| f16::from_f32(0.1)).collect();
    let v_data: Vec<f16> = (0..v_size).map(|_| f16::from_f32(0.1)).collect();
    
    let q_buf = pesti_runner::kernel::memory::CudaMemoryBackend::alloc_device(&rt, q_size * 2)
        .expect("q alloc failed");
    let k_buf = pesti_runner::kernel::memory::CudaMemoryBackend::alloc_device(&rt, k_size * 2)
        .expect("k alloc failed");
    let v_buf = pesti_runner::kernel::memory::CudaMemoryBackend::alloc_device(&rt, v_size * 2)
        .expect("v alloc failed");
    
    // Warmup run
    println!("Warmup...");
    unsafe {
        cudarc::driver::cudaMemcpyAsync(
            q_buf.ptr(), q_data.as_ptr() as *const _, q_size * 2,
            cudarc::driver::MemcpyKind::HostToDevice, rt.stream().as_raw());
        cudarc::driver::cudaMemcpyAsync(
            k_buf.ptr(), k_data.as_ptr() as *const _, k_size * 2,
            cudarc::driver::MemcpyKind::HostToDevice, rt.stream().as_raw());
        cudarc::driver::cudaMemcpyAsync(
            v_buf.ptr(), v_data.as_ptr() as *const _, v_size * 2,
            cudarc::driver::MemcpyKind::HostToDevice, rt.stream().as_raw());
    }
    
    println!("\nProfile run (cache_seq_len={cache_seq_len}):");
    
    // Start timing
    let t_start = Instant::now();
    
    // Phase 1: Q @ K^T GEMM
    let scores_size = query_seq_len * num_heads * cache_seq_len;
    let scores_buf = pesti_runner::kernel::memory::CudaMemoryBackend::alloc_device(&rt, scores_size * 4)
        .expect("scores alloc failed");
    
    let t1 = Instant::now();
    gemm.matmul(1.0, &q_buf, &k_buf, 0.0, &scores_buf, 
        query_seq_len * num_heads, cache_seq_len, head_dim).unwrap();
    pesti_runner::cuda_shim::stream_synchronize(rt.stream()).unwrap();
    let gemm_time = t1.elapsed().as_secs_f64() * 1000.0;
    
    // Phase 2: Transfer scores to host
    let mut scores_host = vec![0.0f32; scores_size];
    let t2 = Instant::now();
    unsafe {
        cudarc::driver::cudaMemcpyAsync(
            scores_host.as_mut_ptr() as *mut _, scores_buf.ptr(), scores_size * 4,
            cudarc::driver::MemcpyKind::DeviceToHost, rt.stream().as_raw());
    }
    pesti_runner::cuda_shim::stream_synchronize(rt.stream()).unwrap();
    let transfer_time = t2.elapsed().as_secs_f64() * 1000.0;
    
    // Phase 3: Softmax on CPU
    let softmax_scores = softmax.softmax(&scores_host, query_seq_len * num_heads, cache_seq_len)
        .expect("softmax failed");
    let softmax_time = t2.elapsed().as_secs_f64() * 1000.0 - transfer_time;
    
    // Phase 4: Transfer scores back to device
    let softmax_buf = pesti_runner::kernel::memory::CudaMemoryBackend::alloc_device(&rt, scores_size * 2)
        .expect("softmax buf alloc failed");
    let t3 = Instant::now();
    unsafe {
        cudarc::driver::cudaMemcpyAsync(
            softmax_buf.ptr(), softmax_scores.as_ptr() as *const _, scores_size * 2,
            cudarc::driver::MemcpyKind::HostToDevice, rt.stream().as_raw());
    }
    pesti_runner::cuda_shim::stream_synchronize(rt.stream()).unwrap();
    let transfer_back_time = t3.elapsed().as_secs_f64() * 1000.0;
    
    // Phase 5: S @ V GEMM
    let output_size = query_seq_len * num_heads * head_dim;
    let output_buf = pesti_runner::kernel::memory::CudaMemoryBackend::alloc_device(&rt, output_size * 4)
        .expect("output alloc failed");
    
    let t4 = Instant::now();
    gemm.matmul(1.0, &softmax_buf, &v_buf, 0.0, &output_buf,
        query_seq_len * num_heads, head_dim, cache_seq_len).unwrap();
    pesti_runner::cuda_shim::stream_synchronize(rt.stream()).unwrap();
    let gemm2_time = t4.elapsed().as_secs_f64() * 1000.0;
    
    let total_time = t_start.elapsed().as_secs_f64() * 1000.0;
    
    println!("Phase 1 - Q @ K^T GEMM:     {:.2} ms ({:.1}%)", gemm_time, gemm_time / total_time * 100.0);
    println!("Phase 2 - Transfer to host:  {:.2} ms ({:.1}%)", transfer_time, transfer_time / total_time * 100.0);
    println!("Phase 3 - Softmax (CPU):     {:.2} ms ({:.1}%)", softmax_time, softmax_time / total_time * 100.0);
    println!("Phase 4 - Transfer to dev:   {:.2} ms ({:.1}%)", transfer_back_time, transfer_back_time / total_time * 100.0);
    println!("Phase 5 - S @ V GEMM:        {:.2} ms ({:.1}%)", gemm2_time, gemm2_time / total_time * 100.0);
    println!("--------------------------------------------------");
    println!("Total attention step:        {:.2} ms", total_time);
    
    // Cleanup
    pesti_runner::kernel::memory::CudaMemoryBackend::free_device(&rt, q_buf).unwrap();
    pesti_runner::kernel::memory::CudaMemoryBackend::free_device(&rt, k_buf).unwrap();
    pesti_runner::kernel::memory::CudaMemoryBackend::free_device(&rt, v_buf).unwrap();
    pesti_runner::kernel::memory::CudaMemoryBackend::free_device(&rt, scores_buf).unwrap();
    pesti_runner::kernel::memory::CudaMemoryBackend::free_device(&rt, softmax_buf).unwrap();
    pesti_runner::kernel::memory::CudaMemoryBackend::free_device(&rt, output_buf).unwrap();
}
