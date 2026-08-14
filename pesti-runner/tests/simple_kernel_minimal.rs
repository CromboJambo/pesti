//! Minimal test to isolate CUDA_ERROR_ILLEGAL_ADDRESS in simple kernel

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{launch_kernel, cu_stream};

#[cfg(feature = "cuda")]
#[test]
fn test_simple_kernel_minimal() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }

    println!("=== Minimal Simple Kernel Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);

    // Very small configuration to isolate the issue
    let seq_q = 1;
    let seq_k = 2;
    let num_heads = 1;
    let head_dim = 4;  // Small dimension

    println!("Configuration: seq_q={}, seq_k={}, heads={}, dim={}", 
             seq_q, seq_k, num_heads, head_dim);

    // Create simple Q, K, V (all ones)
    let q_h: Vec<f16> = vec![f16::from_f32(1.0); seq_q * num_heads * head_dim];
    let k_h: Vec<f16> = vec![f16::from_f32(1.0); seq_k * num_heads * head_dim];
    let v_h: Vec<f16> = vec![f16::from_f32(1.0); seq_k * num_heads * head_dim];

    // Allocate device memory
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let out_size = seq_q * num_heads * head_dim * 2;

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };

    // Copy to device
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size).unwrap();
    }

    // Load PTX
    let stream = cuda_rt.new_stream().unwrap();
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();

    let mangled_name = "_Z29fused_attention_simple_kernelPK6__halfS1_S1_PS_fiiii";
    let function = module.load_function(mangled_name).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch with minimal grid/block
    unsafe {
        let mut scale_v: f32 = scale;
        let mut q_v: u64 = q_ptr as u64;
        let mut k_v: u64 = k_ptr as u64;
        let mut v_v: u64 = v_ptr as u64;
        let mut out_v: u64 = out_ptr as u64;
        let mut seq_q_v: u32 = seq_q as u32;
        let mut seq_k_v: u32 = seq_k as u32;
        let mut num_heads_v: u32 = num_heads as u32;
        let mut head_dim_v: u32 = head_dim as u32;

        let mut params: [*mut std::ffi::c_void; 9] = [
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut out_v as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
        ];

        // Minimal grid/block: 1x1x1 grid, 32 threads
        let grid = (1u32, 1u32, 1u32);
        let block = (32u32, 1u32, 1u32);

        println!("Launching with grid={:?}, block={:?}", grid, block);

        launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            cu_stream(&stream),
            &mut params,
        ).unwrap();
    }

    println!("✅ Kernel launched");

    // Synchronize to catch errors
    cuda_rt.synchronize().unwrap();

    println!("✅ GPU execution completed successfully!");

    // Copy results back
    let mut gpu_out = vec![0.0f32; seq_q * num_heads * head_dim];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_out.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        ).unwrap();
    }

    println!("✅ Results copied back: {:?}", &gpu_out[..4]);

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
}
