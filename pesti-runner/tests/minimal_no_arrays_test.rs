//! Test minimal kernel with NO local arrays

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_minimal_no_arrays_kernel() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }
    
    println!("=== Minimal Kernel (No Arrays) Test ===");
    
    // Very small: 1x1x4, seq_k=2
    let seq_q = 1;
    let num_heads = 1;
    let head_dim = 4;
    let seq_k = 2;
    
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let out_size = seq_q * num_heads * head_dim * 2;
    
    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };
    
    println!("✅ Allocated minimal memory");
    
    // Load kernel
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_minimal.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();
    
    // Get mangled name
    let mangled_name = "_Z30fused_attention_minimal_kernelPK6__halfS1_S1_PS_fiiii";
    let function = module.load_function(mangled_name).unwrap();
    
    println!("✅ Loaded minimal kernel");
    
    // Parameters
    let mut scale_v: f32 = 1.0 / (head_dim as f32).sqrt();
    let mut q_ptr_v: u64 = q_ptr as u64;
    let mut k_ptr_v: u64 = k_ptr as u64;
    let mut v_ptr_v: u64 = v_ptr as u64;
    let mut out_ptr_v: u64 = out_ptr as u64;
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;
    
    let mut params: [*mut std::ffi::c_void; 9] = [
        &mut scale_v as *mut f32 as *mut std::ffi::c_void,
        &mut q_ptr_v as *mut u64 as *mut std::ffi::c_void,
        &mut k_ptr_v as *mut u64 as *mut std::ffi::c_void,
        &mut v_ptr_v as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr_v as *mut u64 as *mut std::ffi::c_void,
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
        &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
    ];
    
    let stream = cuda_rt.new_stream().unwrap();
    let grid = (1u32, 1u32, 1u32);
    let block = (4u32, 1u32, 1u32);  // head_dim threads
    
    println!("🚀 Launching with grid={:?}, block={:?}", grid, block);
    
    unsafe {
        pesti_runner::cuda_shim::launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            pesti_runner::cuda_shim::cu_stream(&stream),
            &mut params,
        ).unwrap();
    }
    
    println!("✅ Kernel launched");
    
    cuda_rt.synchronize().unwrap();
    println!("✅ Minimal kernel (no arrays) completed successfully!");
    
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
    
    println!("\n=== Minimal No-Arrays Test PASSED ===");
}
