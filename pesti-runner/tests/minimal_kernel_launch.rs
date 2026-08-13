//! Minimal kernel launch test - verify basic cuLaunchKernel works

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_minimal_kernel_launch() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }
    
    println!("=== Minimal Kernel Launch Test ===");
    
    // Simple kernel: just write a constant to output
    let seq_q = 2;
    let num_heads = 4;
    let head_dim = 16;
    
    let out_size = seq_q * num_heads * head_dim * 2; // f16
    
    let out_ptr = unsafe { 
        pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() 
    };
    
    println!("✅ Allocated {} bytes for output", out_size);
    
    // Load a simple kernel - we'll use the attention kernel but with minimal params
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();
    
    let mangled_name = "_Z29fused_attention_simple_kernelPK6__halfS1_S1_PS_fiiii";
    let function = module.load_function(mangled_name).unwrap();
    
    println!("✅ Loaded kernel: {}", mangled_name);
    
    // Prepare minimal parameters (all zeros except scale=1.0)
    let mut scale_v: f32 = 1.0;
    let mut q_ptr_v: u64 = out_ptr as u64;  // Reuse same pointer for all inputs (simplified)
    let mut k_ptr_v: u64 = out_ptr as u64;
    let mut v_ptr_v: u64 = out_ptr as u64;
    let mut out_ptr_v: u64 = out_ptr as u64;
    let mut seq_q_v: u32 = 1;  // Just process q=0
    let mut seq_k_v: u32 = 1;  // Just attend to k=0
    let mut num_heads_v: u32 = 1;  // Just one head
    let mut head_dim_v: u32 = 16;
    
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
    
    // Launch with minimal grid/block
    let stream = cuda_rt.new_stream().unwrap();
    let grid = (1u32, 1u32, 1u32);  // Just one block
    let block = (64u32, 1u32, 1u32);
    
    println!("🚀 Launching kernel with grid={:?}, block={:?}", grid, block);
    
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
    
    println!("✅ Kernel launched successfully!");
    
    // Synchronize and check for errors
    cuda_rt.synchronize().unwrap();
    println!("✅ Kernel completed (no errors)");
    
    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
    
    println!("\n=== Minimal Launch Test PASSED ===");
}
