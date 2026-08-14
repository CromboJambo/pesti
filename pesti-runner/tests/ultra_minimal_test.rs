//! Ultra minimal test - just copy data from Q to out

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{launch_kernel, cu_stream};

#[cfg(feature = "cuda")]
#[test]
fn test_copy_kernel() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }

    println!("=== Ultra Minimal Copy Kernel Test ===");

    // Simple 1D copy: out[i] = q[i]
    let size = 16;
    
    let q_h: Vec<f16> = (0..size).map(|i| f16::from_f32(i as f32)).collect();
    let mut expected: Vec<f32> = (0..size).map(|i| i as f32).collect();

    let q_size = size * 2;
    let out_size = size * 2;

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };

    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
    }

    // Simple kernel: out[tid] = q[tid]
    let ptx_src = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry _Z18test_copy_kernelPfS_S_i(
    .param .u64 _Z18test_copy_kernelPfS_S_i_param_0,
    .param .u64 _Z18test_copy_kernelPfS_S_i_param_1,
    .param .u32 _Z18test_copy_kernelPfS_S_i_param_2
)
{
    .local .align 8 .size __local_DSA_0,112;
    .visible .shared .align 8 .size _ZN18test_copy_kernelEvPS_S_S_0,4;
    .visible .shared .align 8 .size _ZN18test_copy_kernelEvPS_S_1,4;
    
    // Get thread ID
    ld.global.u32 %r1, [.%tid];
    and.b32 %r1, %r1, %r1;
    
    // Check bounds
    ge.b32 %p1, %r1, %r2;
    @!%p1 bra done;
    
    // Copy q[tid] to out[tid]
    ld.global.f32 %f3, [%r0 + %r1*4];
    st.global.f32 [%r1 + %r1*4], %f3;
    
done:
    ret;
}
"#;

    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();
    let function = module.load_function("_Z18test_copy_kernelPfS_S_i").unwrap();

    let mut q_v: u64 = q_ptr as u64;
    let mut out_v: u64 = out_ptr as u64;
    let mut size_v: u32 = size as u32;

    let mut params: [*mut std::ffi::c_void; 3] = [
        &mut q_v as *mut u64 as *mut std::ffi::c_void,
        &mut out_v as *mut u64 as *mut std::ffi::c_void,
        &mut size_v as *mut u32 as *mut std::ffi::c_void,
    ];

    let stream = cuda_rt.new_stream().unwrap();
    
    unsafe {
        launch_kernel(
            function.cu_function(),
            (1u32, 1u32, 1u32),
            (16u32, 1u32, 1u32),
            0,
            cu_stream(&stream),
            &mut params,
        ).unwrap();
    }

    println!("✅ Kernel launched");
    cuda_rt.synchronize().unwrap();
    println!("✅ GPU execution completed!");

    let mut gpu_out = vec![0.0f32; size];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_out.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        ).unwrap();
    }

    println!("✅ Results: {:?}", &gpu_out[..4]);

    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
}
