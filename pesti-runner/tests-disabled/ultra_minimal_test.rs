//! Ultra minimal test - just copy data from Q to out
//!
//! Uses a small f16 copy kernel (`copy_kernel`, compiled from
//! `src/kernel/ptx/copy_kernel.cu`) to exercise the full
//! PTX -> module -> function -> launch -> readback path.
//! The kernel is `out[i] = q[i]` for `i < size`.

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{cu_stream, launch_kernel};

#[cfg(feature = "cuda")]
#[test]
fn test_copy_kernel() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }

    println!("=== Ultra Minimal Copy Kernel Test ===");

    // Simple 1D f16 copy: out[i] = q[i]
    let size = 16;

    let q_h: Vec<f16> = (0..size).map(|i| f16::from_f32(i as f32)).collect();

    let q_size = size * 2;
    let out_size = size * 2;

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };

    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)
            .unwrap();
    }

    // Load the NVVM-generated copy kernel PTX (same pattern as the other
    // kernel tests in this directory).
    let ptx_src = include_str!("../src/kernel/ptx/copy_kernel.ptx");
    let module =
        pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();
    let function = module.load_function("_Z11copy_kernelPKtPti").unwrap();

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
        )
        .unwrap();
    }

    println!("✅ Kernel launched");
    cuda_rt.synchronize().unwrap();
    println!("✅ GPU execution completed!");

    // Read back f16 values and verify the copy.
    let mut gpu_out = vec![0u8; out_size];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_out.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        )
        .unwrap();
    }

    let mut mismatches = 0usize;
    for i in 0..size {
        let got = f16::from_bits(u16::from_le_bytes([gpu_out[i * 2], gpu_out[i * 2 + 1]]));
        let want = q_h[i];
        if got.to_f32() != want.to_f32() {
            mismatches += 1;
            println!(
                "  mismatch at {}: got {} want {}",
                i,
                got.to_f32(),
                want.to_f32()
            );
        }
    }
    println!(
        "✅ Results: first 4 = {:?}, mismatches = {}",
        (0..4)
            .map(|i| {
                f16::from_bits(u16::from_le_bytes([gpu_out[i * 2], gpu_out[i * 2 + 1]])).to_f32()
            })
            .collect::<Vec<_>>(),
        mismatches
    );
    assert_eq!(mismatches, 0, "copy kernel produced wrong values");

    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
}
