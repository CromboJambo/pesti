//! Minimal test: verify kernel 1 (Q @ K^T) works correctly

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

fn main() {
    println!("=== Kernel 1 Sanity Check (Q @ K^T only) ===\n");

    let seq_q = 1;
    let seq_k = 2;
    let num_heads = 1;
    let head_dim = 4;
    let rope_base = 10_000.0;

    let q_h: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];

    let k_h: Vec<f16> = vec![
        f16::from_f32(5.0),
        f16::from_f32(6.0),
        f16::from_f32(7.0),
        f16::from_f32(8.0),
        f16::from_f32(9.0),
        f16::from_f32(10.0),
        f16::from_f32(11.0),
        f16::from_f32(12.0),
    ];

    let scale = 1.0 / (head_dim as f32).sqrt();

    println!("Q = [1, 2, 3, 4]");
    println!("K = [[5, 6, 7, 8], [9, 10, 11, 12]]");
    println!("Expected scores: [70.0, 110.0] (before scaling)");
    println!(
        "Expected scaled scores: [{:.1}, {:.1}]",
        scale * 70.0,
        scale * 110.0
    );

    let cuda_rt = CudaRuntime::new(0).unwrap();
    let stream = cuda_rt.new_stream().unwrap();

    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * num_heads * seq_k * 4; // scores output

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let s_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(s_size).unwrap() };

    println!("\nAllocated device memory:");
    println!("  q_ptr: {} bytes", q_size);
    println!("  k_ptr: {} bytes", k_size);
    println!("  s_ptr: {} bytes", s_size);

    let zero_init = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(
            s_ptr,
            zero_init.as_ptr() as *const u8,
            s_size,
        )
        .unwrap();
    }

    println!("Copied Q and K to device");

    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(
        &cuda_rt.context(),
        include_str!("../src/kernel/ptx/attention_rope_softmax.ptx"),
    )
    .unwrap();

    println!("Loaded PTX module");

    let func_name = "_Z22fused_attention_kernelfPK6__halfS1_S1_Pfiiiifi";
    let kernel_func = module.load_function(func_name).unwrap();

    println!("Found kernel function: {}", func_name);

    let mut scale_v: f32 = scale;
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut v_v: u64 = 0u64; // Not used in kernel 1
    let mut s_v: u64 = s_ptr as u64;
    let mut seq_q_v: i32 = seq_q as i32;
    let mut seq_k_v: i32 = seq_k as i32;
    let mut num_heads_v: i32 = num_heads as i32;
    let mut head_dim_v: i32 = head_dim as i32;
    let mut rope_base_v: f32 = rope_base;
    let mut max_pos_v: i32 = seq_k as i32;

    let mut params: [*mut std::ffi::c_void; 11] = [
        &mut scale_v as *mut f32 as *mut std::ffi::c_void,
        &mut q_v as *mut u64 as *mut std::ffi::c_void,
        &mut k_v as *mut u64 as *mut std::ffi::c_void,
        &mut v_v as *mut u64 as *mut std::ffi::c_void,
        &mut s_v as *mut u64 as *mut std::ffi::c_void,
        &mut seq_q_v as *mut i32 as *mut std::ffi::c_void,
        &mut seq_k_v as *mut i32 as *mut std::ffi::c_void,
        &mut num_heads_v as *mut i32 as *mut std::ffi::c_void,
        &mut head_dim_v as *mut i32 as *mut std::ffi::c_void,
        &mut rope_base_v as *mut f32 as *mut std::ffi::c_void,
        &mut max_pos_v as *mut i32 as *mut std::ffi::c_void,
    ];

    let grid = (1u32, 2u32, 1u32); // seq_q=1, seq_k=2, num_heads=1
    let block = (4u32, 1u32, 1u32);

    println!("\nLaunching kernel...");
    println!("  Grid: {:?}", grid);
    println!("  Block: {:?}", block);

    unsafe {
        match pesti_runner::cuda_shim::launch_kernel(
            kernel_func.cu_function(),
            grid,
            block,
            0,
            pesti_runner::cuda_shim::cu_stream(&stream),
            &mut params,
        ) {
            Ok(_) => println!("✅ Kernel launched successfully!"),
            Err(e) => {
                println!("❌ Launch failed: {:?}", e);
                return;
            }
        }
    }

    println!("Kernel launch returned (not hanging!)");

    cuda_rt.synchronize().unwrap();
    println!("Synchronized stream");

    let mut gpu_scores = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_scores.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        )
        .unwrap();
    }

    println!("\nGPU Output: [{:.4}, {:.4}]", gpu_scores[0], gpu_scores[1]);

    let expected_0 = scale * 70.0;
    let expected_1 = scale * 110.0;

    let err_0 = (gpu_scores[0] - expected_0).abs();
    let err_1 = (gpu_scores[1] - expected_1).abs();

    println!("\nExpected: [{:.4}, {:.4}]", expected_0, expected_1);
    println!("Error: [{:.6e}, {:.6e}]", err_0, err_1);

    if err_0 < 1e-3 && err_1 < 1e-3 {
        println!("\n✅ PASS - Kernel 1 (Q @ K^T) works correctly!");
    } else {
        println!("\n❌ FAIL - Output doesn't match expected values");
    }
}
