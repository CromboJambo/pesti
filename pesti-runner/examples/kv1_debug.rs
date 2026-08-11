//! Minimal test with CORRECT causal mask expectations

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

fn main() {
    println!("=== Kernel 1 Debug Test (with correct causal mask) ===\n");
    
    let seq_q = 1;
    let seq_k = 2;
    let num_heads = 1;
    let head_dim = 4;
    
    // Q = [1, 2, 3, 4] (one query vector)
    let q_h: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    
    // K = [[5, 6, 7, 8], [9, 10, 11, 12]] (two key vectors)
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
    
    let mut scale = 1.0 / (head_dim as f32).sqrt(); // 0.5
    
    println!("Input data:");
    println!(
        "  Q (host): [{:.1}, {:.1}, {:.1}, {:.1}]",
        q_h[0].to_f32(),
        q_h[1].to_f32(),
        q_h[2].to_f32(),
        q_h[3].to_f32()
    );
    println!(
        "  K (host): [[{:.1}, {:.1}, {:.1}, {:.1}], [{:.1}, {:.1}, {:.1}, {:.1}]]",
        k_h[0].to_f32(),
        k_h[1].to_f32(),
        k_h[2].to_f32(),
        k_h[3].to_f32(),
        k_h[4].to_f32(),
        k_h[5].to_f32(),
        k_h[6].to_f32(),
        k_h[7].to_f32()
    );
    
    let cuda_rt = CudaRuntime::new(0).unwrap();
    let stream = cuda_rt.new_stream().unwrap();
    
    let q_size = seq_q * num_heads * head_dim * 2; // 8 bytes
    let k_size = seq_k * num_heads * head_dim * 2; // 16 bytes
    let s_size = seq_q * num_heads * seq_k * 4; // 8 bytes (2 scores)
    
    println!("\nMemory sizes:");
    println!("  q_size: {} bytes ({} f16)", q_size, q_size / 2);
    println!("  k_size: {} bytes ({} f16)", k_size, k_size / 2);
    println!("  s_size: {} bytes ({} f32)", s_size, s_size / 4);
    
    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let s_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(s_size).unwrap() };
    
    // Zero-init output
    let zero_init = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(
            s_ptr,
            zero_init.as_ptr() as *const u8,
            s_size,
        )
            .unwrap();
    }
    
    // Copy inputs
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)
            .unwrap();
    }
    
    println!("\nCopied to device");
    
    // Load module directly
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(
        &cuda_rt.context(),
        include_str!("../src/kernel/ptx/attention_rope_softmax.ptx"),
    )
        .unwrap();
    
    let func_name = "_Z22fused_attention_kernelfPK6__halfS1_S1_Pfiiiifi";
    let kernel_func = module.load_function(func_name).unwrap();
    
    println!("Found kernel: {}", func_name);
    
    // Launch with grid (seq_q, seq_k, num_heads) = (1, 2, 1)
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut v_v: u64 = 0u64; // Not used in this test
    let mut s_v: u64 = s_ptr as u64;
    
    let mut seq_q_v: i32 = seq_q as i32;
    let mut seq_k_v: i32 = seq_k as i32;
    let mut num_heads_v: i32 = num_heads as i32;
    let mut head_dim_v: i32 = head_dim as i32;
    let mut rope_base_v: f32 = 10_000.0f32;
    let mut max_pos_v: i32 = seq_k as i32;
    
    let mut params: [*mut std::ffi::c_void; 11] = [
        &mut scale as *const f32 as *mut std::ffi::c_void,
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
    
    let grid = (1u32, 2u32, 1u32); // q_pos=0..1, k_pos=0..2, head=0
    let block = (4u32, 1u32, 1u32); // 4 threads per block for parallel dot product
    
    println!("\nLaunch config:");
    println!("  Grid: {:?}", grid);
    println!("  Block: {:?}", block);
    
    unsafe {
        match pesti_runner::cuda_shim::launch_kernel(
            kernel_func.cu_function(),
            grid,
            block,
            16u32, // shared memory: 4 threads * 4 bytes each = 16 bytes
            stream.cu_stream(),
            &mut params,
        ) {
            Ok(_) => println!("\n✅ Kernel launched!"),
            Err(e) => println!("\n❌ Kernel launch failed: {:?}", e),
        }
    }
    
    cuda_rt.synchronize().unwrap();
    
    // Read output
    let mut gpu_scores = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_scores.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        )
            .unwrap();
    }
    
    println!("\nGPU Output:");
    for (i, &score) in gpu_scores.iter().enumerate() {
        let q = i / seq_k;
        let k = i % seq_k;
        println!("  scores[{}, {}] = {:.4}", q, k, score);
    }
    
    // Manual computation (with causal mask: mask if k_pos > q_pos)
    let q_dot_k0 = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0; // 70.0
    let q_dot_k1 = 1.0 * 9.0 + 2.0 * 10.0 + 3.0 * 11.0 + 4.0 * 12.0; // 110.0
    
    println!("\nManual computation:");
    println!("  q·k[0] = {:.1}", q_dot_k0);
    println!("  q·k[1] = {:.1}", q_dot_k1);
    
    // With causal mask: k_pos=1 > q_pos=0, so score[1] should be -inf
    let expected_0 = scale * q_dot_k0; // 35.0
    let expected_1 = if 1 > 0 { f32::NEG_INFINITY } else { scale * q_dot_k1 }; // -inf
    
    println!(
        "  Expected (scaled + causal): [{:.4}, {:.4}]",
        expected_0, expected_1
    );
    
    // Check errors
    let err_0 = (gpu_scores[0] - expected_0).abs();
    let err_1 = if gpu_scores[1] == f32::NEG_INFINITY && expected_1 == f32::NEG_INFINITY {
        0.0
    } else {
        (gpu_scores[1] - expected_1).abs()
    };
    
    println!("\nErrors:");
    println!("  Error[0] = {:.6e}", err_0);
    println!("  Error[1] = {:.6e}", err_1);
    
    if err_0 < 1e-3 && (err_1 < 1e-3 || gpu_scores[1] == f32::NEG_INFINITY) {
        println!("\n✅ PASS - Output matches expected values!");
    } else {
        println!("\n❌ FAIL - Output doesn't match");
        
        // Check for specific issues
        if gpu_scores[0] == f32::NEG_INFINITY {
            println!("  ⚠️  First score is -inf (causal mask issue?)");
        }
    }
}
