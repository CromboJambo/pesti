//! Minimal sanity check: verify fused attention produces correct output for trivial case
//! 
//! Test case: 1 token, 2 cache positions, 1 head, dim=4
//! Manually compute expected output to verify kernel correctness

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

fn main() {
    println!("=== Fused Attention Sanity Check ===\n");
    
    // Trivial case: 1 query position, 2 key positions, 1 head, dim=4
    let seq_q = 1;
    let seq_k = 2;
    let num_heads = 1;
    let head_dim = 4;
    let rope_base = 10_000.0;
    
    // Use identity RoPE (pos=0) to simplify: cos(0)=1, sin(0)=0 → no rotation
    // Q = [1, 2, 3, 4], K = [[5, 6, 7, 8], [9, 10, 11, 12]]
    // V = [[1, 1, 1, 1], [2, 2, 2, 2]] (constant values per row)
    
    let q_h: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(2.0), 
        f16::from_f32(3.0), f16::from_f32(4.0)
    ];
    
    let k_h: Vec<f16> = vec![
        // Position 0: [5, 6, 7, 8]
        f16::from_f32(5.0), f16::from_f32(6.0), 
        f16::from_f32(7.0), f16::from_f32(8.0),
        // Position 1: [9, 10, 11, 12]
        f16::from_f32(9.0), f16::from_f32(10.0), 
        f16::from_f32(11.0), f16::from_f32(12.0)
    ];
    
    let v_h: Vec<f16> = vec![
        // Position 0: [1, 1, 1, 1]
        f16::from_f32(1.0), f16::from_f32(1.0), 
        f16::from_f32(1.0), f16::from_f32(1.0),
        // Position 1: [2, 2, 2, 2]
        f16::from_f32(2.0), f16::from_f32(2.0), 
        f16::from_f32(2.0), f16::from_f32(2.0)
    ];
    
    let scale = 1.0 / (head_dim as f32).sqrt(); // 1/sqrt(4) = 0.5
    
    println!("Input:");
    println!("  Q = [1, 2, 3, 4]");
    println!("  K = [[5, 6, 7, 8], [9, 10, 11, 12]]");
    println!("  V = [[1, 1, 1, 1], [2, 2, 2, 2]]");
    println!("  scale = {:.3}", scale);
    
    // Manual computation (no RoPE since pos=0):
    // Step 1: Q @ K^T → scores
    let q_dot_k0 = 1*5 + 2*6 + 3*7 + 4*8; // = 5+12+21+32 = 70
    let q_dot_k1 = 1*9 + 2*10 + 3*11 + 4*12; // = 9+20+33+48 = 110
    
    println!("\nStep 1: Q @ K^T");
    println!("  q·k[0] = {:.1}", q_dot_k0);
    println!("  q·k[1] = {:.1}", q_dot_k1);
    
    // Step 2: Apply scale
    let score0 = q_dot_k0 * scale; // = 35.0
    let score1 = q_dot_k1 * scale; // = 55.0
    
    println!("\nStep 2: Apply scale");
    println!("  score[0] = {:.1}", score0);
    println!("  score[1] = {:.1}", score1);
    
    // Step 3: Softmax (causal mask: q_pos=0, k_pos=0 OK; q_pos=0, k_pos=1 OK)
    let max_val = f32::max(score0, score1); // = 55.0
    let exp0 = (score0 - max_val).exp(); // = exp(-20.0) ≈ 2.06e-9
    let exp1 = (score1 - max_val).exp(); // = exp(0.0) = 1.0
    let sum = exp0 + exp1; // ≈ 1.0
    
    let weight0 = exp0 / sum; // ≈ 2.06e-9 (essentially 0)
    let weight1 = exp1 / sum; // = 1.0
    
    println!("\nStep 3: Softmax");
    println!("  max = {:.1}", max_val);
    println!("  exp[0] ≈ {:.2e}", exp0);
    println!("  exp[1] = {:.1}", exp1);
    println!("  weight[0] ≈ {:.2e} (essentially 0)", weight0);
    println!("  weight[1] = {:.1}", weight1);
    
    // Step 4: Weighted sum of V → output
    // Since weight[0]≈0, weight[1]=1, output = V[1] = [2, 2, 2, 2]
    let expected_output: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0]; // All dimensions should be 2.0
    
    println!("\nStep 4: Weighted sum of V");
    println!("  output = [ {:.1}, {:.1}, {:.1}, {:.1} ]", expected_output[0], expected_output[1], expected_output[2], expected_output[3]);
    
    println!("\n=== Expected GPU Output ===");
    println!("  [2.0, 2.0, 2.0, 2.0]");
    println!("  (essentially just V[1] since softmax weights = [0, 1])\n");
    
    // Now run actual kernel and compare
    println!("=== Running GPU Kernel ===");
    
    let cuda_rt = CudaRuntime::new(0).unwrap();
    let stream = cuda_rt.new_stream().unwrap();
    
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * num_heads * head_dim * 4;
    
    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let s_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(s_size).unwrap() };
    
    // Zero-initialize output
    let zero_init = vec![0.0f32; seq_q * num_heads * head_dim];
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(
            s_ptr, zero_init.as_ptr() as *const u8, s_size).unwrap();
    }
    
    // Copy inputs
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
    }
    
    let kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(), stream.clone(),
    ).unwrap();
    
    unsafe {
        kernel.launch(scale, q_ptr as u64, k_ptr as u64, v_ptr as u64, s_ptr as u64,
            seq_q, seq_k, num_heads, head_dim, rope_base, seq_k).unwrap();
    }
    cuda_rt.synchronize().unwrap();
    
    // Read output
    let mut gpu_output = vec![0.0f32; seq_q * num_heads * head_dim];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_output.as_mut_ptr() as *mut u8, s_ptr as *const u8, s_size).unwrap();
    }
    
    println!("GPU Output: [{:.4}, {:.4}, {:.4}, {:.4}]", 
             gpu_output[0], gpu_output[1], gpu_output[2], gpu_output[3]);
    
    // Check for correctness - FIXED BUG: use expected_output[i] instead of out_i
    let mut max_err = 0.0;
    for i in 0..head_dim {
        let err = (gpu_output[i] - expected_output[i]).abs();
        if err > max_err { max_err = err; }
    }
    
    println!("\n=== Verification ===");
    println!("Max error: {:.6e}", max_err);
    
    if max_err < 1e-3 {
        println!("✅ PASS - Kernel produces correct output!");
    } else if max_err < 1e-1 {
        println!("⚠️ WARNING - Some numerical drift (but algorithm may be correct)");
    } else {
        println!("❌ FAIL - Kernel output is WRONG (algorithm bug!)");
        println!("   Expected: [2.0, 2.0, 2.0, 2.0]");
        println!("   Got:      [{:.4}, {:.4}, {:.4}, {:.4}]", 
                 gpu_output[0], gpu_output[1], gpu_output[2], gpu_output[3]);
    }
    
    // Check for NaN/Inf
    let has_nan = gpu_output.iter().any(|&x| x.is_nan());
    let has_inf = gpu_output.iter().any(|&x| x.is_infinite());
    
    if has_nan {
        println!("❌ FAIL - Output contains NaN!");
    } else if has_inf {
        println!("⚠️ WARNING - Output contains Inf!");
    } else {
        println!("✅ No NaN/Inf detected");
    }
    
    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(s_ptr).unwrap();
    }
}
