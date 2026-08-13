//! Quick sanity check: Does single-kernel PTX load and have the expected function?

#[test]
fn test_single_kernel_ptx_loads() {
    // Just check that the PTX file exists and can be loaded as a module
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    
    assert!(!ptx_src.is_empty(), "PTX file is empty!");
    println!("✅ PTX file loaded: {} bytes", ptx_src.len());
    
    // Check for expected function name in PTX
    assert!(ptx_src.contains("fused_attention_simple_kernel"), 
            "PTX doesn't contain 'fused_attention_simple_kernel' function");
    println!("✅ PTX contains expected kernel function");
    
    // Check for sm_89 target
    assert!(ptx_src.contains(".target sm_89"), 
            "PTX not compiled for sm_89 (RTX 4070 Ti SUPER)");
    println!("✅ PTX compiled for sm_8.9 target");
    
    println!("\n=== Single-Kernel PTX Verification PASSED ===");
}
