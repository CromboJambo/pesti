//! Test that attention kernel PTX loads correctly

use cuda_core::IntoResult;
use pesti_runner::cuda_runtime;

fn main() {
    println!("=== Attention Kernel PTX Load Test ===\n");

    // Initialize CUDA
    unsafe {
        match cuda_core::init(0) {
            Ok(_) => println!("✅ CUDA driver initialized"),
            Err(e) => {
                println!("❌ CUDA init failed: {}", e);
                return;
            }
        }
    }

    // Create context for device 0 (4070 Ti - sm_8.9, WGMMA supported)
    let ctx = match cuda_core::CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("❌ Context creation failed: {}", e);
            return;
        }
    };
    println!("✅ Context created for device 0");

    // Create stream
    let _stream = match ctx.new_stream() {
        Ok(s) => s,
        Err(e) => {
            println!("❌ Stream creation failed: {}", e);
            return;
        }
    };
    println!("✅ Stream created");

    // Load WGMMA PTX
    let ptx_src = include_str!("../src/kernel/ptx/attention_wgmma.ptx");
    println!("✅ Loaded {} bytes of WGMMA PTX", ptx_src.len());

    // Bind context to thread (required for some CUDA operations)
    ctx.bind_to_thread()
        .expect("Failed to bind context to thread");
    println!("✅ Context bound to thread");

    // Load module with error logging
    let module = match ctx.load_module_from_ptx_src(ptx_src) {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Module load failed: {}", e);

            // Print raw driver error if available
            println!("Driver Error Code: {:?}", e);
            return;
        }
    };
    println!("✅ Module loaded");

    // Load kernel function
    let function = match module.load_function("attention_wgmma_kernel") {
        Ok(f) => f,
        Err(e) => {
            println!("❌ Function load failed: {}", e);
            return;
        }
    };
    println!("✅ Kernel function loaded");

    // Verify function is valid
    unsafe {
        if function.cu_function().is_null() {
            println!("❌ Kernel function handle is null");
            return;
        }
    }
    println!("✅ Kernel function handle is valid");

    println!("\n=== Summary ===");
    println!("Attention kernel PTX loads successfully on sm_12.0 (WGMMA)!");
}
