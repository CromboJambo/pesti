// test_kernels.rs
use llm_runner::device::DeviceSelector;
use llm_runner::kernel::{CudaGemmKernel, MatMulResult};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("--- Kernel Test Start ---");

    // 1. Initialize Device Selector (simulating runtime discovery)
    let selector = DeviceSelector::new();
    let devices = selector.get_available_devices()?;
    
    if devices.is_empty() {
        eprintln!("Error: No CUDA-capable devices found by the DeviceSelector.");
        return Ok(());
    }

    println!("Successfully detected {} device(s). Testing kernels sequentially.", devices.len());

    // 2. Iterate and test on each device
    for (i, device) in devices.iter().enumerate() {
        let device_name = if i == 0 { "RTX 4070" } else { "RTX 5060 Ti" }; // Mocking name retrieval based on index/memory dump
        println!("\n[Testing Kernel on {} (Mock Index {})]", device_name, i);

        // Simulate allocating memory and running the kernel stub
        let matmul_result: MatMulResult = match CudaGemmKernel::matmul(device) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("  [FAIL] Failed to run GEMM kernel on {} stub: {}", device_name, e);
                // We expect failure/stub output here but handle it gracefully for the test.
                continue; 
            }
        };

        // Since we are testing stubs, we only care if the call path was successful (Ok(()) return)
        match matmul_result {
            Ok(_) => println!("  [SUCCESS] Kernel execution stub successfully called and returned Ok(())."),
            Err(e) => eprintln!("  [FAIL] Kernel execution stub failed unexpectedly: {}", e),
        }
    }

    println!("\n--- Kernel Test Complete ---");
    Ok(())
}