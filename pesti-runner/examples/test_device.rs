//! Minimal CUDA test for device 1 (RTX 5060 Ti)

use cuda_core::IntoResult;

fn main() {
    println!("=== CUDA Device Test ===\n");
    
    // Initialize CUDA driver
    unsafe {
        match cuda_core::init(0) {
            Ok(_) => println!("✅ Device 0 (4070 Ti): CUDA init OK"),
            Err(e) => println!("❌ Device 0: {}", e),
        }
    }
    
    unsafe {
        match cuda_core::init(1) {
            Ok(_) => println!("✅ Device 1 (5060 Ti): CUDA init OK"),
            Err(e) => println!("❌ Device 1: {}", e),
        }
    }
    
    // Check device count
    let mut count: i32 = 0;
    unsafe {
        match cuda_core::sys::cuDeviceGetCount(&mut count).result() {
            Ok(_) => println!("✅ Device count: {}", count),
            Err(e) => println!("❌ Device count: {}", e),
        }
    }
    
    // Try to get device info for both devices
    for i in 0..count {
        let mut cu_device = std::mem::MaybeUninit::uninit();
        match unsafe {
            cuda_core::sys::cuDeviceGet(cu_device.as_mut_ptr(), i).result()
        } {
            Ok(_) => {},
            Err(e) => {
                println!("❌ Device {}: {}", i, e);
                continue;
            }
        }
        
        let cu_device = unsafe { cu_device.assume_init() };
        
        let mut name_buf = [0i8; 256];
        unsafe {
            cuda_core::sys::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, cu_device);
        }
        let name: String = name_buf.iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect::<Vec<u8>>()
            .into_iter()
            .map(|b| b as char)
            .collect();
        
        println!("\nDevice {}:", i);
        println!("  Name: {}", name);
    }
}
