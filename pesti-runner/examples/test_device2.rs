//! Test CUDA with only device 1 (RTX 5060 Ti)

use cuda_core::IntoResult;

fn main() {
    println!("=== CUDA Device 1 Test (RTX 5060 Ti) ===\n");
    
    // Set CUDA_VISIBLE_DEVICES to only show device 1
    std::env::set_var("CUDA_VISIBLE_DEVICES", "1");
    
    // Re-import after env change - this forces cuda-core to re-read
    // Actually we need to init before setting the var, so let's try a different approach
    
    // First, check if we can use the default device
    unsafe {
        match cuda_core::init(0) {
            Ok(_) => println!("✅ Device 0: CUDA init OK"),
            Err(e) => println!("❌ Device 0: {}", e),
        }
    }
    
    // Get device count
    let mut count: i32 = 0;
    unsafe {
        match cuda_core::sys::cuDeviceGetCount(&mut count).result() {
            Ok(_) => println!("✅ Device count: {}", count),
            Err(e) => println!("❌ Device count: {}", e),
        }
    }
    
    // Try device 0
    if count > 0 {
        let mut cu_device = std::mem::MaybeUninit::uninit();
        match unsafe {
            cuda_core::sys::cuDeviceGet(cu_device.as_mut_ptr(), 0).result()
        } {
            Ok(_) => {},
            Err(e) => {
                println!("❌ Device 0 get failed: {}", e);
                return;
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
        
        let mut major = std::mem::MaybeUninit::uninit();
        let mut minor = std::mem::MaybeUninit::uninit();
        unsafe {
            cuda_core::sys::cuDeviceGetAttribute(
                major.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            ).result().unwrap();
            cuda_core::sys::cuDeviceGetAttribute(
                minor.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            ).result().unwrap();
        }
        let (major, minor) = unsafe { (major.assume_init(), minor.assume_init()) };
        
        println!("\nDevice 0: {} (sm_{}.{})", name, major, minor);
    }
    
    // Try device 1
    if count > 1 {
        let mut cu_device = std::mem::MaybeUninit::uninit();
        match unsafe {
            cuda_core::sys::cuDeviceGet(cu_device.as_mut_ptr(), 1).result()
        } {
            Ok(_) => {},
            Err(e) => {
                println!("❌ Device 1 get failed: {}", e);
                return;
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
        
        let mut major = std::mem::MaybeUninit::uninit();
        let mut minor = std::mem::MaybeUninit::uninit();
        unsafe {
            cuda_core::sys::cuDeviceGetAttribute(
                major.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            ).result().unwrap();
            cuda_core::sys::cuDeviceGetAttribute(
                minor.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            ).result().unwrap();
        }
        let (major, minor) = unsafe { (major.assume_init(), minor.assume_init()) };
        
        println!("\nDevice 1: {} (sm_{}.{})", name, major, minor);
    }
}
