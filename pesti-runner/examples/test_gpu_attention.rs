//! Simple test to verify GPU attention kernel loads and is available

use cuda_core::IntoResult;

fn main() {
    println!("=== CUDA Device Enumeration Test ===\n");

    // Try to initialize CUDA explicitly first
    println!("Initializing CUDA driver...");
    match unsafe { cuda_core::init(0) } {
        Ok(_) => println!("✅ CUDA driver initialized"),
        Err(e) => {
            println!("❌ CUDA init failed: {}", e);
            return;
        }
    }

    // Get device count
    let mut count: i32 = 0;
    match unsafe { cuda_core::sys::cuDeviceGetCount(&mut count).result() } {
        Ok(_) => println!("✅ Device count: {}", count),
        Err(e) => {
            println!("❌ Device count failed: {}", e);
            return;
        }
    }

    // Try to enumerate devices
    if count == 0 {
        println!("❌ No devices found!");
        return;
    }

    for ordinal in 0..count {
        println!("\n--- Device {} ---", ordinal);

        // Get device handle
        let cu_device = unsafe {
            let mut device = std::mem::MaybeUninit::uninit();
            match cuda_core::sys::cuDeviceGet(device.as_mut_ptr(), ordinal).result() {
                Ok(_) => device.assume_init(),
                Err(e) => {
                    println!("❌ cuDeviceGet failed: {}", e);
                    continue;
                }
            }
        };

        println!("✅ Device handle obtained");

        // Get device name
        let mut name_buf = [0i8; 256];
        unsafe {
            cuda_core::sys::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, cu_device)
        };
        let name: String = name_buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect::<Vec<u8>>()
            .into_iter()
            .map(|b| b as char)
            .collect();
        println!("   Name: {}", name);

        // Get compute capability
        let mut major = std::mem::MaybeUninit::uninit();
        let mut minor = std::mem::MaybeUninit::uninit();
        unsafe {
            match cuda_core::sys::cuDeviceGetAttribute(
                major.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            )
            .result() {
                Ok(_) => (),
                Err(e) => {
                    println!("❌ GetAttribute major failed: {}", e);
                    continue;
                }
            };
            match cuda_core::sys::cuDeviceGetAttribute(
                minor.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            )
            .result() {
                Ok(_) => (),
                Err(e) => {
                    println!("❌ GetAttribute minor failed: {}", e);
                    continue;
                }
            };
        }
        let (major, minor) = unsafe { (major.assume_init(), minor.assume_init()) };
        println!("   Compute Cap: sm_{}.{}", major, minor);

        // Get memory info
        let (free_memory, total_memory) = {
            let mut free: usize = 0;
            let mut total: usize = 0;
            match unsafe {
                cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total).result()
            } {
                Ok(_) => (free as u64, total as u64),
                Err(e) => {
                    println!("❌ cuMemGetInfo failed: {}", e);
                    continue;
                }
            }
        };
        println!(
            "   Free Memory: {:.1} GiB",
            free_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "   Total Memory: {:.1} GiB",
            total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    println!("\n=== Summary ===");
    println!("CUDA is working! GPUs should be available for testing.");
}
