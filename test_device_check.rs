use pesti_runner::device::DeviceSelector;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Attempt to initialize and enumerate local CUDA devices.
    // This implicitly tests the cuda-oxide integration and device detection logic.
    let selector = DeviceSelector::new();
    println!("--- PESTI Device Check ---");
    match selector.list_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("[INFO] No CUDA/Compute devices found by the system.");
            } else {
                println!("[SUCCESS] Detected {} device(s):", devices.len());
                for (i, device) in devices.iter().enumerate() {
                    println!("  [{}] Name: {:?}, Capabilities: {:?}", 
                        i, 
                        device.name(), 
                        device.capabilities()
                    );
                }
            }
        }
        Err(e) => {
             // If this fails (e.g., missing CUDA runtime), it proves the path is blocked.
            eprintln!("[ERROR] Failed to initialize DeviceSelector: {:?}", e);
            return Err(e);
        }
    }
    println!("--------------------------");
    Ok(())
}