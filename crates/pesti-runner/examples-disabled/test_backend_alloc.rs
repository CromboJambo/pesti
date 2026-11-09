use std::sync::Arc;
use cudarc::driver::safe::{CudaContext, CudaStream};
use pesti_runner::kernel::memory::CudaMemoryBackend;
use pesti_runner::kernel::device_buf::DeviceBuffer;

fn main() {
    // Get CUDA context and stream (same as benchmark does)
    let ctx = Arc::new(CudaContext::new(0).expect("Failed to get context"));
    let stream = Arc::new(ctx.create_stream(cudarc::driver::stream_flags()).expect("Failed to create stream"));

    println!("CUDA context and stream created");

    // Test 1: Create backend and check total_memory before try_init_device_info
    let mut backend = CudaMemoryBackend::new(stream.clone());
    println!("Before try_init_device_info: total_memory={}", backend.total_memory());

    // Call try_init_device_info
    let info = backend.try_init_device_info();
    println!("After try_init_device_info: total_memory={}, device_name={}", 
             backend.total_memory(), info.device_name);

    // Test 2: Try to create a small buffer via from_host_device
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    match DeviceBuffer::from_host_device(&backend, &data) {
        Ok(buf) => {
            println!("Successfully created buffer at ptr={:x}", buf.device_ptr());
            let back = buf.to_host_vec(&backend).expect("Failed to copy back");
            println!("Roundtrip: {:?}", back);
        }
        Err(e) => {
            println!("Failed to create buffer: {}", e);
        }
    }

    // Test 3: Try zeros_device
    match DeviceBuffer::zeros_device(&backend, 1024) {
        Ok(buf) => {
            println!("Successfully created zeroed buffer at ptr={:x}", buf.device_ptr());
        }
        Err(e) => {
            println!("Failed to create zeroed buffer: {}", e);
        }
    }
}
