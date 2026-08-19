use pesti_runner::inference_engine::InferenceEngine;
use candle_core::{Device, DType};
use pesti_runner::cuda_runtime::{is_available, CudaRuntime};

fn main() {
    println!("is_available: {}", is_available());
    match CudaRuntime::new(0) {
        Ok(rt) => {
            let info = rt.device_info();
            println!(
                "RT OK: {} cc={:?} total={}MB",
                info.name,
                info.compute_capability,
                info.total_memory / 1024 / 1024
            );
            match rt.new_stream() {
                Ok(_) => println!("stream OK"),
                Err(e) => println!("stream FAIL: {:?}", e),
            }
        }
        Err(e) => println!("RT FAIL: {:?}", e),
    }

    // New behavior: engine inits CUDA on is_available(), device arg is just a hint.
    let eng = InferenceEngine::new(Device::Cpu, DType::F32);
    println!("gpu_available: {}", eng.gpu_available());
    println!("backend: {}", eng.backend_description());
    if let Some(stream) = eng.cuda_stream() {
        println!("cuda_stream: Some({:?})", stream);
    } else {
        println!("cuda_stream: None");
    }
    if let Some(info) = eng.cuda_device_info() {
        println!("cuda_device_info: {} cc={:?}", info.name, info.compute_capability);
    } else {
        println!("cuda_device_info: None");
    }
}
