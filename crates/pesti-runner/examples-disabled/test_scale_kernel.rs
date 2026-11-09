fn main() {
    let rt = pesti_runner::CudaRuntime::new(0).expect("Failed to create CUDA runtime");
    let stream = rt.new_stream().expect("Failed to create CUDA stream");

    // Simple test: scale input by 2
    let n = 4;
    let x_host = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let x_buf = pesti_runner::kernel::device_buf::DeviceBuffer::from_host(x_host.clone());

    let mut y_buf: pesti_runner::kernel::device_buf::DeviceBuffer<f32> =
        pesti_runner::kernel::device_buf::DeviceBuffer::zeros(n);

    // Load scale kernel
    let scale_kernel = pesti_runner::kernel::scale_test::ScaleKernel::load(
        rt.context().clone(),
        stream.clone(),
    )
    .expect("Failed to load scale kernel");

    println!("Scale kernel loaded successfully");

    // Launch kernel
    let result = scale_kernel.launch(x_buf.device_ptr(), y_buf.device_ptr(), n);

    match result {
        Ok(_) => {
            rt.synchronize().expect("Failed to sync");
            println!("Kernel launched successfully");
            let y_host = y_buf.to_host();
            println!("Input: {:?}", x_host);
            println!("Output: {:?}", y_host);
            // Expected: [2.0, 4.0, 6.0, 8.0]
        }
        Err(e) => {
            println!("Kernel launch failed: {}", e);
        }
    }
}