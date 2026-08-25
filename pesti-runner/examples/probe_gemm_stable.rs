//! Decisive probe: stable buffers, repeated launches, no alloc/free churn.
//!
//! Separates three hypotheses for the k=896 all-zero / k=16,32 partial
//! failures:
//!   H1 kernel bug  -> same buffers, repeated launches, STILL wrong (deterministic)
//!   H2 stream race -> sync-context (device-wide) fixes it, sync-stream doesn't
//!   H3 alloc/free  -> stable buffers fix it (dispatch path churn is the cause)
//!
//! Allocates A, B, C ONCE on the engine's stream, H2D once, then loops:
//!   launch -> sync -> d2h -> check
//! with a sync-mode toggle (stream vs context).
use std::sync::Arc;

use half::f16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a DispatchContext just to get a working engine + memory backend
    // on the SAME stream (mirrors the real dispatch path's topology).
    let ctx = pesti_runner::kernel::DispatchContext::new();
    eprintln!(
        "gpu={} arch={}",
        ctx.gpu_available(),
        ctx.gemm_arch().name()
    );

    // We need direct access to the engine + memory. DispatchContext doesn't
    // expose them, so rebuild the same topology here explicitly.
    use pesti_runner::cuda_runtime::CudaRuntime;
    use pesti_runner::kernel::memory::{CudaMemoryBackend, MemoryBackend};
    use pesti_runner::kernel::{CudaGemmKernelBuilder, GemmArch};

    let rt = Arc::new(CudaRuntime::new(0)?);
    let info = rt.device_info().clone();
    eprintln!(
        "device={} sm={}.{}",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    let stream = rt.new_stream()?;
    let backend = CudaMemoryBackend::with_device_info(stream.clone(), info.clone());

    let kernel = CudaGemmKernelBuilder::new(
        GemmArch::Mma,
        rt.context().clone(),
        stream.clone(),
        info.clone(),
    )
    .build()?;

    // Test shapes: (m, k, n). A=ones[m x k], B=identity[k x n].
    // Expected C[i][j] = 1.0 if j < k else 0.0.
    let shapes: [(usize, usize, usize); 4] = [
        (1, 16, 64),
        (1, 32, 64),
        (1, 896, 96),
        (1, 1024, 1024),
    ];

    for (m, k, n) in shapes {
        // Allocate ONCE.
        let a: Vec<f16> = vec![f16::from_f32(1.0); m * k];
        let b: Vec<f16> = (0..k * n)
            .map(|i| {
                let kk = i / n;
                let j = i % n;
                f16::from_f32(if j == kk { 1.0 } else { 0.0 })
            })
            .collect();

        let a_handle = backend.alloc(a.len() * 2)?;
        let b_handle = backend.alloc(b.len() * 2)?;
        let c_handle = backend.alloc(m * n * 4)?;

        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 2) };
        let b_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 2) };
        backend.h2d(a_bytes, a_handle)?;
        backend.h2d(b_bytes, b_handle)?;
        backend.sync()?;

        let a_buf = pesti_runner::kernel::DeviceBuffer::<f16>::from_backend(a_handle, a.len());
        let b_buf = pesti_runner::kernel::DeviceBuffer::<f16>::from_backend(b_handle, b.len());

        let iters = 20usize;
        for mode in ["stream", "context"] {
            let mut correct = 0usize;
            let mut all_zero = 0usize;
            let mut partial = 0usize;
            let mut shown = 0usize;
            for _ in 0..iters {
                let mut c_buf =
                    pesti_runner::kernel::DeviceBuffer::<f32>::from_backend(c_handle, m * n);
                // Re-zero C each iter (kernel uses beta=0 so it overwrites, but
                // be explicit to avoid reading stale data on a no-op).
                kernel.launch(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
                // Sync per mode.
                if mode == "context" {
                    rt.synchronize()?;
                } else {
                    stream.synchronize()?;
                }
                let mut c_host = vec![0.0f32; m * n];
                let c_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(c_host.as_mut_ptr() as *mut u8, m * n * 4)
                };
                backend.d2h(c_handle, c_bytes)?;

                let mut bad: Vec<usize> = Vec::new();
                let mut az = true;
                for i in 0..m {
                    for j in 0..n {
                        let exp = if j < k { 1.0f32 } else { 0.0 };
                        if (c_host[i * n + j] - exp).abs() > 1e-3 {
                            bad.push(j);
                        }
                        if c_host[i * n + j] != 0.0 {
                            az = false;
                        }
                    }
                }
                if az {
                    all_zero += 1;
                    if shown < 2 {
                        eprintln!("  [{mode}] m={m} k={k} n={n} ALL-ZERO");
                        shown += 1;
                    }
                } else if bad.is_empty() {
                    correct += 1;
                } else {
                    partial += 1;
                    if shown < 4 {
                        let first: Vec<usize> = bad.iter().take(16).cloned().collect();
                        eprintln!(
                            "  [{mode}] m={m} k={k} n={n} PARTIAL bad={} first={first:?}",
                            bad.len()
                        );
                        shown += 1;
                    }
                }
            }
            eprintln!(
                "m={m} k={k} n={n} sync={mode}: correct={correct} all_zero={all_zero} partial={partial} (of {iters})"
            );
        }

        // Free once.
        let _ = backend.free(a_handle);
        let _ = backend.free(b_handle);
        let _ = backend.free(c_handle);
    }

    Ok(())
}
