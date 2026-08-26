//! FINAL diagnostic: pin down the exact mechanism + return codes.
//!
//! The prior diag showed cuStreamSynchronize is a no-op (0.2us) and async-d2h
//! passes WITH warmup but fails WITHOUT. This probe:
//!   - checks the actual CUresult of each sync call (is it silently erroring?)
//!   - tests warmup on/off
//!   - tests the candidate fixes with correct blocking waits
//!
//! Usage: probe_final <mode> [warmup]
//!   sync_d2h        : sync D2H (legacy stream)              -> expect FAIL
//!   async_ss        : async D2H on S + cuStreamSynchronize  -> prints ret code
//!   async_es        : async D2H on S + cuEventSynchronize   -> expect PASS
//!   blocking_async  : BLOCKING_SYNC + async D2H + cuStreamSynchronize
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::sys;
    use pesti_runner::cuda_runtime::CudaRuntime;
    use pesti_runner::cuda_shim::IntoResult;
    use pesti_runner::kernel::memory::{CudaMemoryBackend, MemoryBackend};
    use pesti_runner::kernel::{CudaGemmKernelBuilder, GemmArch};

    let mode = std::env::args().nth(1).unwrap_or_else(|| "sync_d2h".into());
    let warmup: bool = std::env::args()
        .nth(2)
        .map(|s| s == "1")
        .unwrap_or(false);

    let rt = Arc::new(CudaRuntime::new(0)?);
    let info = rt.device_info().clone();
    let stream = rt.new_stream()?;
    let s = pesti_runner::cuda_shim::cu_stream(&stream);

    if mode == "blocking_async" {
        rt.context().set_blocking_synchronize().unwrap();
    }

    let backend = CudaMemoryBackend::with_device_info(stream.clone(), info.clone());
    let kernel = CudaGemmKernelBuilder::new(
        GemmArch::Mma,
        rt.context().clone(),
        stream.clone(),
        info.clone(),
    )
    .build()?;

    let (m, k, n) = (1usize, 1024usize, 1024usize);
    let a: Vec<half::f16> = vec![half::f16::from_f32(1.0); m * k];
    let b: Vec<half::f16> = (0..k * n)
        .map(|i| {
            let kk = i / n;
            let j = i % n;
            half::f16::from_f32(if j == kk { 1.0 } else { 0.0 })
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
    let a_buf = pesti_runner::kernel::DeviceBuffer::<half::f16>::from_backend(a_handle, a.len());
    let b_buf = pesti_runner::kernel::DeviceBuffer::<half::f16>::from_backend(b_handle, b.len());
    let c_ptr = c_handle.as_ptr() as sys::CUdeviceptr;

    if warmup {
        for _ in 0..5 {
            unsafe {
                sys::cuMemsetD32Async(c_ptr, 0, m * n, s).result().unwrap();
            }
            let mut c_buf =
                pesti_runner::kernel::DeviceBuffer::<f32>::from_backend(c_handle, m * n);
            kernel.launch(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
            rt.context().synchronize().unwrap();
        }
    }

    let iters = 60usize;
    let mut partial = 0usize;
    let mut ss_ret: i32 = 0; // last cuStreamSynchronize CUresult
    let mut es_ret: i32 = 0; // last cuEventSynchronize CUresult
    let mut ss_nanos: u128 = 0;

    for _ in 0..iters {
        unsafe {
            sys::cuMemsetD32Async(c_ptr, 0, m * n, s).result().unwrap();
        }
        let mut c_buf = pesti_runner::kernel::DeviceBuffer::<f32>::from_backend(c_handle, m * n);
        kernel.launch(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;

        let mut c_host = vec![0.0f32; m * n];
        let t0 = Instant::now();
        match mode.as_str() {
            "sync_d2h" => {
                let c_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(c_host.as_mut_ptr() as *mut u8, m * n * 4)
                };
                backend.d2h(c_handle, c_bytes)?;
            }
            "async_ss" => unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    c_host.as_mut_ptr() as *mut std::ffi::c_void,
                    c_ptr,
                    m * n * 4,
                    s,
                )
                .result()
                .unwrap();
                let r = sys::cuStreamSynchronize(s);
                ss_ret = r as i32;
                r.result().unwrap();
            }
            "async_es" => unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    c_host.as_mut_ptr() as *mut std::ffi::c_void,
                    c_ptr,
                    m * n * 4,
                    s,
                )
                .result()
                .unwrap();
                let mut ev = std::mem::zeroed();
                sys::cuEventCreate(&mut ev, sys::CUevent_flags::CU_EVENT_DEFAULT as u32)
                    .result()
                    .unwrap();
                sys::cuEventRecord(ev, s).result().unwrap();
                let r = sys::cuEventSynchronize(ev);
                es_ret = r as i32;
                r.result().unwrap();
                sys::cuEventDestroy_v2(ev);
            }
            "blocking_async" => unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    c_host.as_mut_ptr() as *mut std::ffi::c_void,
                    c_ptr,
                    m * n * 4,
                    s,
                )
                .result()
                .unwrap();
                let r = sys::cuStreamSynchronize(s);
                ss_ret = r as i32;
                r.result().unwrap();
            }
            other => panic!("unknown mode {other}"),
        }
        ss_nanos += t0.elapsed().as_nanos();

        let bad = c_host.iter().filter(|&&v| (v - 1.0).abs() > 1e-3).count();
        if bad > 0 {
            partial += 1;
        }
    }
    eprintln!(
        "{mode} warmup={warmup}: partial={partial}/{iters} avg_wait={:.1}us ss_ret={ss_ret} es_ret={es_ret}",
        ss_nanos as f64 / iters as f64 / 1000.0
    );
    Ok(())
}
