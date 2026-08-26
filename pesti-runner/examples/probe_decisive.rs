//! DECISIVE probe: sync race vs kernel bug.
//!
//! For k=1024 (m=1 n=1024), A=ones, B=identity => expected C[j]=1.0 for all j.
//! C is re-zeroed on-stream each iteration so a race reads zeros.
//!
//! Three wait modes, each a fresh process:
//!   stream : launch (internal stream-sync) + explicit stream.synchronize
//!   event  : launch + record event on S + cuEventSynchronize (ALWAYS blocks)
//!   ctx    : launch + cuCtxSynchronize
//!
//! Interpretation:
//!   event/ctx pass, stream fails  -> SYNC RACE (stream-sync insufficient)
//!   event ALSO fails              -> KERNEL BUG (kernel writes wrong values)
//! Bad-column dump + values disambiguate stale-zero (race) vs wrong (bug).
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::sys;
    use pesti_runner::cuda_runtime::CudaRuntime;
    use pesti_runner::cuda_shim::IntoResult;
    use pesti_runner::kernel::memory::{CudaMemoryBackend, MemoryBackend};
    use pesti_runner::kernel::{CudaGemmKernelBuilder, GemmArch};

    let mode = std::env::args().nth(1).unwrap_or_else(|| "event".into());
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    let rt = Arc::new(CudaRuntime::new(0)?);
    let info = rt.device_info().clone();
    let stream = rt.new_stream()?;
    let s = pesti_runner::cuda_shim::cu_stream(&stream);
    let backend = CudaMemoryBackend::with_device_info(stream.clone(), info.clone());
    let kernel = CudaGemmKernelBuilder::new(
        GemmArch::Mma,
        rt.context().clone(),
        stream.clone(),
        info.clone(),
    )
    .build()?;

    let (m, n, k) = (1usize, 1024usize, 1024usize);
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
    let a_bytes: &[u8] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 2) };
    let b_bytes: &[u8] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 2) };
    backend.h2d(a_bytes, a_handle)?;
    backend.h2d(b_bytes, b_handle)?;
    rt.context().synchronize().unwrap();
    let a_buf = pesti_runner::kernel::DeviceBuffer::<half::f16>::from_backend(a_handle, a.len());
    let b_buf = pesti_runner::kernel::DeviceBuffer::<half::f16>::from_backend(b_handle, b.len());
    let c_ptr = c_handle.as_ptr() as sys::CUdeviceptr;

    let mut bad_iters = 0usize;
    let mut shown = 0usize;
    for it in 0..iters {
        // Re-zero C on-stream each iter (expose any race deterministically).
        unsafe {
            sys::cuMemsetD32Async(c_ptr, 0, m * n, s).result().unwrap();
        }
        let mut c_buf = pesti_runner::kernel::DeviceBuffer::<f32>::from_backend(c_handle, m * n);
        kernel.launch(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
        // Wait per mode.
        unsafe {
            match mode.as_str() {
                "stream" => {
                    sys::cuStreamSynchronize(s).result().unwrap();
                }
                "event" => {
                    let mut ev = std::mem::zeroed();
                    sys::cuEventCreate(&mut ev, sys::CUevent_flags::CU_EVENT_DEFAULT as u32)
                        .result()
                        .unwrap();
                    sys::cuEventRecord(ev, s).result().unwrap();
                    sys::cuEventSynchronize(ev).result().unwrap();
                    sys::cuEventDestroy_v2(ev);
                }
                "ctx" => {
                    sys::cuCtxSynchronize().result().unwrap();
                }
                _ => unreachable!(),
            }
        }
        // d2h (synchronous) after the wait.
        let mut c_host = vec![0.0f32; m * n];
        let c_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(c_host.as_mut_ptr() as *mut u8, m * n * 4) };
        backend.d2h(c_handle, c_bytes)?;

        // Find bad columns. Expected C[j] = 1.0 for all j (since k==n==1024).
        let mut bad: Vec<(usize, f32)> = Vec::new();
        for j in 0..n {
            let v = c_host[j];
            if (v - 1.0).abs() > 1e-3 {
                bad.push((j, v));
            }
        }
        if !bad.is_empty() {
            bad_iters += 1;
            if shown < 5 {
                shown += 1;
                let min_j = bad.first().map(|(j, _)| *j).unwrap();
                let max_j = bad.last().map(|(j, _)| *j).unwrap();
                let all_zero = bad.iter().all(|(_, v)| *v == 0.0);
                let sample: Vec<String> = bad.iter().take(8).map(|(j, v)| format!("{}={v:.3}", j)).collect();
                eprintln!(
                    "  [{}] iter={it} bad={} range=[{min_j}..{max_j}] all_zero={} sample=[{}]",
                    mode, bad.len(), all_zero, sample.join(",")
                );
            }
        }
    }
    eprintln!(
        "mode={} k=1024: bad_iters={}/{}",
        mode, bad_iters, iters
    );
    Ok(())
}
