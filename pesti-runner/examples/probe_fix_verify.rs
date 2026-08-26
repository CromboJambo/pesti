//! Fix-verification probe for the mma.sync k=1024 D2H race.
//!
//! Baseline (probe_sync_mechanism) proved: stream-sync fails ~1/60 (tail cols
//! 829-1023), context-sync passes 60/60, when C is re-zeroed on-stream each
//! iter. This probe tests the two candidate FIXES under the same control:
//!
//!   S1  baseline: default ctx, kernel on S, sync S, sync-D2H (legacy stream)
//!       -> expect ~1/60 partial (reproduces the bug)
//!   S2  flag fix: set CU_CTX_SCHED_BLOCKING_SYNC, then sync S, sync-D2H
//!       -> if 60/60, the SCHED flag is the root cause and setting it fixes it
//!   S3  async fix: default ctx, sync-D2H replaced by cuMemcpyDtoHAsync_v2 on
//!       S (same stream as kernel) + cuStreamSynchronize(S)
//!       -> if 60/60, stream-ordered async copy is the idiomatic fix
//!   S4  async+ctx: like S3 but cuCtxSynchronize at the end
//!       -> control; should be 60/60
//!
//! The winner is the first of S2/S3/S4 that is 60/60. Prefer S3 (no global
//! context flag change) if it passes.
use std::sync::Arc;

use half::f16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::sys;
    use pesti_runner::cuda_runtime::CudaRuntime;
    use pesti_runner::cuda_shim::IntoResult;
    use pesti_runner::kernel::memory::{CudaMemoryBackend, MemoryBackend};
    use pesti_runner::kernel::{CudaGemmKernelBuilder, GemmArch};

    let rt = Arc::new(CudaRuntime::new(0)?);
    let info = rt.device_info().clone();
    eprintln!(
        "device={} sm={}.{}",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    let stream = rt.new_stream()?;
    let s_handle = pesti_runner::cuda_shim::cu_stream(&stream);
    let backend = CudaMemoryBackend::with_device_info(stream.clone(), info.clone());

    let kernel = CudaGemmKernelBuilder::new(
        GemmArch::Mma,
        rt.context().clone(),
        stream.clone(),
        info.clone(),
    )
    .build()?;

    let (m, k, n) = (1usize, 1024usize, 1024usize);
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
    let c_ptr = c_handle.as_ptr() as sys::CUdeviceptr;

    // Re-zero C on the stream (exposes any D2H race deterministically).
    fn rezero(c_ptr: sys::CUdeviceptr, count: usize, s: sys::CUstream) {
        unsafe {
            sys::cuMemsetD32Async(c_ptr, 0, count, s)
                .result()
                .unwrap();
        }
    }

    // Count bad columns after a D2H readback.
    fn count_bad(
        backend: &CudaMemoryBackend,
        c_handle: pesti_runner::kernel::memory::RawHandle,
        m: usize,
        n: usize,
    ) -> usize {
        let mut c_host = vec![0.0f32; m * n];
        let c_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(c_host.as_mut_ptr() as *mut u8, m * n * 4) };
        backend.d2h(c_handle, c_bytes).unwrap();
        c_host
            .iter()
            .enumerate()
            .filter(|e| {
                let (idx, v) = e;
                let j = idx % n;
                let exp = if j < n { 1.0f32 } else { 0.0 };
                // expected is 1.0 for all j<k; here k==n so all 1.0
                (*v - exp).abs() > 1e-3
            })
            .count()
    }

    let iters = 60usize;

    // ---- S1 baseline: sync-D2H (legacy default stream) after sync S ----
    {
        let mut partial = 0;
        for _ in 0..iters {
            rezero(c_ptr, m * n, s_handle);
            let mut c_buf =
                pesti_runner::kernel::DeviceBuffer::<f32>::from_backend(c_handle, m * n);
            kernel.launch(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
            unsafe { sys::cuStreamSynchronize(s_handle).result().unwrap(); }
            if count_bad(&backend, c_handle, m, n) > 0 {
                partial += 1;
            }
        }
        eprintln!("S1 baseline(sync-d2h, sync S): partial={partial}/{iters}");
    }

    // ---- S2 flag fix: BLOCKING_SYNC, then sync S, sync-D2H ----
    {
        rt.context()
            .set_blocking_synchronize()
            .unwrap();
        eprintln!("  (set CU_CTX_SCHED_BLOCKING_SYNC)");
        let mut partial = 0;
        for _ in 0..iters {
            rezero(c_ptr, m * n, s_handle);
            let mut c_buf =
                pesti_runner::kernel::DeviceBuffer::<f32>::from_backend(c_handle, m * n);
            kernel.launch(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
            unsafe { sys::cuStreamSynchronize(s_handle).result().unwrap(); }
            if count_bad(&backend, c_handle, m, n) > 0 {
                partial += 1;
            }
        }
        eprintln!("S2 flag-fix(BLOCKING_SYNC, sync S): partial={partial}/{iters}");
    }

    // ---- S3 async fix: stream-ordered async D2H on S + sync S ----
    {
        let mut partial = 0;
        for _ in 0..iters {
            rezero(c_ptr, m * n, s_handle);
            let mut c_buf =
                pesti_runner::kernel::DeviceBuffer::<f32>::from_backend(c_handle, m * n);
            kernel.launch(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
            // async D2H on the SAME stream as the kernel -> stream-ordered
            let mut c_host = vec![0.0f32; m * n];
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    c_host.as_mut_ptr() as *mut std::ffi::c_void,
                    c_ptr,
                    m * n * 4,
                    s_handle,
                )
                .result()
                .unwrap();
                sys::cuStreamSynchronize(s_handle).result().unwrap();
            }
            let bad = c_host
                .iter()
                .filter(|&&v| (v - 1.0).abs() > 1e-3)
                .count();
            if bad > 0 {
                partial += 1;
            }
        }
        eprintln!("S3 async-fix(d2h on S, sync S): partial={partial}/{iters}");
    }

    // ---- S4 async + ctx sync (control) ----
    {
        let mut partial = 0;
        for _ in 0..iters {
            rezero(c_ptr, m * n, s_handle);
            let mut c_buf =
                pesti_runner::kernel::DeviceBuffer::<f32>::from_backend(c_handle, m * n);
            kernel.launch(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
            let mut c_host = vec![0.0f32; m * n];
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    c_host.as_mut_ptr() as *mut std::ffi::c_void,
                    c_ptr,
                    m * n * 4,
                    s_handle,
                )
                .result()
                .unwrap();
                rt.synchronize().unwrap();
            }
            let bad = c_host
                .iter()
                .filter(|&&v| (v - 1.0).abs() > 1e-3)
                .count();
            if bad > 0 {
                partial += 1;
            }
        }
        eprintln!("S4 async+ctx(d2h on S, ctx sync): partial={partial}/{iters}");
    }

    let _ = backend.free(a_handle);
    let _ = backend.free(b_handle);
    let _ = backend.free(c_handle);
    Ok(())
}
