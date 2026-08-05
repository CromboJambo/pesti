//! Stub types for CUDA dependencies when building without the `cuda` feature.
//!
//! These allow the code to compile on CPU-only systems or when GPU support is disabled.

#[cfg(not(feature = "cuda"))]
pub mod stub {
    use std::sync::Arc;

    /// Dummy CudaContext for CPU builds
    #[derive(Clone)]
    pub struct CudaContext;

    /// Dummy CudaStream for CPU builds
    #[derive(Clone)]
    pub struct CudaStream;

    /// Dummy CudaModule for CPU builds
    #[derive(Clone)]
    pub struct CudaModule;

    /// Dummy CudaFunction for CPU builds
    pub type CudaFunction = ();

    /// Dummy sys module with types
    pub mod sys {
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy)]
        pub struct CUdeviceptr(pub u64);

        impl CUdeviceptr {
            pub fn as_u64(self) -> u64 {
                self.0
            }
        }
    }

    /// Dummy memory module
    pub mod memory {
        use super::sys::CUdeviceptr;

        pub unsafe fn malloc_async(_stream: u64, _bytes: usize) -> Result<CUdeviceptr, ()> {
            Err(())
        }

        pub unsafe fn free_async(_ptr: CUdeviceptr, _stream: u64) -> Result<(), ()> {
            Ok(())
        }

        pub unsafe fn memcpy_htod_async<T>(
            _dst: CUdeviceptr,
            _src: *const T,
            _bytes: usize,
            _stream: u64,
        ) -> Result<(), ()> {
            Ok(())
        }

        pub unsafe fn memcpy_dtoh_async<T>(
            _dst: *mut T,
            _src: CUdeviceptr,
            _bytes: usize,
            _stream: u64,
        ) -> Result<(), ()> {
            Ok(())
        }

        pub unsafe fn memcpy_dtod_async(
            _dst: CUdeviceptr,
            _src: CUdeviceptr,
            _bytes: usize,
            _stream: u64,
        ) -> Result<(), ()> {
            Ok(())
        }
    }

    /// Dummy device buffer
    pub struct DeviceBuffer<T> {
        _marker: std::marker::PhantomData<T>,
    }

    impl<T> DeviceBuffer<T> {
        pub fn from_raw_parts(_ptr: *mut u8, _len: usize) -> Self {
            Self {
                _marker: std::marker::PhantomData,
            }
        }

        pub fn from_host(_data: &[T]) -> Self {
            Self {
                _marker: std::marker::PhantomData,
            }
        }

        pub fn device_ptr(&self) -> Option<u64> {
            Some(0)
        }
    }

    /// Dummy launch function
    pub fn launch_kernel(
        _function: (),
        _grid_dims: &[i32],
        _param_ptrs: *const *mut std::ffi::c_void,
        _stream: u64,
    ) -> Result<(), ()> {
        Ok(())
    }

    pub fn launch_kernel_on_stream(
        _function: (),
        _grid_dims: &[i32],
        _param_ptrs: *const *mut std::ffi::c_void,
        _stream: u64,
    ) -> Result<(), ()> {
        Ok(())
    }

    /// Dummy init function
    pub fn init(_device: u32) -> Result<(), ()> {
        Err(())
    }
}
