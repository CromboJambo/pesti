//! One-stage full fusion attention kernel integration into AttentionDispatch
//!
//! This module provides a drop-in replacement for the SDPA path that uses
//! our custom one-stage kernel instead of candle_bridge::sdpa.

use crate::kernel::{device_buf::DeviceBuffer, kvcache::Kvcache};
use half::f16;
use std::sync::Arc;

/// Configuration for one-stage full fusion attention
pub struct OneStageAttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub scale: f32,
}

impl OneStageAttentionConfig {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq: 4096,
            scale,
        }
    }
}

/// One-stage full fusion attention kernel using our custom PTX
pub struct OneStageAttentionKernel {
    config: OneStageAttentionConfig,
    #[allow(dead_code)]
    backend: Arc<crate::kernel::memory::CudaMemoryBackend>,
}

impl OneStageAttentionKernel {
    pub fn new(
        config: OneStageAttentionConfig,
        backend: Arc<crate::kernel::memory::CudaMemoryBackend>,
    ) -> Self {
        Self { config, backend }
    }

    /// Forward pass using one-stage full fusion kernel
    pub fn forward(
        &self,
        q: &[f32],
        k_cache: &Kvcache,
        v_cache: &Kvcache,
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let cache_len = start_pos + seq_len;

        // Convert Q to f16 (batch_size x seq_len x num_heads x head_dim)
        let q_f16: Vec<f16> = q.iter().map(|&x| f16::from_f32(x)).collect();

        // Allocate device buffers
        let q_size = batch_size * seq_len * num_heads * head_dim * 2; // f16
        let k_size = cache_len * num_heads * head_dim * 2; // f16 (K is already in cache)
        let v_size = cache_len * num_heads * head_dim * 2; // f16 (V is already in cache)
        let output_size = batch_size * seq_len * num_heads * head_dim * 4; // f32

        let q_ptr = unsafe { crate::cuda_runtime::allocate_device_memory(q_size)? };
        let out_ptr = unsafe { crate::cuda_runtime::allocate_device_memory(output_size)? };

        // Copy Q to device
        unsafe {
            crate::cuda_runtime::copy_host_to_device(q_ptr, q_f16.as_ptr() as *const u8, q_size)?;
        }

        // Get K/V pointers from cache
        let k_ptr = k_cache.device_ptr().unwrap();
        let v_ptr = v_cache.device_ptr().unwrap();

        // Launch kernel - use include_str! for PTX content
        let ptx_src = include_str!("../../src/kernel/ptx/fused_attention_full_kernel.ptx");

        // Get CUDA context from memory backend (using a runtime instance)
        let cuda_rt = crate::cuda_runtime::CudaRuntime::new(0)?;
        let module = crate::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src)?;

        // Parameters for kernel launch (10 params)
        let mut q_v: u64 = q_ptr as u64;
        let mut k_v: u64 = k_ptr;
        let mut v_v: u64 = v_ptr;
        let mut out_v: u64 = out_ptr as u64;
        let mut seq_q_v: u32 = (batch_size * seq_len) as u32;
        let mut seq_k_v: u32 = cache_len as u32;
        let mut num_heads_v: u32 = num_heads as u32;
        let mut head_dim_v: u32 = head_dim as u32;

        // Launch with grid (seq_q, seq_k, num_heads), block (head_dim, 1, 1)
        let grid = (
            (batch_size * seq_len) as u32,
            cache_len as u32,
            num_heads as u32,
        );
        let block = (head_dim as u32, 1u32, 1u32);

        let mut params: [*mut std::ffi::c_void; 10] = [
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut out_v as *mut u64 as *mut std::ffi::c_void,
            &mut (self.config.scale as f32) as *mut f32 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
            &mut 0u64 as *mut u64 as *mut std::ffi::c_void,
        ];

        unsafe {
            let stream = cuda_rt.new_stream()?;
            let cu_stream = crate::cuda_shim::cu_stream(&stream);
            crate::cuda_shim::launch_kernel(
                module
                    .load_function("_Z27fused_attention_full_kernelPK6__halfS1_S1_Pfiiii")?
                    .cu_function(),
                grid,
                block,
                0,
                cu_stream,
                &mut params,
            )?;
        }

        // Read back output
        let mut gpu_output: Vec<f32> = vec![0.0; output_size / 4];
        unsafe {
            crate::cuda_runtime::copy_device_to_host(
                gpu_output.as_mut_ptr() as *mut u8,
                out_ptr as *const u8,
                output_size,
            )?;
        }

        Ok(gpu_output)
    }
}
