use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache::Kvcache;
use crate::kernel::{AttentionArch, AttentionConfig, AttentionError, AttentionKernel};
use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use std::sync::Arc;

// Import MemoryBackend trait to use CudaMemoryBackend::alloc
use crate::kernel::memory::MemoryBackend;

/// Flash Attention configuration
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub block_size: usize,
    pub rope_base: f32,
    pub max_pos: usize,
    pub scale: f32, // Pre-computed scaling factor (1/sqrt(head_dim))
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        let scale = 1.0 / 64_f32.sqrt(); // Default head_dim = 64
        Self {
            num_heads: 32,
            head_dim: 64,
            max_seq: 4096,
            block_size: 128,
            rope_base: 10000.0,
            max_pos: 32768,
            scale,
        }
    }
}

/// Flash Attention kernel wrapper implementing the AttentionKernel trait
pub struct FlashAttentionKernel {
    /// CUDA context for kernel execution
    context: Arc<CudaContext>,
    /// CUDA stream for async operations
    stream: Arc<CudaStream>,
    /// Memory backend for device allocations
    memory: crate::kernel::memory::CudaMemoryBackend,
    /// PTX module loaded from file
    ptx_module: Arc<crate::cuda_shim::CudaModule>,
    /// Flash attention config
    config: FlashAttentionConfig,
    /// Whether the kernel is ready to launch
    ready: bool,
}

impl FlashAttentionKernel {
    /// Create a new Flash Attention kernel from PTX file
    #[cfg(feature = "cuda")]
    pub fn new(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        memory: crate::kernel::memory::CudaMemoryBackend,
        config: FlashAttentionConfig,
    ) -> Result<Self, AttentionError> {
        // Load PTX module from file
        // CARGO_MANIFEST_DIR is the pesti-runner directory, so we join relative to that
        let ptx_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("kernel")
            .join("ptx")
            .join("flash_attention_kernel.ptx");

        tracing::info!(ptx_path = ?ptx_path, "Loading Flash Attention PTX kernel");

        // Read PTX file content
        let ptx_content = std::fs::read_to_string(&ptx_path)
            .map_err(|e| AttentionError::Cuda(format!("PTX read: {e}")))?;

        // Load PTX module using cuda_shim
        let ptx_module = crate::cuda_shim::CudaModule::load_from_ptx(&context, &ptx_content)
            .map_err(|e| AttentionError::Cuda(format!("PTX load: {e}")))?;

        tracing::info!("Flash Attention PTX loaded successfully");

        Ok(Self {
            context,
            stream,
            memory,
            ptx_module,
            config,
            ready: true,
        })
    }

    /// Get the CUDA context for backend allocation.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get the CUDA stream for kernel launch and memory operations.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Check if kernel is ready
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Get the config
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }
}

// Implement AttentionKernel trait for FlashAttentionKernel
impl AttentionKernel for FlashAttentionKernel {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        // Use the kernel's internal config (matches what it was built with)
        let flash_config = self.config();

        // Validate shapes
        let query_seq_len = query.len() / (flash_config.num_heads * flash_config.head_dim);

        if key_cache.seq_len() != value_cache.seq_len() {
            return Err(AttentionError::Cuda(format!(
                "Shape mismatch: key seq_len={}, value seq_len={}",
                key_cache.seq_len(), value_cache.seq_len()
            )));
        }

        let seq_len = key_cache.seq_len();

        // Allocate output buffer (Q @ K^T + softmax) @ V -> [query_seq_len, num_heads * head_dim]
        let output_size = query_seq_len * flash_config.num_heads * flash_config.head_dim;
        let output_handle = self
            .memory
            .alloc(output_size * 4) // f32 = 4 bytes
            .map_err(|e| AttentionError::Cuda(format!("Output allocation: {e}")))?;

        let mut output = DeviceBuffer::<f32>::from_backend(output_handle, output_size);

        // Get PTX module and kernel handles
        let ptx_module = self.ptx_module.as_ref();

        // Launch kernel (simplified - actual implementation would need proper grid/block config)
        // TODO: Implement proper kernel launch with mma.sync instructions
        tracing::debug!(
            query_seq_len,
            seq_len,
            num_heads = flash_config.num_heads,
            head_dim = flash_config.head_dim,
            "Launching Flash Attention kernel"
        );

        // Placeholder: return zero output (needs actual PTX launch)
        // In real implementation: ptx_module.get_function("flash_attention")?.launch(...)

        Ok(output)
    }

    fn is_available(&self) -> bool {
        self.ready
    }

    fn arch(&self) -> AttentionArch {
        // Flash attention uses tensor cores (mma.sync) for sm_8.9
        AttentionArch::Wgmma
    }
}

#[cfg(not(feature = "cuda"))]
impl FlashAttentionKernel {
    pub fn new(
        _context: Arc<CudaContext>,
        _stream: Arc<CudaStream>,
        _memory: crate::kernel::memory::CudaMemoryBackend,
        _config: FlashAttentionConfig,
    ) -> Result<Self, AttentionError> {
        Err(AttentionError::NotAvailable)
    }

    pub fn forward(
        &self,
        _query: &DeviceBuffer<f16>,
        _key_cache: &Kvcache,
        _value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        _config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        Err(AttentionError::NotAvailable)
    }

    pub fn is_ready(&self) -> bool {
        false
    }

    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }
}
