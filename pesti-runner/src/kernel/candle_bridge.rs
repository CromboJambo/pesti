//! Candle-core tensor bridge — converts PESTI's DeviceBuffer<f16> to candle-core Tensor.
//!
//! This provides GPU-accelerated tensor operations for PESTI's dispatch layer
//! using candle-core's CUDA backend. When CUDA is available, operations run on GPU;
//! otherwise they fall back to CPU.
//!
//! ## Architecture
//!
//! ```text
//! DeviceBuffer<f16> (PESTI)
//!     │
//!     ├── to_tensor() → Tensor (candle-core)
//!     ├── from_tensor(Tensor) → DeviceBuffer<f16>
//!     ├── rope() → Tensor (RoPE with cos/sin)
//!     ├── sdpa() → Tensor (scaled dot-product attention)
//!     └── gemm() → Tensor (GEMM via matmul)
//! ```
//!
//! ## CUDA Backend
//!
//! candle-core uses `cudarc` under the hood for CUDA. When `Device::Cuda` is used,
//! operations automatically route to GPU. The bridge manages device selection
//! transparently.

use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::ops::{sigmoid, softmax};
use half::f16;
use std::sync::OnceLock;

/// GPU device singleton for the bridge.
///
/// Lazily initializes a CUDA device if available, otherwise falls back to CPU.
static BRIDGE_DEVICE: OnceLock<Device> = OnceLock::new();

/// Get the bridge's GPU device.
///
/// Returns CUDA device if available, otherwise CPU.
pub fn bridge_device() -> &'static Device {
    BRIDGE_DEVICE.get_or_init(|| {
        // Try CUDA first
        if let Ok(device) = Device::new_cuda(0) {
            return device;
        }
        // Fallback to CPU
        Device::Cpu
    })
}

/// Convert a `DeviceBuffer<f16>` to a candle-core `Tensor`.
///
/// # Safety
///
/// The buffer must be valid and contain `len` elements of type `f16`.
pub fn f16_to_tensor(
    data: &[f16],
    shape: &[usize],
    device: Option<&Device>,
) -> Result<Tensor, candle_core::Error> {
    let device = match device {
        Some(d) => d.clone(),
        None => bridge_device().clone(),
    };
    // Convert f16 to f32 for candle-core (candle uses f32 internally)
    let data_f32: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
    Tensor::from_vec(data_f32, shape, &device)
}

/// Convert a candle-core `Tensor` (f32) back to `DeviceBuffer<f16>`.
pub fn tensor_to_f16(tensor: &Tensor) -> Result<Vec<f16>, candle_core::Error> {
    let f32_data: Vec<f32> = tensor.to_vec1()?;
    Ok(f32_data.iter().map(|&x| f16::from_f32(x)).collect())
}

/// Convert a candle-core `Tensor` (f32) to host f32 Vec.
pub fn tensor_to_f32(tensor: &Tensor) -> Result<Vec<f32>, candle_core::Error> {
    tensor.to_vec1()
}

/// Apply Rotary Positional Embedding (RoPE) using candle-core ops.
///
/// Computes RoPE on the last dimension of the input tensor:
/// ```text
/// x_rotated[i, j] = x[i, j] * cos(j) - x[i, j+dim//2] * sin(j)
/// x_rotated[i, j+dim//2] = x[i, j] * sin(j) + x[i, j+dim//2] * cos(j)
/// ```
///
/// # Arguments
/// * `x` — Input tensor [batch, seq_len, hidden] or [batch, seq_len, heads, head_dim]
/// * `cos` — Cosine embeddings [seq_len, dim//2]
/// * `sin` — Sine embeddings [seq_len, dim//2]
/// * `offset` — Sequence offset for KV cache continuation
pub fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    offset: usize,
) -> Result<Tensor, candle_core::Error> {
    let dims = x.dims();
    let dim = dims
        .last()
        .ok_or(candle_core::Error::Msg("x has no last dim".into()))?
        / 2;

    let chunks = x.narrow(dims.len() - 1, 0, dim)?.chunk(2, dims.len() - 1)?;
    let x0 = &chunks[0];
    let x1 = &chunks[1];

    let cos_chunks = cos.narrow(0, offset, dim)?.chunk(2, dim - 1)?;
    let cos0 = &cos_chunks[0];
    let cos1 = &cos_chunks[1];

    let sin_chunks = sin.narrow(0, offset, dim)?.chunk(2, dim - 1)?;
    let sin0 = &sin_chunks[0];
    let sin1 = &sin_chunks[1];

    // x0 * cos0 - x1 * sin1
    let x0_cos0 = x0.matmul(cos0)?;
    let x1_sin1 = x1.matmul(sin1)?;
    let part0 = x0_cos0.broadcast_sub(&x1_sin1)?;
    // x1 * cos1 + x0 * sin0
    let x1_cos1 = x1.matmul(cos1)?;
    let x0_sin0 = x0.matmul(sin0)?;
    let part1 = x1_cos1.broadcast_add(&x0_sin0)?;

    // Concatenate along last dimension
    Tensor::cat(&[part0, part1], dims.len() - 1)
}

/// Compute RoPE embeddings (cos/sin) for a given sequence length.
///
/// Uses the standard RoPE formula:
/// ```text
/// cos = cos(pos / 10000^(2j/d))
/// sin = sin(pos / 10000^(2j/d))
/// ```
///
/// # Arguments
/// * `seq_len` — Sequence length
/// * `head_dim` — Dimension per attention head
/// * `base` — RoPE base (typically 10000.0)
/// * `max_pos` — Maximum position (for scaling)
pub fn rope_embeddings(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    _max_pos: usize,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..dim)
        .map(|i| base.powf(-(i as f32 * 2.0) / head_dim as f32))
        .collect();

    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();

    // Compute positions * inv_freq
    let _shape = (seq_len, dim);
    let positions_t = Tensor::from_vec(positions, (seq_len, 1), bridge_device())?;
    let inv_freq_t = Tensor::from_vec(inv_freq, (1, dim), bridge_device())?;

    // angles = positions * inv_freq: [seq_len, dim]
    let angles = positions_t.matmul(&inv_freq_t)?;

    let cos = angles.cos()?;
    let sin = angles.sin()?;

    Ok((cos, sin))
}

/// Scaled Dot-Product Attention (SDPA) using candle-core ops.
///
/// Computes: `softmax(Q @ K^T / sqrt(head_dim)) @ V`
///
/// # Arguments
/// * `q` — Query tensor [batch, seq_q, n_heads, head_dim]
/// * `k` — Key tensor [batch, seq_k, n_heads, head_dim]
/// * `v` — Value tensor [batch, seq_k, n_heads, head_dim]
/// * `scale` — Attention scale (1/sqrt(head_dim))
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor, candle_core::Error> {
    let (_, seq_q, _, _) = q.dims4()?;
    let (_, seq_k, _, _) = k.dims4()?;

    // Q @ K^T: [batch, seq_q, n_heads, seq_k]
    let attn_weights = q.matmul(&k.transpose(2, 3)?)?;

    // Scale: attn_weights * scale
    let scale_tensor = Tensor::new(scale, bridge_device())?;
    let attn_weights = (&attn_weights * &scale_tensor)?;

    // Create causal mask: 0 where i >= j, -inf where i < j
    let mask_shape = (seq_q, seq_k);
    let mut mask_data = vec![0.0f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            if i < j {
                mask_data[i * seq_k + j] = -1e9;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, mask_shape, bridge_device())?;
    let attn_weights = (&attn_weights + &mask)?;

    // Softmax along last dimension
    let attn_weights = softmax(&attn_weights, candle_core::D::Minus1)?;

    // attn_weights @ V: [batch, seq_q, n_heads, seq_k] @ [batch, seq_k, n_heads, head_dim]
    // → [batch, seq_q, n_heads, head_dim]
    attn_weights.matmul(v)
}

/// GEMM: matrix multiply on GPU via candle-core.
///
/// Computes: `C = alpha * (A @ B) + beta * C`
///
/// # Arguments
/// * `a` — Input A [m, k]
/// * `b` — Input B [k, n]
/// * `c` — Optional bias/output [n]
/// * `alpha` — Scale for A @ B
/// * `beta` — Scale for C
pub fn gemm(
    a: &[f16],
    b: &[f16],
    c: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<Vec<f32>, candle_core::Error> {
    let device = bridge_device();

    let a_t = Tensor::from_vec(
        a.iter().map(|&x| x.to_f32()).collect::<Vec<_>>(),
        (m, k),
        device,
    )?;
    let b_t = Tensor::from_vec(
        b.iter().map(|&x| x.to_f32()).collect::<Vec<_>>(),
        (k, n),
        device,
    )?;

    // A @ B
    let mut result = a_t.matmul(&b_t)?;

    // alpha * result
    if alpha != 1.0 {
        let alpha_t = Tensor::new(alpha, device)?;
        result = (&result * &alpha_t)?;
    }

    // beta * c + result
    if let Some(c_data) = c {
        let c_t = Tensor::from_vec(c_data.to_vec(), (1, n), device)?;
        let c_broadcast = c_t.broadcast_as((m, n))?;
        if beta != 0.0 {
            let beta_t = Tensor::new(beta, device)?;
            result = (&result + &(&c_broadcast * &beta_t)?)?;
        } else {
            result = (&result + &c_broadcast)?;
        }
    } else if beta != 0.0 {
        // Zero output with beta scale (shouldn't happen, but handle it)
        let zeros = Tensor::zeros((m, n), DType::F32, device)?;
        let beta_t = Tensor::new(beta, device)?;
        result = (&result + &(&zeros * &beta_t)?)?;
    }

    result
        .to_vec2::<f32>()
        .map(|mat| mat.into_iter().flatten().collect())
}

/// Apply RMSNorm (Root Mean Square Layer Normalization).
///
/// ```text
/// rms = sqrt(mean(x^2) + eps)
/// output = x / rms
/// ```
pub fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor, candle_core::Error> {
    let variance = (x * x)?.mean_keepdim(candle_core::D::Minus1)?;
    let normed = (x / (variance + eps)?.sqrt()?)?;
    Ok(normed)
}

/// GELU activation.
pub fn gelu(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    x.gelu()
}

/// SwiGLU activation (Swish-Gated Linear Unit).
///
/// ```text
/// x → (x1, x2) split on last dim
/// output = x1 * sigmoid(x2)
/// ```
pub fn swiglu(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let chunks = x.chunk(2, candle_core::D::Minus1)?;
    let x1 = &chunks[0];
    let x2 = &chunks[1];
    x1 * sigmoid(x2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_device() {
        let device = bridge_device();
        assert!(device.is_cpu() || device.is_cuda());
    }

    #[test]
    fn test_f16_roundtrip() {
        let data: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let tensor = f16_to_tensor(&data, &[3], None).unwrap();
        let result = tensor_to_f16(&tensor).unwrap();
        for (a, b) in data.iter().zip(result.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_embeddings() {
        let (cos, sin) = rope_embeddings(10, 64, 10000.0, 2048).unwrap();
        let cos_shape = cos.dims();
        assert_eq!(cos_shape, &[10usize, 32]);
        let sin_shape = sin.dims();
        assert_eq!(sin_shape, &[10usize, 32]);
    }

    #[test]
    fn test_gemm_identity() {
        let a: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
        ];
        let b: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];

        let result = gemm(&a, &b, None, 2, 2, 2, 1.0, 0.0).unwrap();
        // Identity @ b = b
        assert!((result[0] - 1.0).abs() < 1e-3);
        assert!((result[1] - 2.0).abs() < 1e-3);
        assert!((result[2] - 3.0).abs() < 1e-3);
        assert!((result[3] - 4.0).abs() < 1e-3);
    }
}
