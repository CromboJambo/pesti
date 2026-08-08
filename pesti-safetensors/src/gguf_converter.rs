use std::collections::HashMap;
use std::io::{Read, Seek};
use std::path::Path;

use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::{GgufDtype, GgufHeader, GgufKvPair, GgufTensorInfo};
use safetensors::serialize;
use safetensors::tensor::{Dtype, TensorView};

// Use half crate from pesti-gguf dependency
use pesti_gguf::types as gguf_types;

/// Result of a GGUF → safetensors conversion.
pub struct GgufConversionResult {
    pub model_name: String,
    pub tensor_count: usize,
    pub total_bytes: u64,
    pub dtype: String,
    pub kv_pairs: HashMap<String, String>,
}

/// Error type for GGUF → safetensors conversion.
#[derive(Debug, thiserror::Error)]
pub enum GgufConvertError {
    #[error("GGUF parse error: {0}")]
    GgufParse(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("safetensors serialize error: {0}")]
    Serialize(String),

    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(u32),

    #[error("GGUF error: {0}")]
    Gguf(#[from] pesti_gguf::GgufError),

    #[error("tensor mismatch: {0}")]
    TensorMismatch(String),
}

/// Dequantize Q4_0 data to f32.
fn dequantize_q4_0(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_full_blocks * 20
        + if remaining > 0 {
            4 + remaining.div_ceil(2)
        } else {
            0
        };
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q4_0 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 20;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        let min = f16_to_f32(&data[base + 2..base + 4])[0];

        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32 + min);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 20;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        let min = f16_to_f32(&data[base + 2..base + 4])[0];
        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32 + min);
        }
    }

    Ok(result)
}

/// Dequantize Q4_1 data to f32.
fn dequantize_q4_1(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_full_blocks * 20
        + if remaining > 0 {
            4 + remaining.div_ceil(2)
        } else {
            0
        };
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q4_1 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 20;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        let min = f16_to_f32(&data[base + 2..base + 4])[0];

        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as f32;
            result.push(scale * q + min);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 20;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        let min = f16_to_f32(&data[base + 2..base + 4])[0];
        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as f32;
            result.push(scale * q + min);
        }
    }

    Ok(result)
}

/// Dequantize Q8_0 data to f32.
fn dequantize_q8_0(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_blocks = element_count.div_ceil(256);
    let expected_size = num_blocks * 258;
    if data.len() < expected_size {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q8_0 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_blocks {
        let base = block * 258;
        let scale = f16_to_f32(&data[base..base + 2])[0];
        for i in 0..256usize {
            if result.len() >= element_count {
                break;
            }
            let q = data[base + 2 + i] as i8 as f32 / 128.0;
            result.push(scale * q);
        }
    }
    Ok(result)
}

/// Dequantize Q2_K data to f32.
fn dequantize_q2_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64) / 4 + (element_count as u64) * 6 / 32 + 8;

    if data.len() < expected_size as usize {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q2_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 16;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let d2 = f16_to_f32(&data[base + 2..base + 4])[0];
        let q1 = [data[base + 4], data[base + 5]];
        let q2 = [
            data[base + 6],
            data[base + 7],
            data[base + 8],
            data[base + 9],
            data[base + 10],
            data[base + 11],
        ];
        let h_bits = [&data[base + 12..base + 14], &data[base + 14..base + 16]];
        let h0 = f16_to_f32(h_bits[0])[0];
        let h1 = f16_to_f32(h_bits[1])[0];

        for i in 0..16usize {
            let q1_val = (q1[i / 4] >> (2 * (i % 4))) & 0x03;
            let q2_val = ((q2[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
            let h_val = if i < 4 { h0 } else { h1 };
            let q = (q2_val - 2) * (q1_val as i32 + 1);
            result.push(d * (q as f32 / 16.0 - h_val) + d2);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 16;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let d2 = f16_to_f32(&data[base + 2..base + 4])[0];
        let q1 = [data[base + 4], data[base + 5]];
        let q2 = [
            data[base + 6],
            data[base + 7],
            data[base + 8],
            data[base + 9],
        ];
        let h_bits = [&data[base + 10..base + 12]];
        let h0 = f16_to_f32(h_bits[0])[0];

        for i in 0..remaining {
            let q1_val = (q1[i / 4] >> (2 * (i % 4))) & 0x03;
            let q2_val = ((q2[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
            let q = (q2_val - 2) * (q1_val as i32 + 1);
            result.push(d * (q as f32 / 16.0 - h0) + d2);
        }
    }

    Ok(result)
}

/// Dequantize Q3_K data to f32.
fn dequantize_q3_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64) / 8 + (element_count as u64) * 6 / 32 + 16;

    if data.len() < expected_size as usize {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q3_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for _block in 0..num_full_blocks {
        let base = _block * 24;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let d_min = f16_to_f32(&data[base + 2..base + 4])[0];
        let delta = data[base + 4] as i32;
        let k_scale = [
            data[base + 5],
            data[base + 6],
            data[base + 7],
            data[base + 8],
        ];
        let q3_bits = [
            data[base + 9],
            data[base + 10],
            data[base + 11],
            data[base + 12],
            data[base + 13],
            data[base + 14],
        ];
        let mask = data[base + 15];
        let h_bits = [
            &data[base + 16..base + 18],
            &data[base + 18..base + 20],
            &data[base + 20..base + 22],
            &data[base + 22..base + 24],
        ];
        let h = [
            f16_to_f32(h_bits[0])[0],
            f16_to_f32(h_bits[1])[0],
            f16_to_f32(h_bits[2])[0],
            f16_to_f32(h_bits[3])[0],
        ];

        for i in 0..16usize {
            let k = k_scale[i / 4] as i32;
            let h_val = h[i / 4];
            let q3_val = (q3_bits[i / 8] >> (3 * (i % 8))) & 0x07;
            let mask_bit = (mask >> i) & 1;
            let combined_scale = k as f32 * delta as f32;
            let q = if mask_bit != 0 {
                (q3_val as i32 - 4) * k
            } else {
                (q3_val as i32 - 4) * k + (1 << 6)
            };
            result.push(d * (q as f32 / 64.0) + d_min - h_val * combined_scale);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 24;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let d_min = f16_to_f32(&data[base + 2..base + 4])[0];
        let delta = data[base + 4] as i32;
        let k_scale = [
            data[base + 5],
            data[base + 6],
            data[base + 7],
            data[base + 8],
        ];
        let q3_bits = [
            data[base + 9],
            data[base + 10],
            data[base + 11],
            data[base + 12],
            data[base + 13],
            data[base + 14],
        ];
        let mask = data[base + 15];
        let h_bits = [
            &data[base + 16..base + 18],
            &data[base + 18..base + 20],
            &data[base + 20..base + 22],
            &data[base + 22..base + 24],
        ];
        let h = [
            f16_to_f32(h_bits[0])[0],
            f16_to_f32(h_bits[1])[0],
            f16_to_f32(h_bits[2])[0],
            f16_to_f32(h_bits[3])[0],
        ];

        for i in 0..remaining {
            let k = k_scale[i / 4] as i32;
            let h_val = h[i / 4];
            let q3_val = (q3_bits[i / 8] >> (3 * (i % 8))) & 0x07;
            let mask_bit = (mask >> i) & 1;
            let combined_scale = k as f32 * delta as f32;
            let q = if mask_bit != 0 {
                (q3_val as i32 - 4) * k
            } else {
                (q3_val as i32 - 4) * k + (1 << 6)
            };
            result.push(d * (q as f32 / 64.0) + d_min - h_val * combined_scale);
        }
    }

    Ok(result)
}

/// Dequantize Q4_K data to f32.
fn dequantize_q4_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64) / 4 + (element_count as u64) * 6 / 32 + 16 + 32;

    if data.len() < expected_size as usize {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q4_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 24;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let d_min = f16_to_f32(&data[base + 2..base + 4])[0];
        let scale_lo = data[base + 4];
        let scale_hi = data[base + 5];
        let q4_lo = [data[base + 6], data[base + 7]];
        let q4_hi = [data[base + 8], data[base + 9]];

        for i in 0..16usize {
            let lo = (q4_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = (q4_hi[i / 2] >> (4 * (i % 2))) & 0x0F;
            let scale = if hi > 0 {
                d * (scale_lo as f32 + scale_hi as f32 * 1.0 / 32.0)
            } else {
                d * scale_lo as f32
            };
            let q = (lo as i32) - 8 + (hi as i32) * 16;
            result.push(scale * (q as f32 / 16.0) + d_min);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 24;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let d_min = f16_to_f32(&data[base + 2..base + 4])[0];
        let scale_lo = data[base + 4];
        let scale_hi = data[base + 5];
        let q4_lo = [data[base + 6], data[base + 7]];
        let q4_hi = [data[base + 8], data[base + 9]];

        for i in 0..remaining {
            let lo = (q4_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = (q4_hi[i / 2] >> (4 * (i % 2))) & 0x0F;
            let scale = if hi > 0 {
                d * (scale_lo as f32 + scale_hi as f32 * 1.0 / 32.0)
            } else {
                d * scale_lo as f32
            };
            let q = (lo as i32) - 8 + (hi as i32) * 16;
            result.push(scale * (q as f32 / 16.0) + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q5_K data to f32.
fn dequantize_q5_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64) / 4 + (element_count as u64) * 6 / 32 + 16 + 32 + 16;

    if data.len() < expected_size as usize {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q5_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 32;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let d_min = f16_to_f32(&data[base + 2..base + 4])[0];
        let scale = data[base + 4] as f32;
        let q5_lo = [data[base + 6], data[base + 7]];
        let q5_h = [data[base + 10], data[base + 11]];

        for i in 0..16usize {
            let lo = (q5_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = ((q5_h[i / 8] >> (i % 8)) & 1) as i32;
            let q = lo as i32 + hi * 16;
            result.push(d * ((q as f32 - 16.0) / 16.0) + d_min + scale);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 32;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let d_min = f16_to_f32(&data[base + 2..base + 4])[0];
        let scale = data[base + 4] as f32;
        let q5_lo = [data[base + 6], data[base + 7]];
        let q5_h = [data[base + 10]];

        for i in 0..remaining {
            let lo = (q5_lo[i / 2] >> (4 * (i % 2))) & 0x0F;
            let hi = ((q5_h[i / 8] >> (i % 8)) & 1) as i32;
            let q = lo as i32 + hi * 16;
            result.push(d * ((q as f32 - 16.0) / 16.0) + d_min + scale);
        }
    }

    Ok(result)
}

/// Dequantize Q6_K data to f32.
fn dequantize_q6_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64) / 2 + (element_count as u64) / 4 + 256;

    if data.len() < expected_size as usize {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q6_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 24;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let mask = data[base + 2];
        let q6 = [
            data[base + 3],
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
            data[base + 8],
            data[base + 9],
            data[base + 10],
            data[base + 11],
            data[base + 12],
            data[base + 13],
            data[base + 14],
        ];
        let scale = data[base + 15] as f32;

        for i in 0..16usize {
            let q6_val = ((q6[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
            let mask_bit = (mask >> i) & 1;
            let combined = if mask_bit != 0 { q6_val + 4 } else { q6_val };
            result.push(d * ((combined as f32 - 32.0) / 32.0) * scale);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 24;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let mask = data[base + 2];
        let q6 = [
            data[base + 3],
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
            data[base + 8],
        ];
        let scale = data[base + 9] as f32;

        for i in 0..remaining {
            let q6_val = ((q6[i / 4] >> (2 * (i % 4))) & 0x03) as i32;
            let mask_bit = (mask >> i) & 1;
            let combined = if mask_bit != 0 { q6_val + 4 } else { q6_val };
            result.push(d * ((combined as f32 - 32.0) / 32.0) * scale);
        }
    }

    Ok(result)
}

/// Dequantize Q8_K data to f32.
fn dequantize_q8_k(data: &[u8], element_count: usize) -> Result<Vec<f32>, GgufConvertError> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (element_count as u64) / 2 + (element_count as u64) * 6 / 32 + 256;

    if data.len() < expected_size as usize {
        return Err(GgufConvertError::TensorMismatch(format!(
            "Q8_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    for block in 0..num_full_blocks {
        let base = block * 18;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let q8 = &data[base + 2..base + 18];

        for q in q8.iter() {
            let q_val = *q as i8 as f32 / 128.0;
            result.push(d * q_val);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 18;
        let d = f16_to_f32(&data[base..base + 2])[0];
        let q8 = &data[base + 2..base + 2 + remaining];

        for q in q8.iter() {
            let q_val = *q as i8 as f32 / 128.0;
            result.push(d * q_val);
        }
    }

    Ok(result)
}

/// Convert u16 (f16) to Vec<f32>.
fn f16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let val = u16::from_le_bytes([chunk[0], chunk[1]]);
            half::f16::from_bits(val).to_f32()
        })
        .collect()
}

/// Dequantize tensor data based on dtype.
fn dequantize_tensor(
    tensor: &GgufTensorInfo,
    raw_data: &[u8],
) -> Result<Vec<u8>, GgufConvertError> {
    let dtype = GgufDtype::from_u32(tensor.dtype);
    let element_count = tensor.element_count() as usize;

    match dtype {
        GgufDtype::F32 => Ok(raw_data.to_vec()),
        GgufDtype::F16 | GgufDtype::BF16 => {
            let f32_data = f16_to_f32(raw_data);
            Ok(f32_data.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::I8 | GgufDtype::I16 | GgufDtype::I32 | GgufDtype::I64 => Ok(raw_data.to_vec()),
        GgufDtype::F64 => {
            let f32_data: Vec<f32> = raw_data
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect();
            Ok(f32_data.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q4_0 => {
            let dequantized = dequantize_q4_0(raw_data, element_count)?;
            Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q4_1 => {
            let dequantized = dequantize_q4_1(raw_data, element_count)?;
            Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q8_0 => {
            let dequantized = dequantize_q8_0(raw_data, element_count)?;
            Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q2K | GgufDtype::Q3K => {
            // Placeholder - would need full implementation
            Err(GgufConvertError::UnsupportedDtype(tensor.dtype))
        }
        GgufDtype::Q4K | GgufDtype::Q4K_M | GgufDtype::Q4K_S => {
            let dequantized = dequantize_q4_k(raw_data, element_count)?;
            Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q5K | GgufDtype::Q5K_M | GgufDtype::Q5K_S => {
            let dequantized = dequantize_q5_k(raw_data, element_count)?;
            Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q6K | GgufDtype::Q6K_S => {
            let dequantized = dequantize_q6_k(raw_data, element_count)?;
            Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q8K | GgufDtype::Q8K_M => {
            let dequantized = dequantize_q8_k(raw_data, element_count)?;
            Ok(dequantized.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        // Unsupported variants
        GgufDtype::Q5_0
        | GgufDtype::Q5_1
        | GgufDtype::Q8_1
        | GgufDtype::Q1K
        | GgufDtype::Q2K_S
        | GgufDtype::Q3K_S => Err(GgufConvertError::UnsupportedDtype(tensor.dtype)),
        // Catch-all for new IQ* quantization types and Unknown variants
        GgufDtype::Unknown(_)
        | GgufDtype::IQ2_XXS
        | GgufDtype::IQ2_XS
        | GgufDtype::IQ3_XXS
        | GgufDtype::IQ1_S
        | GgufDtype::Q4_0_4_4
        | GgufDtype::Q4_0_4_8
        | GgufDtype::Q4_0_8_8
        | GgufDtype::TQ1_0
        | GgufDtype::TQ2_0
        | GgufDtype::IQ4NL_4_4
        | GgufDtype::IQ4NL_4_8
        | GgufDtype::IQ4NL_8_8
        | GgufDtype::MXFP4
        | GgufDtype::NVFP4
        | GgufDtype::Q1_0
        | GgufDtype::Q2_0
        | GgufDtype::Q2K_M => Err(GgufConvertError::UnsupportedDtype(tensor.dtype)),
    }
}

/// Map a GGUF dtype to a safetensors Dtype.
fn gguf_dtype_to_safetensors(gguf_dtype: GgufDtype) -> Result<Dtype, GgufConvertError> {
    match gguf_dtype {
        GgufDtype::F32 => Ok(Dtype::F32),
        GgufDtype::F16 | GgufDtype::BF16 => Ok(Dtype::F32),
        GgufDtype::I8 => Ok(Dtype::I8),
        GgufDtype::I16 => Ok(Dtype::I16),
        GgufDtype::I32 => Ok(Dtype::I32),
        GgufDtype::I64 => Ok(Dtype::I64),
        GgufDtype::F64 => Ok(Dtype::F64),
        // Supported dequantization types — dequantize_tensor() handles these
        GgufDtype::Q4_0 | GgufDtype::Q4_1 | GgufDtype::Q8_0 => Ok(Dtype::F32),
        // Unsupported: K-family types without full dequantization
        GgufDtype::Q5_0
        | GgufDtype::Q5_1
        | GgufDtype::Q8_1
        | GgufDtype::Q1K
        | GgufDtype::Q2K_S
        | GgufDtype::Q3K_S => Err(GgufConvertError::UnsupportedDtype(gguf_dtype.to_u32())),
        // K-family types with dequantization
        GgufDtype::Q2K
        | GgufDtype::Q3K
        | GgufDtype::Q4K
        | GgufDtype::Q4K_M
        | GgufDtype::Q4K_S
        | GgufDtype::Q5K
        | GgufDtype::Q5K_M
        | GgufDtype::Q5K_S
        | GgufDtype::Q6K
        | GgufDtype::Q6K_S
        | GgufDtype::Q8K
        | GgufDtype::Q8K_M => Ok(Dtype::F32),
        // Catch-all for new IQ* quantization types and Unknown variants
        GgufDtype::Unknown(_)
        | GgufDtype::IQ2_XXS
        | GgufDtype::IQ2_XS
        | GgufDtype::IQ3_XXS
        | GgufDtype::IQ1_S
        | GgufDtype::Q4_0_4_4
        | GgufDtype::Q4_0_4_8
        | GgufDtype::Q4_0_8_8
        | GgufDtype::TQ1_0
        | GgufDtype::TQ2_0
        | GgufDtype::IQ4NL_4_4
        | GgufDtype::IQ4NL_4_8
        | GgufDtype::IQ4NL_8_8
        | GgufDtype::MXFP4
        | GgufDtype::NVFP4
        | GgufDtype::Q1_0
        | GgufDtype::Q2_0
        | GgufDtype::Q2K_M => Err(GgufConvertError::UnsupportedDtype(gguf_dtype.to_u32())),
    }
}

/// Convert a GGUF model to safetensors format.
pub fn convert_gguf_to_safetensors(
    gguf_path: &Path,
    output_dir: &Path,
) -> Result<GgufConversionResult, GgufConvertError> {
    // Parse GGUF header
    let header = parse_gguf(gguf_path)?;

    // Extract metadata from kv_pairs
    let mut kv_map: HashMap<String, String> = HashMap::new();
    for kv in &header.kv_pairs {
        if let Some(s) = kv.value.as_str() {
            kv_map.insert(kv.key.clone(), s.to_string());
        } else if let Some(n) = kv.value.as_u32() {
            kv_map.insert(kv.key.clone(), n.to_string());
        }
    }

    let model_name = header
        .get_kv_str("general.name")
        .unwrap_or(&"unknown".to_string())
        .to_string();
    let base_model = header
        .get_kv_str("general.base_model")
        .unwrap_or(&"unknown".to_string())
        .to_string();

    eprintln!(
        "Converting GGUF model: {} (base: {}, tensors: {})",
        model_name,
        base_model,
        header.tensors.len()
    );

    // Convert each tensor
    let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> = Vec::new();
    let mut total_bytes: u64 = 0;

    for tensor in &header.tensors {
        let raw_data = extract_tensor_bytes(gguf_path, tensor)?;
        let dequantized = dequantize_tensor(tensor, &raw_data)?;
        let shape: Vec<usize> = tensor.shape.iter().map(|s| *s as usize).collect();

        total_bytes += dequantized.len() as u64;
        let dtype = gguf_dtype_to_safetensors(GgufDtype::from_u32(tensor.dtype))?;

        tensor_data.push((tensor.name.clone(), dequantized, shape, dtype));
    }

    // Serialize to safetensors format
    let output_path = output_dir.join(format!("{}.safetensors", model_name));

    let tensors: Vec<(String, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, data, shape, dtype)| {
            let view = TensorView::new(*dtype, shape.clone(), data.as_slice()).unwrap();
            (name.clone(), view)
        })
        .collect();

    let serialized = serialize(tensors.into_iter().collect::<Vec<_>>(), None)
        .map_err(|e| GgufConvertError::Serialize(e.to_string()))?;
    std::fs::write(&output_path, &serialized)?;

    eprintln!("Converted GGUF to safetensors: {}", output_path.display());

    Ok(GgufConversionResult {
        model_name,
        tensor_count: header.tensors.len(),
        total_bytes,
        dtype: "F32".to_string(),
        kv_pairs: kv_map,
    })
}

/// Extract raw tensor bytes from a GGUF file.
fn extract_tensor_bytes(
    gguf_path: &Path,
    tensor: &GgufTensorInfo,
) -> Result<Vec<u8>, GgufConvertError> {
    let mut file = std::fs::File::open(gguf_path)?;
    let offset = tensor.offset as u64;
    let size = tensor.stored_size().unwrap_or(0) as usize;

    file.seek(std::io::SeekFrom::Start(offset))?;
    let mut buffer = vec![0u8; size];
    file.read_exact(&mut buffer)?;

    Ok(buffer)
}

/// Verify a GGUF file's integrity by parsing its header.
pub fn verify_gguf_integrity(gguf_path: &Path) -> Result<(), GgufConvertError> {
    let _header = parse_gguf(gguf_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        // f16 0x3C00 = 1.0
        let bytes = vec![0x00u8, 0x3C];
        let result = f16_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gguf_dtype_to_safetensors_known_types() {
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::F32).unwrap(),
            Dtype::F32
        );
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::F16).unwrap(),
            Dtype::F32
        );
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::BF16).unwrap(),
            Dtype::F32
        );
        assert_eq!(gguf_dtype_to_safetensors(GgufDtype::I8).unwrap(), Dtype::I8);
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::I16).unwrap(),
            Dtype::I16
        );
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::I32).unwrap(),
            Dtype::I32
        );
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::I64).unwrap(),
            Dtype::I64
        );
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::F64).unwrap(),
            Dtype::F64
        );
    }

    #[test]
    fn test_gguf_dtype_to_safetensors_quantized_returns_f32() {
        // Supported dequantization types map to f32
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::Q4_0).unwrap(),
            Dtype::F32
        );
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::Q4_1).unwrap(),
            Dtype::F32
        );
        assert_eq!(
            gguf_dtype_to_safetensors(GgufDtype::Q8_0).unwrap(),
            Dtype::F32
        );
    }

    #[test]
    fn test_dequantize_q4_0() {
        // Create a simple Q4_0 block: scale=1.0, min=0.0, all quantized values = 0 (which means -8)
        let mut block = vec![0u8; 20];
        // scale = 1.0 f16 = 0x3C00
        block[0..2].copy_from_slice(&[0x00, 0x3C]);
        // min = 0.0 f16 = 0x0000
        block[2..4].copy_from_slice(&[0x00, 0x00]);
        // quantized values = 0 (all nibbles = 0)
        block[4] = 0x00;

        let result = dequantize_q4_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        // All values should be -8.0 (scale * (0-8) + min = 1.0 * (-8) + 0.0 = -8.0)
        for v in result.iter() {
            assert!((v - (-8.0)).abs() < 0.1, "Expected -8.0, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q4_k() {
        // Create a simple Q4_K block with known values
        let mut block = vec![0u8; 24];
        // d (scale) = 1.0 f16
        block[0..2].copy_from_slice(&[0x00, 0x3C]);
        // d_min (min) = 0.0 f16
        block[2..4].copy_from_slice(&[0x00, 0x00]);
        // scale_lo = 1.0, scale_hi = 0.0
        block[4] = 0x01;
        block[5] = 0x00;
        // q4_lo = all zeros (quantized values = -8)
        block[6] = 0x00;
        block[7] = 0x00;
        // q4_hi = all zeros
        block[8] = 0x00;
        block[9] = 0x00;

        let result = dequantize_q4_k(&block, 16).unwrap();
        assert_eq!(result.len(), 16);
        // With all quantized values = 0 and scale_hi = 0: q = -8, result = d * (-8/16) + d_min = -0.5
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-0.5)).abs() < 0.1,
                "Q4_K element {i} = {v}, expected -0.5"
            );
        }
    }

    #[test]
    fn test_dequantize_q5_k() {
        // Create a simple Q5_K block with known values
        let mut block = vec![0u8; 32];
        // d (scale) = 1.0 f16
        block[0..2].copy_from_slice(&[0x00, 0x3C]);
        // d_min (min) = 0.0 f16
        block[2..4].copy_from_slice(&[0x00, 0x00]);
        // scale = 0.0
        block[4] = 0x00;
        // q5_lo = all zeros (quantized values = 0)
        block[6] = 0x00;
        block[7] = 0x00;
        // q5_h = all zeros (upper bits = 0)
        block[10] = 0x00;

        let result = dequantize_q5_k(&block, 16).unwrap();
        assert_eq!(result.len(), 16);
        // With all quantized values = 0: q = 0, result = d * (-16/16) + d_min + scale = -1.0
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 0.1,
                "Q5_K element {i} = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequantize_q6_k() {
        // Create a simple Q6_K block with known values
        let mut block = vec![0u8; 24];
        // d (scale) = 1.0 f16
        block[0..2].copy_from_slice(&[0x00, 0x3C]);
        // mask = 0 (all bits set in dequantization)
        block[2] = 0x00;
        // q6 = all zeros (quantized values = 0)
        block[3..15].fill(0x00);
        // scale = 1.0
        block[15] = 0x80; // f16 1.0 in u8

        let result = dequantize_q6_k(&block, 16).unwrap();
        assert_eq!(result.len(), 16);
        // With all quantized values = 0 and mask = 0: combined = 0, result = d * (-32/32) * scale = -1.0
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 0.1,
                "Q6_K element {i} = {v}, expected -1.0"
            );
        }
    }
}
