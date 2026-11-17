/// Little-endian byte-order helpers for GGUF writing (shared with parser).
mod le_bytes {
    #[inline]
    pub fn write_u8<W: std::io::Write>(writer: &mut W, val: u8) -> Result<(), std::io::Error> {
        writer.write_all(&[val])
    }

    #[inline]
    pub fn write_u16_le<W: std::io::Write>(writer: &mut W, val: u16) -> Result<(), std::io::Error> {
        writer.write_all(&val.to_le_bytes())
    }

    #[inline]
    pub fn write_u32_le<W: std::io::Write>(writer: &mut W, val: u32) -> Result<(), std::io::Error> {
        writer.write_all(&val.to_le_bytes())
    }

    #[inline]
    pub fn write_u64_le<W: std::io::Write>(writer: &mut W, val: u64) -> Result<(), std::io::Error> {
        writer.write_all(&val.to_le_bytes())
    }

    #[inline]
    pub fn write_f32_le<W: std::io::Write>(writer: &mut W, val: f32) -> Result<(), std::io::Error> {
        writer.write_all(&val.to_le_bytes())
    }

    #[inline]
    pub fn write_f64_le<W: std::io::Write>(writer: &mut W, val: f64) -> Result<(), std::io::Error> {
        writer.write_all(&val.to_le_bytes())
    }

    #[inline]
    pub fn write_i8<W: std::io::Write>(writer: &mut W, val: i8) -> Result<(), std::io::Error> {
        writer.write_all(&[val as u8])
    }

    #[inline]
    pub fn write_i16_le<W: std::io::Write>(writer: &mut W, val: i16) -> Result<(), std::io::Error> {
        writer.write_all(&val.to_le_bytes())
    }

    #[inline]
    pub fn write_i32_le<W: std::io::Write>(writer: &mut W, val: i32) -> Result<(), std::io::Error> {
        writer.write_all(&val.to_le_bytes())
    }

    #[inline]
    pub fn write_i64_le<W: std::io::Write>(writer: &mut W, val: i64) -> Result<(), std::io::Error> {
        writer.write_all(&val.to_le_bytes())
    }
}

use le_bytes::{write_f32_le, write_f64_le, write_i16_le, write_i32_le, write_i64_le, write_i8, write_u16_le, write_u32_le, write_u64_le, write_u8};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, Write};
use std::path::Path;

use crate::error::GgufError;
use crate::types::{GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType, GgufWireFormat, IntWidth};

/// GGUF file writer for serializing parsed models back to disk.
///
/// Supports GGUF v3 practical format with u64 key lengths and alignment padding.
pub struct GgufWriter {
    version: u32,
    kv_pairs: Vec<GgufKvPair>,
    tensors: Vec<GgufTensorInfo>,
    tensor_data: Vec<(String, Vec<u8>)>, // (tensor_name, raw_bytes)
    alignment: u64,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    /// Create a new GGUF writer with v3 format.
    pub fn new() -> Self {
        Self {
            version: 3,
            kv_pairs: Vec::new(),
            tensors: Vec::new(),
            tensor_data: Vec::new(),
            alignment: 256, // Default to 256-byte alignment as per llama.cpp
        }
    }

    /// Set the GGUF version (1, 2, or 3).
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Returns the wire format for this writer's version.
    fn wire_format(&self) -> GgufWireFormat {
        match self.version {
            1 => GgufWireFormat::V1,
            2 => GgufWireFormat::V2,
            _ => GgufWireFormat::V3,
        }
    }

    /// Write a u32 or u64 based on the wire format's integer width.
    fn write_int(writer: &mut BufWriter<File>, val: u64, width: IntWidth) -> Result<(), GgufError> {
        match width {
            IntWidth::U32 => write_u32_le(writer, val as u32)?,
            IntWidth::U64 => write_u64_le(writer, val)?,
        }
        Ok(())
    }

    /// Set alignment (default: 256 bytes).
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment;
        self
    }

    /// Add a key-value pair to the header.
    pub fn add_kv(&mut self, kv: GgufKvPair) {
        self.kv_pairs.push(kv);
    }

    /// Add tensor metadata.
    pub fn add_tensor(&mut self, tensor: GgufTensorInfo) {
        self.tensors.push(tensor);
    }

    /// Add raw tensor data (should match tensor names in order).
    pub fn add_tensor_data(&mut self, name: &str, data: Vec<u8>) {
        self.tensor_data.push((name.to_string(), data));
    }

    /// Write the GGUF file to disk.
    ///
    /// This serializes:
    /// 1. Header (magic + version + counts)
    /// 2. KV pairs section
    /// 3. Tensor metadata section
    /// 4. Data section (with alignment padding)
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), GgufError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let format = self.wire_format();

        // 1. Write magic (4 bytes)
        writer.write_all(b"GGUF")?;

        // 2. Write version (u32 LE)
        write_u32_le(&mut writer, self.version)?;

        // 3. Write tensor count and KV count (format-dependent widths)
        Self::write_int(&mut writer, self.tensors.len() as u64, format.header_count_width)?;
        Self::write_int(&mut writer, self.kv_pairs.len() as u64, format.header_count_width)?;

        // Calculate header sizes
        let kv_section_size = self.calculate_kv_section_size();
        let tensor_section_size = self.calculate_tensor_section_size();

        // 4. Write KV pairs section
        for kv in &self.kv_pairs {
            self.write_kv_pair(&mut writer, kv, &format)?;
        }

        // 5. Write tensor metadata section
        for tensor in &self.tensors {
            self.write_tensor_info(&mut writer, tensor, &format)?;
        }

        // Calculate data section start with alignment
        let count_bytes: u64 = match format.header_count_width {
            IntWidth::U32 => 4,
            IntWidth::U64 => 8,
        };
        let header_size = 4 + 4 + count_bytes + count_bytes + kv_section_size + tensor_section_size;
        let data_start = self.align_up(header_size, self.alignment);

        // 6. Write padding to data section start
        let current_pos = writer.stream_position()?;
        if current_pos < data_start {
            let padding = (data_start - current_pos) as usize;
            for _ in 0..padding {
                write_u8(&mut writer, 0)?;
            }
        }

        // 7. Validate tensor data matches tensor metadata
        for (name, _) in &self.tensor_data {
            if !self.tensors.iter().any(|t| t.name == *name) {
                return Err(GgufError::Io(format!(
                    "tensor data '{}' has no matching tensor metadata",
                    name
                )));
            }
        }

        // 8. Write tensor data (each aligned to alignment boundary)
        for (name, data) in &self.tensor_data {
            // Pad to alignment boundary before each tensor
            let current_pos = writer.stream_position()?;
            let aligned_pos = self.align_up(current_pos, self.alignment);
            if current_pos < aligned_pos {
                let padding = (aligned_pos - current_pos) as usize;
                writer.write_all(&vec![0u8; padding])?;
            }

            // Find corresponding tensor info to get dtype and shape
            if let Some(tensor) = self.tensors.iter().find(|t| t.name == *name) {
                let expected_size = tensor.stored_size()? as usize;
                if data.len() != expected_size {
                    return Err(GgufError::Io(format!(
                        "tensor '{}' data size mismatch: expected {}, got {}",
                        name,
                        expected_size,
                        data.len()
                    )));
                }
                writer.write_all(data)?;
            } else {
                // No tensor info found, write as-is
                writer.write_all(data)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Calculate the byte size of the KV section.
    fn calculate_kv_section_size(&self) -> u64 {
        let format = self.wire_format();
        self.kv_pairs
            .iter()
            .map(|p| p.raw_byte_size_for_format(&format) as u64)
            .sum()
    }

    /// Calculate the byte size of the tensor metadata section.
    fn calculate_tensor_section_size(&self) -> u64 {
        let format = self.wire_format();
        self.tensors.iter().map(|t| t.raw_byte_size_for_format(&format) as u64).sum()
    }

    /// Align a position up to the next alignment boundary.
    fn align_up(&self, pos: u64, alignment: u64) -> u64 {
        if alignment == 0 {
            return pos;
        }
        let remainder = pos % alignment;
        if remainder == 0 {
            pos
        } else {
            pos + alignment - remainder
        }
    }

    /// Write a KV pair using wire format to determine field widths.
    fn write_kv_pair(
        &self,
        writer: &mut BufWriter<File>,
        kv: &GgufKvPair,
        format: &GgufWireFormat,
    ) -> Result<(), GgufError> {
        // 1. Write key length (format-dependent width)
        let key_bytes = kv.key.as_bytes();
        Self::write_int(writer, key_bytes.len() as u64, format.key_width)?;

        // 2. Write key name
        writer.write_all(key_bytes)?;

        // 3. Write value type (u32 LE)
        write_u32_le(writer, kv.value_type.to_u32())?;

        // 4. Write value based on type
        self.write_kv_value(writer, &kv.value_type, &kv.value, format)?;

        Ok(())
    }

    /// Write a KV value using wire format to determine string/array count widths.
    fn write_kv_value(
        &self,
        writer: &mut BufWriter<File>,
        value_type: &GgufValueType,
        value: &GgufKvValue,
        format: &GgufWireFormat,
    ) -> Result<(), GgufError> {
        match value_type {
            GgufValueType::String => {
                if let GgufKvValue::String(s) = value {
                    let bytes = s.as_bytes();
                    Self::write_int(writer, bytes.len() as u64, format.string_width)?;
                    writer.write_all(bytes)?;
                }
            }
            GgufValueType::Array => {
                if let GgufKvValue::Array(arr) = value {
                    // Write element type
                    let elem_type = arr
                        .first()
                        .map(|v| v.value_type())
                        .unwrap_or(GgufValueType::Uint32);
                    write_u32_le(writer, elem_type.to_u32())?;

                    // Write element count (format-dependent width)
                    Self::write_int(writer, arr.len() as u64, format.array_count_width)?;

                    // Write elements
                    for elem in arr {
                        self.write_kv_value(writer, &elem_type, elem, format)?;
                    }
                }
            }
            // Scalar types are the same across all versions
            _ => self.write_kv_value_scalar(writer, value_type, value)?,
        }

        Ok(())
    }

    /// Write scalar KV value (shared across all GGUF versions).
    fn write_kv_value_scalar(
        &self,
        writer: &mut BufWriter<File>,
        value_type: &GgufValueType,
        value: &GgufKvValue,
    ) -> Result<(), GgufError> {
        match value_type {
            GgufValueType::Int8 => {
                if let GgufKvValue::Int8(v) = value {
                    write_i8(writer, *v)?;
                }
            }
            GgufValueType::Uint16 => {
                if let GgufKvValue::Uint16(v) = value {
                    write_u16_le(writer, *v)?;
                }
            }
            GgufValueType::Int16 => {
                if let GgufKvValue::Int16(v) = value {
                    write_i16_le(writer, *v)?;
                }
            }
            GgufValueType::Uint32 => {
                if let GgufKvValue::Uint32(v) = value {
                    write_u32_le(writer, *v)?;
                }
            }
            GgufValueType::Int32 => {
                if let GgufKvValue::Int32(v) = value {
                    write_i32_le(writer, *v)?;
                }
            }
            GgufValueType::Uint64 => {
                if let GgufKvValue::Uint64(v) = value {
                    write_u64_le(writer, *v)?;
                }
            }
            GgufValueType::Int64 => {
                if let GgufKvValue::Int64(v) = value {
                    write_i64_le(writer, *v)?;
                }
            }
            GgufValueType::Float32 => {
                if let GgufKvValue::Float32(v) = value {
                    write_f32_le(writer, *v)?;
                }
            }
            GgufValueType::Float64 => {
                if let GgufKvValue::Float64(v) = value {
                    write_f64_le(writer, *v)?;
                }
            }
            GgufValueType::Bool => {
                if let GgufKvValue::Bool(v) = value {
                    write_u8(writer, *v as u8)?;
                }
            }
            // Missing scalar types - add stub implementations
            GgufValueType::Uint8 => {
                if let GgufKvValue::Uint8(v) = value {
                    write_u8(writer, *v)?;
                }
            }
            GgufValueType::Bfloat16 => {
                return Err(GgufError::QuantizationNotSupported(
                    "Bfloat16 array element type not yet supported in writer".into(),
                ));
            }
            GgufValueType::Float16 => {
                return Err(GgufError::QuantizationNotSupported(
                    "Float16 array element type not yet supported in writer".into(),
                ));
            }
            GgufValueType::Int8Array | GgufValueType::Uint8Array => {
                return Err(GgufError::QuantizationNotSupported(
                    "Int8Array/Uint8Array not yet supported in writer".into(),
                ));
            }
            // These should be handled by the parent function, but Rust wants exhaustive match
            GgufValueType::String | GgufValueType::Array => {
                return Err(GgufError::QuantizationNotSupported(
                    "String/Array passed to scalar writer (should be handled by caller)".into(),
                ));
            }
        }

        Ok(())
    }

    /// Write tensor info using wire format to determine name length width.
    fn write_tensor_info(
        &self,
        writer: &mut BufWriter<File>,
        tensor: &GgufTensorInfo,
        format: &GgufWireFormat,
    ) -> Result<(), GgufError> {
        // 1. Write name length (format-dependent width)
        let name_bytes = tensor.name.as_bytes();
        Self::write_int(writer, name_bytes.len() as u64, format.tensor_name_width)?;

        // 2. Write name
        writer.write_all(name_bytes)?;

        // 3. Write number of dimensions (u32 LE) - same across all versions
        write_u32_le(writer, tensor.ndims())?;

        // 4. Write shape array (n_dims * u64 LE) - same across all versions
        for dim in &tensor.shape {
            write_u64_le(writer, *dim)?;
        }

        // 5. Write data type (u32 LE) - same across all versions
        write_u32_le(writer, tensor.dtype)?;

        // 6. Write offset (u64 LE) - same across all versions
        write_u64_le(writer, tensor.offset)?;

        Ok(())
    }
}

/// Helper function to parse and rewrite a GGUF file.
///
/// This is useful for:
/// - Normalizing alignment
/// - Updating metadata
/// - Converting between versions
pub fn parse_and_rewrite<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
) -> Result<(), GgufError> {
    use crate::parser::parse_gguf;

    // Parse the input file
    let header = parse_gguf(input_path.as_ref())?;

    // Create a new writer with the same metadata
    let mut writer = GgufWriter::new()
        .with_version(header.version)
        .with_alignment(header.data_alignment.unwrap_or(256));

    // Copy KV pairs
    for kv in &header.kv_pairs {
        writer.add_kv(kv.clone());
    }

    // Copy tensor metadata (need to clone since we'll iterate twice)
    let tensors_clone: Vec<_> = header.tensors.to_vec();

    // Read and copy tensor data
    // Each tensor's offset is relative to the start of the file (per GGUF spec),
    // so we seek to each tensor's actual position rather than reading sequentially.
    // This correctly handles alignment padding between tensors.
    let input_file = File::open(input_path.as_ref())?;
    let mut reader = std::io::BufReader::new(input_file);

    for tensor in &tensors_clone {
        let data_size = tensor.stored_size()? as usize;
        reader.seek(std::io::SeekFrom::Start(tensor.offset))?;
        let mut data = vec![0u8; data_size];
        reader.read_exact(&mut data)?;

        writer.add_tensor_data(&tensor.name, data);
    }

    // Write the new file
    writer.write(output_path.as_ref())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_gguf;
    use crate::types::GgufDtype;

    #[test]
    fn test_write_round_trip() {
        // Create a simple GGUF file
        let mut writer = GgufWriter::new();

        // Add metadata
        writer.add_kv(GgufKvPair {
            key: "general.architecture".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("llama".to_string()),
        });

        writer.add_kv(GgufKvPair {
            key: "general.quantization_version".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(2),
        });

        // Add a simple tensor (F32, 4 elements = 16 bytes)
        let tensor = GgufTensorInfo {
            name: "test.weight".to_string(),
            shape: vec![4],
            dtype: GgufDtype::F32.to_u32(),
            offset: 0,
        };
        writer.add_tensor(tensor);

        // Add F32 data (4 elements * 4 bytes = 16 bytes)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let raw_data: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.add_tensor_data("test.weight", raw_data);

        // Write to temp file
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_write.gguf");

        writer.write(&output_path).expect("Failed to write GGUF");

        // Read back
        let header = parse_gguf(&output_path).expect("Failed to parse written GGUF");

        // Verify metadata
        assert_eq!(header.kv_pairs.len(), 2);
        assert_eq!(header.tensors.len(), 1);

        // Clean up
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_write_with_alignment() {
        let mut writer = GgufWriter::new().with_alignment(4096);

        writer.add_kv(GgufKvPair {
            key: "general.architecture".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("llama".to_string()),
        });

        let tensor = GgufTensorInfo {
            name: "test_tensor".to_string(),
            shape: vec![10],
            dtype: GgufDtype::F32.to_u32(),
            offset: 0,
        };
        writer.add_tensor(tensor);

        let data: Vec<f32> = (0..10).map(|v| v as f32).collect();
        let raw_data: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.add_tensor_data("test_tensor", raw_data);

        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_alignment.gguf");

        writer
            .write(&output_path)
            .expect("Failed to write with alignment");

        // Verify file exists and has content
        let metadata = std::fs::metadata(&output_path).expect("Failed to get metadata");
        assert!(metadata.len() > 0);

        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_round_trip_full_model() -> Result<(), GgufError> {
        // Simulate a complete model write/read cycle
        let mut writer = GgufWriter::new();

        // Add comprehensive metadata
        writer.add_kv(GgufKvPair {
            key: "general.architecture".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("qwen2".to_string()),
        });

        writer.add_kv(GgufKvPair {
            key: "general.quantization_version".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(2),
        });

        writer.add_kv(GgufKvPair {
            key: "general.type".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("q4_k_m".to_string()),
        });

        writer.add_kv(GgufKvPair {
            key: "general.file_type".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(15), // Q4_K_M
        });

        writer.add_kv(GgufKvPair {
            key: "qwen2.block_count".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(32),
        });

        writer.add_kv(GgufKvPair {
            key: "qwen2.context_length".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(4096),
        });

        writer.add_kv(GgufKvPair {
            key: "qwen2.embedding_length".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(4096),
        });

        writer.add_kv(GgufKvPair {
            key: "qwen2.feed_forward_length".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(11008),
        });

        writer.add_kv(GgufKvPair {
            key: "qwen2.attention.head_count".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(32),
        });

        writer.add_kv(GgufKvPair {
            key: "qwen2.attention.head_count_kv".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(8),
        });

        writer.add_kv(GgufKvPair {
            key: "qwen2.rope.freq_base".to_string(),
            value_type: GgufValueType::Float32,
            value: GgufKvValue::Float32(10000.0),
        });

        writer.add_kv(GgufKvPair {
            key: "qwen2.attention.layer_norm_rms_epsilon".to_string(),
            value_type: GgufValueType::Float32,
            value: GgufKvValue::Float32(1e-5),
        });

        writer.add_kv(GgufKvPair {
            key: "tokenizer.ggml.model".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("qwen2".to_string()),
        });

        writer.add_kv(GgufKvPair {
            key: "tokenizer.ggml.tokens".to_string(),
            value_type: GgufValueType::Array,
            value: GgufKvValue::Array(vec![
                GgufKvValue::String("[PAD]".to_string()),
                GgufKvValue::String("[UNK]".to_string()),
                GgufKvValue::String("[CLS]".to_string()),
                GgufKvValue::String("[SEP]".to_string()),
                GgufKvValue::String("[MASK]".to_string()),
            ]),
        });

        // Add multiple tensors
        let embedding_tensor = GgufTensorInfo {
            name: "token_embd.weight".to_string(),
            shape: vec![4096, 32000],
            dtype: GgufDtype::Q4_0.to_u32(),
            offset: 0,
        };
        writer.add_tensor(embedding_tensor);

        let output_tensor = GgufTensorInfo {
            name: "output.weight".to_string(),
            shape: vec![32000, 4096],
            dtype: GgufDtype::F32.to_u32(),
            offset: 0,
        };
        writer.add_tensor(output_tensor);

        let attention_norm = GgufTensorInfo {
            name: "blk.0.attn_norm.weight".to_string(),
            shape: vec![4096],
            dtype: GgufDtype::F32.to_u32(),
            offset: 0,
        };
        writer.add_tensor(attention_norm);

        let ffn_norm = GgufTensorInfo {
            name: "blk.0.ffn_norm.weight".to_string(),
            shape: vec![4096],
            dtype: GgufDtype::F32.to_u32(),
            offset: 0,
        };
        writer.add_tensor(ffn_norm);

        let wq = GgufTensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: GgufDtype::Q4_0.to_u32(),
            offset: 0,
        };
        writer.add_tensor(wq);

        let wk = GgufTensorInfo {
            name: "blk.0.attn_k.weight".to_string(),
            shape: vec![1024, 4096],
            dtype: GgufDtype::Q4_0.to_u32(),
            offset: 0,
        };
        writer.add_tensor(wk);

        let wv = GgufTensorInfo {
            name: "blk.0.attn_v.weight".to_string(),
            shape: vec![1024, 4096],
            dtype: GgufDtype::Q4_0.to_u32(),
            offset: 0,
        };
        writer.add_tensor(wv);

        let wo = GgufTensorInfo {
            name: "blk.0.attn_output.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: GgufDtype::Q4_0.to_u32(),
            offset: 0,
        };
        writer.add_tensor(wo);

        let w1 = GgufTensorInfo {
            name: "blk.0.ffn_gate.weight".to_string(),
            shape: vec![11008, 4096],
            dtype: GgufDtype::Q4_0.to_u32(),
            offset: 0,
        };
        writer.add_tensor(w1);

        let w2 = GgufTensorInfo {
            name: "blk.0.ffn_down.weight".to_string(),
            shape: vec![4096, 11008],
            dtype: GgufDtype::Q4_0.to_u32(),
            offset: 0,
        };
        writer.add_tensor(w2);

        let w3 = GgufTensorInfo {
            name: "blk.0.ffn_up.weight".to_string(),
            shape: vec![11008, 4096],
            dtype: GgufDtype::Q4_0.to_u32(),
            offset: 0,
        };
        writer.add_tensor(w3);

        // Add tensor data (simulated quantized data - just zeros for round-trip test)
        // Use the actual stored_size() method to get correct sizes
        let tensor_names: Vec<String> = writer.tensors.iter().map(|t| t.name.clone()).collect();
        
        for name in &tensor_names {
            if let Some(tensor) = writer.tensors.iter().find(|t| t.name == *name) {
                let data_size = tensor.stored_size()? as usize;
                let data = vec![0u8; data_size];
                writer.add_tensor_data(name, data);
            }
        }

        // Write to temp file
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_round_trip_full.gguf");

        writer.write(&output_path).expect("Failed to write full model GGUF");

        // Read back and verify
        let header = parse_gguf(&output_path).expect("Failed to parse written GGUF");

        // Verify metadata
        assert_eq!(header.kv_pairs.len(), 14, "Should have 14 KV pairs");
        assert_eq!(header.tensors.len(), 11, "Should have 11 tensors");

        // Verify specific KV pairs exist
        let kv_map: std::collections::HashMap<&str, &GgufKvValue> = header
            .kv_pairs
            .iter()
            .map(|p| (p.key.as_str(), &p.value))
            .collect();

        assert_eq!(
            kv_map.get("general.architecture"),
            Some(&GgufKvValue::String("qwen2".to_string())).as_ref()
        );
        assert_eq!(
            kv_map.get("general.quantization_version"),
            Some(&GgufKvValue::Uint32(2)).as_ref()
        );
        assert_eq!(
            kv_map.get("qwen2.block_count"),
            Some(&GgufKvValue::Uint32(32)).as_ref()
        );

        // Verify tensor names
        let tensor_names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(tensor_names.contains(&"token_embd.weight"));
        assert!(tensor_names.contains(&"output.weight"));
        assert!(tensor_names.contains(&"blk.0.attn_q.weight"));

        // Clean up
        let _ = std::fs::remove_file(output_path);
        Ok(())
    }
}
