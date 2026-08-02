use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, Write};
use std::path::Path;

use crate::error::GgufError;
use crate::types::{GgufDtype, GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType};

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

        // 1. Write magic (4 bytes)
        writer.write_all(b"GGUF")?;

        // 2. Write version (u64 LE)
        writer.write_u32::<LittleEndian>(self.version)?;

        // 3. Write tensor count and KV count (u64 LE each)
        writer.write_u64::<LittleEndian>(self.tensors.len() as u64)?;
        writer.write_u64::<LittleEndian>(self.kv_pairs.len() as u64)?;

        // Calculate header sizes
        let kv_section_size = self.calculate_kv_section_size();
        let tensor_section_size = self.calculate_tensor_section_size();

        // 4. Write KV pairs section
        for kv in &self.kv_pairs {
            self.write_kv_pair_v3(&mut writer, kv)?;
        }

        // 5. Write tensor metadata section
        for tensor in &self.tensors {
            self.write_tensor_info_v3(&mut writer, tensor)?;
        }

        // Calculate data section start with alignment
        let header_size = 4 + 4 + 8 + 8 + kv_section_size + tensor_section_size;
        let data_start = self.align_up(header_size, self.alignment);

        // 6. Write padding to data section start
        let current_pos = writer.stream_position()?;
        if current_pos < data_start {
            let padding = (data_start - current_pos) as usize;
            for _ in 0..padding {
                writer.write_u8(0)?;
            }
        }

        // 7. Write tensor data
        for (name, data) in &self.tensor_data {
            // Find corresponding tensor info to get dtype and shape
            if let Some(tensor) = self.tensors.iter().find(|t| t.name == *name) {
                let expected_size = tensor.stored_size() as usize;
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
        self.kv_pairs
            .iter()
            .map(|p| p.raw_byte_size_v3() as u64)
            .sum()
    }

    /// Calculate the byte size of the tensor metadata section.
    fn calculate_tensor_section_size(&self) -> u64 {
        self.tensors.iter().map(|t| t.raw_byte_size() as u64).sum()
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

    /// Write a KV pair in v3 format.
    fn write_kv_pair_v3(
        &self,
        writer: &mut BufWriter<File>,
        kv: &GgufKvPair,
    ) -> Result<(), GgufError> {
        // 1. Write key length (u64 LE)
        let key_bytes = kv.key.as_bytes();
        writer.write_u64::<LittleEndian>(key_bytes.len() as u64)?;

        // 2. Write key name
        writer.write_all(key_bytes)?;

        // 3. Write value type (u32 LE)
        writer.write_u32::<LittleEndian>(kv.value_type.to_u32())?;

        // 4. Write value based on type
        self.write_kv_value_v3(writer, &kv.value_type, &kv.value)?;

        Ok(())
    }

    /// Write a KV value in v3 format.
    fn write_kv_value_v3(
        &self,
        writer: &mut BufWriter<File>,
        value_type: &GgufValueType,
        value: &GgufKvValue,
    ) -> Result<(), GgufError> {
        match value_type {
            GgufValueType::String => {
                if let GgufKvValue::String(s) = value {
                    let bytes = s.as_bytes();
                    writer.write_u64::<LittleEndian>(bytes.len() as u64)?;
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
                    writer.write_u32::<LittleEndian>(elem_type.to_u32())?;

                    // Write element count
                    writer.write_u64::<LittleEndian>(arr.len() as u64)?;

                    // Write elements
                    for elem in arr {
                        self.write_kv_value_v3(writer, &elem_type, elem)?;
                    }
                }
            }
            GgufValueType::Int8 => {
                if let GgufKvValue::Int8(v) = value {
                    writer.write_i8(*v)?;
                }
            }
            GgufValueType::Uint16 => {
                if let GgufKvValue::Uint16(v) = value {
                    writer.write_u16::<LittleEndian>(*v)?;
                }
            }
            GgufValueType::Int16 => {
                if let GgufKvValue::Int16(v) = value {
                    writer.write_i16::<LittleEndian>(*v)?;
                }
            }
            GgufValueType::Uint32 => {
                if let GgufKvValue::Uint32(v) = value {
                    writer.write_u32::<LittleEndian>(*v)?;
                }
            }
            GgufValueType::Int32 => {
                if let GgufKvValue::Int32(v) = value {
                    writer.write_i32::<LittleEndian>(*v)?;
                }
            }
            GgufValueType::Uint64 => {
                if let GgufKvValue::Uint64(v) = value {
                    writer.write_u64::<LittleEndian>(*v)?;
                }
            }
            GgufValueType::Int64 => {
                if let GgufKvValue::Int64(v) = value {
                    writer.write_i64::<LittleEndian>(*v)?;
                }
            }
            GgufValueType::Float32 => {
                if let GgufKvValue::Float32(v) = value {
                    writer.write_f32::<LittleEndian>(*v)?;
                }
            }
            GgufValueType::Float64 => {
                if let GgufKvValue::Float64(v) = value {
                    writer.write_f64::<LittleEndian>(*v)?;
                }
            }
            GgufValueType::Bool => {
                if let GgufKvValue::Bool(v) = value {
                    writer.write_u8(*v as u8)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Write tensor info in v3 format.
    fn write_tensor_info_v3(
        &self,
        writer: &mut BufWriter<File>,
        tensor: &GgufTensorInfo,
    ) -> Result<(), GgufError> {
        // 1. Write name length (u64 LE)
        let name_bytes = tensor.name.as_bytes();
        writer.write_u64::<LittleEndian>(name_bytes.len() as u64)?;

        // 2. Write name
        writer.write_all(name_bytes)?;

        // 3. Write number of dimensions (u32 LE)
        writer.write_u32::<LittleEndian>(tensor.ndims())?;

        // 4. Write shape array (n_dims * u64 LE)
        for dim in &tensor.shape {
            writer.write_u64::<LittleEndian>(*dim)?;
        }

        // 5. Write data type (u32 LE)
        writer.write_u32::<LittleEndian>(tensor.dtype)?;

        // 6. Write offset (u64 LE)
        writer.write_u64::<LittleEndian>(tensor.offset)?;

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
    let tensors_clone: Vec<_> = header.tensors.iter().cloned().collect();

    // Read and copy tensor data
    let input_file = File::open(input_path.as_ref())?;
    let mut reader = std::io::BufReader::new(input_file);

    // Seek to data section start
    reader.seek(std::io::SeekFrom::Start(header.data_section_start))?;

    for tensor in &tensors_clone {
        let data_size = tensor.stored_size() as usize;
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
}
