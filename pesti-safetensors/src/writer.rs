use byteorder::WriteBytesExt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::SafetensorsSchemaError;

/// SafeTensors file writer for serializing model weights.
///
/// Implements the SafeTensors format: https://github.com/huggingface/safetensors
pub struct SafetensorsWriter {
    tensors: Vec<SafetensorTensor>,
}

/// A single tensor to be written.
#[derive(Debug, Clone)]
pub struct SafetensorTensor {
    /// Tensor name
    pub name: String,
    /// Data type (F32, F16, Q4_0, etc.)
    pub dtype: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Raw byte data
    pub data: Vec<u8>,
}

impl SafetensorsWriter {
    /// Create a new SafeTensors writer.
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    /// Add a tensor to the writer.
    pub fn add_tensor(&mut self, tensor: SafetensorTensor) {
        self.tensors.push(tensor);
    }

    /// Write the SafeTensors file to disk.
    ///
    /// Format:
    /// - Header length (u64 LE)
    /// - JSON header with metadata
    /// - Padding to 8-byte alignment
    /// - Tensor data (raw bytes, no headers per tensor)
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), SafetensorsSchemaError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Build JSON header
        let header_json = self.build_header_json()?;

        // Calculate sizes
        let header_len = header_json.len() as u64;
        let total_header = header_len + 8; // Header len + JSON
        let padding = (8 - (total_header % 8)) % 8;
        let final_header_len = total_header + padding;

        // 1. Write header length (u64 LE)
        writer.write_all(&final_header_len.to_le_bytes())?;

        // 2. Write JSON header
        writer.write_all(header_json.as_bytes())?;

        // 3. Write padding
        for _ in 0..padding {
            writer.write_u8(0)?;
        }

        // 4. Write tensor data
        for tensor in &self.tensors {
            writer.write_all(&tensor.data)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Build the JSON header string.
    fn build_header_json(&self) -> Result<String, SafetensorsSchemaError> {
        use serde_json::json;

        let mut tensors_obj = serde_json::Map::new();

        // Add metadata (common fields)
        tensors_obj.insert(
            "__metadata__".to_string(),
            json!({
                "pt_type": "safetensors",
                "format": "pt"
            }),
        );

        // Add each tensor with sequential offsets
        let mut current_offset: u64 = 8 + 8; // Header length field + JSON header placeholder

        for tensor in &self.tensors {
            let mut tensor_info = serde_json::Map::new();

            // Shape
            tensor_info.insert(
                "shape".to_string(),
                json!(tensor.shape.iter().map(|&s| s as i64).collect::<Vec<i64>>()),
            );

            // Data type
            tensor_info.insert("dtype".to_string(), json!(tensor.dtype));

            // Byte offset and size
            let element_size = Self::dtype_element_size(&tensor.dtype)?;
            let num_elements: usize = tensor.shape.iter().product();
            let size = (num_elements * element_size) as u64;

            tensor_info.insert("data_offsets".to_string(), json!([current_offset, current_offset + size]));

            tensors_obj.insert(tensor.name.clone(), json!(tensor_info));

            // Update offset for next tensor
            current_offset += size;
        }

        Ok(serde_json::to_string(&tensors_obj).map_err(SafetensorsSchemaError::Json)?)
    }

    /// Get the element size in bytes for a given dtype.
    fn dtype_element_size(dtype: &str) -> Result<usize, SafetensorsSchemaError> {
        match dtype {
            "F32" => Ok(4),
            "F16" | "BF16" => Ok(2),
            "I8" | "U8" => Ok(1),
            "I16" | "U16" => Ok(2),
            "I32" | "U32" => Ok(4),
            "I64" | "U64" => Ok(8),
            "Q4_0" | "Q4_1" | "Q5_0" | "Q5_1" | "Q8_0" | "Q8_1" => {
                // Quantized types - approximate
                Ok(2)
            }
            _ => Err(SafetensorsSchemaError::Internal(format!(
                "Unknown dtype: {}",
                dtype
            ))),
        }
    }
}

/// Helper function to convert GGUF tensors to SafeTensors format.
pub fn gguf_to_safetensors(
    writer: &mut SafetensorsWriter,
    gguf_tensor_name: &str,
    data: &[u8],
    dtype: &str,
    shape: &[usize],
) -> Result<(), SafetensorsSchemaError> {
    let tensor = SafetensorTensor {
        name: gguf_tensor_name.to_string(),
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
        data: data.to_vec(),
    };
    writer.add_tensor(tensor);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_simple() {
        let mut writer = SafetensorsWriter::new();

        // Add a simple F32 tensor
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = SafetensorTensor {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4],
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        writer.add_tensor(tensor);

        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test.safetensors");

        writer.write(&output_path).expect("Failed to write SafeTensors");

        // Verify file exists and has content
        let metadata = std::fs::metadata(&output_path).expect("Failed to get metadata");
        assert!(metadata.len() > 0);

        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_write_multiple_tensors() {
        let mut writer = SafetensorsWriter::new();

        // Add multiple tensors
        let data1: Vec<f32> = vec![1.0, 2.0];
        let tensor1 = SafetensorTensor {
            name: "weight1".to_string(),
            dtype: "F32".to_string(),
            shape: vec![2],
            data: data1.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        writer.add_tensor(tensor1);

        let data2: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5];
        let tensor2 = SafetensorTensor {
            name: "weight2".to_string(),
            dtype: "F16".to_string(),
            shape: vec![4],
            data: data2.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        writer.add_tensor(tensor2);

        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_multi.safetensors");

        writer.write(&output_path).expect("Failed to write SafeTensors");

        let metadata = std::fs::metadata(&output_path).expect("Failed to get metadata");
        assert!(metadata.len() > 0);

        let _ = std::fs::remove_file(output_path);
    }
}
