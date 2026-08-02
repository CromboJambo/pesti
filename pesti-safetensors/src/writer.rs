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

    #[test]
    fn test_round_trip_full_model() {
        use std::collections::HashMap;

        let mut writer = SafetensorsWriter::new();

        // Simulate a complete model with multiple layers
        let vocab_size: usize = 32000;
        let embed_dim: usize = 4096;
        let num_layers: usize = 32;

        // Token embedding (F32)
        let embedding_data: Vec<f32> = (0..embed_dim * vocab_size)
            .map(|i| i as f32 * 0.001)
            .collect();
        let tensor = SafetensorTensor {
            name: "model.embed_tokens.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![vocab_size, embed_dim],
            data: embedding_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        writer.add_tensor(tensor);

        // Output head (F32)
        let output_data: Vec<f32> = (0..embed_dim * vocab_size)
            .map(|i| i as f32 * 0.002)
            .collect();
        let tensor = SafetensorTensor {
            name: "model.norm.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![embed_dim],
            data: output_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        writer.add_tensor(tensor);

        // Transformer layers
        for layer_idx in 0..num_layers {
            let prefix = format!("model.layers.{}", layer_idx);

            // Attention norm (F32)
            let attn_norm_data: Vec<f32> = (0..embed_dim).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.input_layernorm.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim],
                data: attn_norm_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);

            // Q projection (F32)
            let q_proj_data: Vec<f32> =
                (0..embed_dim * embed_dim).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.self_attn.q_proj.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim, embed_dim],
                data: q_proj_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);

            // K projection (F32)
            let k_proj_data: Vec<f32> =
                (0..embed_dim * embed_dim / 4).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.self_attn.k_proj.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim / 4, embed_dim],
                data: k_proj_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);

            // V projection (F32)
            let v_proj_data: Vec<f32> =
                (0..embed_dim * embed_dim / 4).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.self_attn.v_proj.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim / 4, embed_dim],
                data: v_proj_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);

            // O projection (F32)
            let o_proj_data: Vec<f32> =
                (0..embed_dim * embed_dim).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.self_attn.o_proj.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim, embed_dim],
                data: o_proj_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);

            // FFN norm (F32)
            let ffn_norm_data: Vec<f32> = (0..embed_dim).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.post_attention_layernorm.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim],
                data: ffn_norm_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);

            // FFN up (F32)
            let ffn_up_data: Vec<f32> =
                (0..embed_dim * embed_dim * 2).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.mlp.up_proj.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim * 2, embed_dim],
                data: ffn_up_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);

            // FFN gate (F32)
            let ffn_gate_data: Vec<f32> =
                (0..embed_dim * embed_dim * 2).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.mlp.gate_proj.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim * 2, embed_dim],
                data: ffn_gate_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);

            // FFN down (F32)
            let ffn_down_data: Vec<f32> =
                (0..embed_dim * embed_dim * 2).map(|i| i as f32 * 0.001).collect();
            let tensor = SafetensorTensor {
                name: format!("{}.mlp.down_proj.weight", prefix),
                dtype: "F32".to_string(),
                shape: vec![embed_dim, embed_dim * 2],
                data: ffn_down_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            writer.add_tensor(tensor);
        }

        // Write to temp file
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_round_trip_full.safetensors");

        writer.write(&output_path).expect("Failed to write full model SafeTensors");

        // Verify file exists and has content
        let metadata = std::fs::metadata(&output_path).expect("Failed to get metadata");
        assert!(metadata.len() > 100_000, "File should be large (32 layers)");

        // Read back and verify header
        let content = std::fs::read(&output_path).expect("Failed to read file");
        let header_len = u64::from_le_bytes([
            content[0], content[1], content[2], content[3], content[4], content[5], content[6],
            content[7],
        ]) as usize;
        let header_json = String::from_utf8_lossy(&content[8..8 + header_len]);

        // Verify JSON contains expected keys
        assert!(header_json.contains("model.embed_tokens.weight"));
        assert!(header_json.contains("model.layers.0.self_attn.q_proj.weight"));
        assert!(header_json.contains(&format!("model.layers.{}.mlp.down_proj.weight", num_layers - 1)));

        // Count tensors in header
        let tensor_count = header_json.matches("shape").count();
        // 1 embed + 1 norm + 32 layers * (attn_norm + q + k + v + o + ffn_norm + up + gate + down) = 2 + 32*9 = 290
        assert_eq!(tensor_count, 290, "Should have 290 tensors (1 embed + 1 norm + 32*9 layer tensors)");

        // Clean up
        let _ = std::fs::remove_file(output_path);
    }
}
