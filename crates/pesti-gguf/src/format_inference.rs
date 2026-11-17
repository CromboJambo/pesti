//! Format inference engine for pesti-gguf
//!
//! Detects actual quantization format from raw GGUF tensor data,
//! validates metadata consistency, and provides warnings for suspicious assignments.

use crate::error::GgufError;
use crate::types::GgufDtype;

/// Inferred quantization format with confidence score
#[derive(Debug, Clone, PartialEq)]
pub struct InferredFormat {
    /// Detected dtype
    pub dtype: GgufDtype,
    /// Bytes per element (average)
    pub bytes_per_element: f32,
    /// Elements per block
    pub elements_per_block: usize,
    /// Bytes per block
    pub bytes_per_block: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

impl InferredFormat {
    pub fn new(
        dtype: GgufDtype,
        bytes_per_element: f32,
        elements_per_block: usize,
        bytes_per_block: usize,
        confidence: f32,
    ) -> Self {
        Self {
            dtype,
            bytes_per_element,
            elements_per_block,
            bytes_per_block,
            confidence,
        }
    }
}

/// Warning types for metadata inconsistencies
#[derive(Debug, Clone, PartialEq)]
pub enum Warning {
    /// Dtype mismatch: claimed format doesn't match inferred format
    DtypeMismatch {
        tensor_name: String,
        claimed: GgufDtype,
        inferred: GgufDtype,
        confidence: f32,
    },
    /// Suspicious embedding/output layer format (e.g., F32 where quantized expected)
    SuspiciousEmbeddingFormat {
        tensor_name: String,
        suggested: GgufDtype,
    },
}

/// Known format specifications (verified against llama.cpp's `ggml_type_traits`)
///
/// Format  BPE*    E/PB**  B/PB*  Notes
/// ──────  ──────  ──────  ─────  ────────────────────
/// Q4_0    0.5625  32      18     16B qs + 2B d
/// Q4_K    0.5625  256     144    K-quant superblock
/// Q5_0    0.6875  32      22     16B qs + 8B qh + 2B d
/// Q5_K    0.6875  256     176    K-quant superblock
/// Q6_K    0.8203  256     210    K-quant superblock
/// Q8_0    1.0625  32      34     32B qs + 2B d
///
/// ** BPE = bytes per element (B/PB ÷ E/PB)
/// ** E/PB = elements per block, B/PB = bytes per block
///
/// Format specification table
const FORMAT_SPECS: [(GgufDtype, f32, usize, usize); 6] = [
    (GgufDtype::Q4_0, 0.5625, 32, 18),
    (GgufDtype::Q4_K, 0.5625, 256, 144),
    (GgufDtype::Q5_0, 0.6875, 32, 22),
    (GgufDtype::Q5_K, 0.6875, 256, 176),
    (GgufDtype::Q6_K, 0.8203125, 256, 210),
    (GgufDtype::Q8_0, 1.0625, 32, 34),
];

/// Infer actual quantization format from GGUF tensor data
///
/// # Arguments
/// - `claimed_dtype`: The dtype claimed in GGUF header
/// - `claimed_elements`: Element count from GGUF header
/// - `raw_data`: Raw quantized bytes from file
///
/// # Returns
/// Sorted list of inferred formats by confidence (best match first)
pub fn infer_tensor_format(
    _claimed_dtype: GgufDtype,
    claimed_elements: usize,
    raw_data: &[u8],
) -> Result<Vec<InferredFormat>, GgufError> {
    if raw_data.is_empty() {
        return Ok(vec![]);
    }

    let data_size = raw_data.len();
    let mut candidates: Vec<InferredFormat> = Vec::new();

    // Calculate actual bytes per element
    let actual_bpe = data_size as f32 / claimed_elements as f32;

    // Check each known format
    for &(dtype, _spec_bpe, elements_per_block, bytes_per_block) in &FORMAT_SPECS {
        // Skip non-quantized types
        if !dtype.is_quantized() {
            continue;
        }

        // Calculate expected data size for this format
        let num_blocks = claimed_elements.div_ceil(elements_per_block);
        let expected_size = num_blocks * bytes_per_block;

        // Calculate confidence based on size match
        let size_diff = (data_size as i32 - expected_size as i32).abs() as f32;
        let confidence = if size_diff == 0.0 {
            1.0
        } else {
            // Decrease confidence based on relative size difference
            // Use expected_size as baseline for percentage calculation
            let pct_error = size_diff / expected_size as f32;
            (1.0 - pct_error).max(0.0)
        };

        // Only include candidates with reasonable confidence (>50%)
        if confidence > 0.5 {
            let inferred_bpe = actual_bpe;
            candidates.push(InferredFormat::new(
                dtype,
                inferred_bpe,
                elements_per_block,
                bytes_per_block,
                confidence,
            ));
        }
    }

    // Sort by confidence descending
    candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    Ok(candidates)
}

/// Validate tensor metadata and return warnings for inconsistencies
///
/// # Arguments
/// - `name`: Tensor name (for warning messages)
/// - `dtype`: The dtype claimed in GGUF header
/// - `shape`: Shape of the tensor
/// - `raw_data`: Raw quantized bytes from file
///
/// # Returns
/// List of warnings for suspicious assignments
pub fn validate_tensor_metadata(
    name: &str,
    dtype: GgufDtype,
    shape: &[u64],
    raw_data: &[u8],
) -> Result<Vec<Warning>, GgufError> {
    let mut warnings = Vec::new();

    // Calculate element count from shape
    let claimed_elements = shape.iter().product::<u64>() as usize;

    // Skip validation for empty tensors
    if raw_data.is_empty() || claimed_elements == 0 {
        return Ok(vec![]);
    }

    // Infer possible formats
    let inferred = infer_tensor_format(dtype, claimed_elements, raw_data)?;

    // Check for dtype mismatches with high confidence (>80%)
    for candidate in &inferred {
        if candidate.dtype != dtype && candidate.confidence > 0.8 {
            warnings.push(Warning::DtypeMismatch {
                tensor_name: name.to_string(),
                claimed: dtype,
                inferred: candidate.dtype,
                confidence: candidate.confidence,
            });
        }
    }

    // Check for suspicious embedding/output layer formats
    if is_suspicious_layer(name) {
        // Embedding layers should typically be F32 or F16, not quantized
        if dtype.is_quantized() && !matches!(dtype, GgufDtype::Q4_0 | GgufDtype::Q4_K) {
            warnings.push(Warning::SuspiciousEmbeddingFormat {
                tensor_name: name.to_string(),
                suggested: GgufDtype::F16,
            });
        }

        // Output layers should typically be F32 or F16
        if name.contains("output") && dtype.is_quantized() {
            warnings.push(Warning::SuspiciousEmbeddingFormat {
                tensor_name: name.to_string(),
                suggested: GgufDtype::F32,
            });
        }
    }

    Ok(warnings)
}

/// Check if tensor name suggests it's a special layer (embedding/output)
fn is_suspicious_layer(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("token_embd")
        || lower.contains("embeddings")
        || lower.contains("output")
        || lower.contains("lm_head")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_k_correct() {
        // Q4_K: 256 elements, 144 bytes (1 superblock).
        // Note: Q4_0 shares the same bytes-per-element (0.5625), so for 256
        // elements both predict exactly 144 bytes — a genuine tie. Q4_K must be
        // a top-confidence candidate, not necessarily the unique top.
        let raw_data = vec![0u8; 144];
        let inferred = infer_tensor_format(GgufDtype::Q4_K, 256, &raw_data).unwrap();

        assert!(!inferred.is_empty());
        let q4k = inferred.iter().find(|f| f.dtype == GgufDtype::Q4_K).unwrap();
        assert!(q4k.confidence > 0.95);
    }

    #[test]
    fn test_q4_0_data_claimed_as_q4_k() {
        // Q4_0: 32 elements, 18 bytes (1 block)
        let raw_data = vec![0u8; 18];
        let inferred = infer_tensor_format(GgufDtype::Q4_K, 32, &raw_data).unwrap();

        assert!(inferred.iter().any(|f| f.dtype == GgufDtype::Q4_0));
    }

    #[test]
    fn test_token_embd_mismatch() {
        // Simulate a tensor that claims Q4_K but whose data size matches Q5_K.
        // 256 elements: Q4_K stores 144 B, Q5_K stores 176 B.
        let raw_data = vec![0u8; 176];
        let inferred = infer_tensor_format(GgufDtype::Q4_K, 256, &raw_data).unwrap();

        // Should detect Q5_K format (BPE=0.6875) vs claimed Q4_K (BPE=0.5625)
        assert!(inferred.iter().any(|f| f.dtype == GgufDtype::Q5_K));
    }

    #[test]
    fn test_validate_dtype_mismatch() {
        // Create a tensor that claims Q4_K but is actually Q4_0 sized
        let raw_data = vec![0u8; 18]; // Q4_0 size for 32 elements
        let warnings =
            validate_tensor_metadata("test_tensor", GgufDtype::Q4_K, &[32], &raw_data).unwrap();

        assert!(!warnings.is_empty());
        match &warnings[0] {
            Warning::DtypeMismatch {
                inferred,
                confidence,
                ..
            } => {
                assert_eq!(*inferred, GgufDtype::Q4_0);
                assert!(*confidence > 0.8);
            }
            _ => panic!("Expected DtypeMismatch warning"),
        }
    }

    #[test]
    fn test_suspicious_embedding() {
        // Embedding layer with unusual quantization
        let raw_data = vec![0u8; 128];
        let warnings =
            validate_tensor_metadata("token_embd.weight", GgufDtype::Q5_K, &[32], &raw_data)
                .unwrap();

        assert!(warnings.iter().any(|w| matches!(
            w,
            Warning::SuspiciousEmbeddingFormat {
                suggested: GgufDtype::F16,
                ..
            }
        )));
    }

    #[test]
    fn test_empty_data() {
        let inferred = infer_tensor_format(GgufDtype::Q4_K, 32, &[]).unwrap();
        assert!(inferred.is_empty());

        let warnings = validate_tensor_metadata("test", GgufDtype::Q4_K, &[32], &[]).unwrap();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_q5_k_format() {
        // Q5_K: 256 elements, 176 bytes (1 superblock).
        // Note: Q5_0 shares the same bytes-per-element (0.6875), so for 256
        // elements both predict exactly 176 bytes — a genuine tie.
        let raw_data = vec![0u8; 176];
        let inferred = infer_tensor_format(GgufDtype::Q5_K, 256, &raw_data).unwrap();

        assert!(!inferred.is_empty());
        let q5k = inferred.iter().find(|f| f.dtype == GgufDtype::Q5_K).unwrap();
        assert!(q5k.confidence > 0.95);
    }

    #[test]
    fn test_q8_0_format() {
        // Q8_0: 32 elements, 34 bytes (1 block with 2B scale)
        let raw_data = vec![0u8; 34];
        let inferred = infer_tensor_format(GgufDtype::Q8_0, 32, &raw_data).unwrap();

        assert!(!inferred.is_empty());
        assert!(inferred.iter().any(|f| f.dtype == GgufDtype::Q8_0));
        assert!(inferred[0].confidence > 0.95);
    }
}
