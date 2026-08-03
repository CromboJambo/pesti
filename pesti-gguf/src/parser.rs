use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek};
use std::path::Path;

use crate::error::GgufError;
use crate::types::{GgufHeader, GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION_1: u32 = 1;
const GGUF_VERSION_2: u32 = 2;
const GGUF_VERSION_3: u32 = 3;

pub fn parse_gguf(path: &Path) -> Result<GgufHeader, GgufError> {
    let file = std::fs::File::open(path)
        .map_err(|e| GgufError::Io(format!("open {}: {}", path.display(), e)))?;
    let reader = &mut (file as std::fs::File);
    parse_gguf_reader(reader)
}

pub fn parse_gguf_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let magic = read_bytes(reader, 4)?;
    if magic.as_slice() != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(
            String::from_utf8_lossy(&magic).to_string(),
        ));
    }

    let version = reader.read_u32::<LittleEndian>()?;

    match version {
        GGUF_VERSION_1 => parse_v1(reader),
        GGUF_VERSION_2 => parse_v2(reader),
        GGUF_VERSION_3 => parse_v3(reader),
        _ => Err(GgufError::UnsupportedVersion(version)),
    }
}

fn parse_v1<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let kv_count = reader.read_u64::<LittleEndian>()?;

    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair(reader)?);
    }

    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(1, &kv_pairs, &[], alignment);

    Ok(GgufHeader {
        version: 1,
        kv_pairs,
        tensors: vec![],
        data_alignment: alignment,
        data_section_start,
    })
}

fn parse_v2<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let kv_count = reader.read_u64::<LittleEndian>()?;

    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair(reader)?);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_info_v2(reader)?);
    }

    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(2, &kv_pairs, &tensors, alignment);

    Ok(GgufHeader {
        version: 2,
        kv_pairs,
        tensors,
        data_alignment: alignment,
        data_section_start,
    })
}

fn parse_v3<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let kv_count = reader.read_u64::<LittleEndian>()?;
    eprintln!(
        "DEBUG: parse_v3: tensor_count={}, kv_count={}",
        tensor_count, kv_count
    );

    // v3 practical format: same structure as v2, just different semantics
    // No extra padding after counts

    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair_v3(reader)?);
    }
    eprintln!("DEBUG: parse_v3: read {} KV pairs", kv_pairs.len());

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_info_v3(reader)?);
    }

    let alignment = read_alignment_from_kv(&kv_pairs);
    let data_section_start = compute_data_section_start(3, &kv_pairs, &tensors, alignment);

    Ok(GgufHeader {
        version: 3,
        kv_pairs,
        tensors,
        data_alignment: alignment,
        data_section_start,
    })
}

/// Read KV pair - used for v1/v2 (keys and strings use u32 lengths)
fn read_kv_pair<R: Read + Seek>(reader: &mut R) -> Result<GgufKvPair, GgufError> {
    // Key is length-prefixed with u32 LE
    let key_len = reader.read_u32::<LittleEndian>()? as usize;
    if key_len == 0 || key_len > 1024 * 1024 {
        return Err(GgufError::Io(format!(
            "key length {} out of range",
            key_len
        )));
    }

    let key_bytes = read_bytes(reader, key_len)?;
    let key = String::from_utf8(key_bytes).map_err(GgufError::Utf8)?;

    // Value type: u32 LE
    let value_type_raw = reader.read_u32::<LittleEndian>()?;
    let value_type = GgufValueType::from_u32(value_type_raw)
        .ok_or(GgufError::InvalidValueType(value_type_raw))?;

    // Read value based on type
    let value = read_kv_value_v2(reader, value_type)?;

    Ok(GgufKvPair {
        key,
        value_type,
        value,
    })
}

/// Read KV pair for v3 practical format - keys and strings use u64 lengths
///
/// GGUF v3 practical format follows llama.cpp's wire layout:
/// - Key length: u64 (LE) - allows longer metadata keys
/// - Key name: raw bytes
/// - Value type: u32 (LE) - identifies value format
/// - Value data: varies by type
fn read_kv_pair_v3<R: Read + Seek>(reader: &mut R) -> Result<GgufKvPair, GgufError> {
    // 1. Read key length (u64 LE)
    let key_len = reader.read_u64::<LittleEndian>()? as usize;

    if key_len == 0 || key_len > 1024 * 1024 {
        return Err(GgufError::Io(format!(
            "key length {} out of range",
            key_len
        )));
    }

    // 2. Read key name (raw bytes)
    let key_bytes = read_bytes(reader, key_len)?;
    eprintln!(
        "DEBUG: key_bytes = {:?}",
        String::from_utf8_lossy(&key_bytes)
    );
    let key = String::from_utf8(key_bytes).map_err(GgufError::Utf8)?;

    // 3. Read value type (u32 LE)
    let value_type_raw = reader.read_u32::<LittleEndian>()?;
    let value_type = GgufValueType::from_u32(value_type_raw)
        .ok_or(GgufError::InvalidValueType(value_type_raw))?;

    // 4. Read value based on type
    let value = read_kv_value_v3(reader, value_type)?;

    Ok(GgufKvPair {
        key,
        value_type,
        value,
    })
}

/// Read KV value for v3 format (handles alignment and u64 string lengths)
fn read_kv_value_v3<R: Read + Seek>(
    reader: &mut R,
    value_type: GgufValueType,
) -> Result<GgufKvValue, GgufError> {
    match value_type {
        GgufValueType::String => {
            // String value: u64 length per llama.cpp spec
            let str_len = reader.read_u64::<LittleEndian>()? as usize;
            let bytes = read_bytes(reader, str_len)?;
            Ok(GgufKvValue::String(
                String::from_utf8(bytes).map_err(GgufError::Utf8)?,
            ))
        }
        GgufValueType::Array => {
            // Array: read element type (u32), then count (u64), then elements
            // GGUF v3 array format per llama.cpp:
            // - Element type: u32 - identifies type of array elements
            // - Count: u64 - number of elements
            // - Elements: serialized based on element type
            // Note: String array elements use u64 lengths (not u32) to match
            // real GGUF files from llama.cpp
            let elem_type_raw = reader.read_u32::<LittleEndian>()?;
            let elem_type = GgufValueType::from_u32(elem_type_raw)
                .ok_or(GgufError::InvalidValueType(elem_type_raw))?;

            let elem_count = reader.read_u64::<LittleEndian>()? as usize;

            let mut elements = Vec::with_capacity(elem_count);
            for _ in 0..elem_count {
                // For each element, recursively read based on element type
                match elem_type {
                GgufValueType::String => {
                    // Real GGUF files use u64 for string array element lengths
                    // While top-level strings use u32 lengths in some GGUF versions,
                    // string elements inside arrays consistently use u64 lengths
                    // across all llama.cpp models. This matches the wire format
                    // specification and ensures compatibility with real model files.
                    let str_len = reader.read_u64::<LittleEndian>()? as usize;
                        let bytes = read_bytes(reader, str_len)?;
                        elements.push(GgufKvValue::String(
                            String::from_utf8(bytes).map_err(GgufError::Utf8)?,
                        ));
                    }
                    GgufValueType::Uint32 => {
                        let val = reader.read_u32::<LittleEndian>()?;
                        elements.push(GgufKvValue::Uint32(val));
                    }
                    _ => {
                        // Fallback: read as uint32
                        let val = reader.read_u32::<LittleEndian>()?;
                        elements.push(GgufKvValue::Uint32(val));
                    }
                }
            }

            Ok(GgufKvValue::Array(elements))
        }
        GgufValueType::Int8 => {
            let val = reader.read_i8()?;
            Ok(GgufKvValue::Int8(val))
        }
        GgufValueType::Uint16 => {
            let val = reader.read_u16::<LittleEndian>()?;
            Ok(GgufKvValue::Uint16(val))
        }
        GgufValueType::Int16 => {
            let val = reader.read_i16::<LittleEndian>()?;
            Ok(GgufKvValue::Int16(val))
        }
        GgufValueType::Uint32 => {
            let val = reader.read_u32::<LittleEndian>()?;
            Ok(GgufKvValue::Uint32(val))
        }
        GgufValueType::Int32 => {
            let val = reader.read_i32::<LittleEndian>()?;
            Ok(GgufKvValue::Int32(val))
        }
        GgufValueType::Uint64 => {
            let val = reader.read_u64::<LittleEndian>()?;
            Ok(GgufKvValue::Uint64(val))
        }
        GgufValueType::Int64 => {
            let val = reader.read_i64::<LittleEndian>()?;
            Ok(GgufKvValue::Int64(val))
        }
        GgufValueType::Float32 => {
            let val = reader.read_f32::<LittleEndian>()?;
            Ok(GgufKvValue::Float32(val))
        }
        GgufValueType::Float64 => {
            let val = reader.read_f64::<LittleEndian>()?;
            Ok(GgufKvValue::Float64(val))
        }
        GgufValueType::Bool => {
            let val = reader.read_u8()? != 0;
            Ok(GgufKvValue::Bool(val))
        }
        _ => Err(GgufError::InvalidValueType(value_type as u32)),
    }
}

/// Read KV value for v2 format (strings use u32 lengths)
fn read_kv_value_v2<R: Read + Seek>(
    reader: &mut R,
    value_type: GgufValueType,
) -> Result<GgufKvValue, GgufError> {
    match value_type {
        GgufValueType::String => {
            let str_len = reader.read_u32::<LittleEndian>()? as usize;
            let bytes = read_bytes(reader, str_len)?;
            Ok(GgufKvValue::String(
                String::from_utf8(bytes).map_err(GgufError::Utf8)?,
            ))
        }
        GgufValueType::Array => {
            let elem_count = reader.read_u64::<LittleEndian>()? as usize;
            let mut elements = Vec::with_capacity(elem_count);
            for _ in 0..elem_count {
                let elem_type_raw = reader.read_u32::<LittleEndian>()?;
                let elem_type =
                    GgufValueType::from_u32(elem_type_raw).unwrap_or(GgufValueType::String);
                match elem_type {
                    GgufValueType::String => {
                        let str_len = reader.read_u32::<LittleEndian>()? as usize;
                        let bytes = read_bytes(reader, str_len)?;
                        elements.push(GgufKvValue::String(
                            String::from_utf8(bytes).map_err(GgufError::Utf8)?,
                        ));
                    }
                    GgufValueType::Uint32 => {
                        let val = reader.read_u32::<LittleEndian>()?;
                        elements.push(GgufKvValue::Uint32(val));
                    }
                    _ => {
                        let val = reader.read_u32::<LittleEndian>()?;
                        elements.push(GgufKvValue::Uint32(val));
                    }
                }
            }
            Ok(GgufKvValue::Array(elements))
        }
        GgufValueType::Int8 => {
            let val = reader.read_i8()?;
            Ok(GgufKvValue::Int8(val))
        }
        GgufValueType::Uint16 => {
            let val = reader.read_u16::<LittleEndian>()?;
            Ok(GgufKvValue::Uint16(val))
        }
        GgufValueType::Int16 => {
            let val = reader.read_i16::<LittleEndian>()?;
            Ok(GgufKvValue::Int16(val))
        }
        GgufValueType::Uint32 => {
            let val = reader.read_u32::<LittleEndian>()?;
            Ok(GgufKvValue::Uint32(val))
        }
        GgufValueType::Int32 => {
            let val = reader.read_i32::<LittleEndian>()?;
            Ok(GgufKvValue::Int32(val))
        }
        GgufValueType::Uint64 => {
            let val = reader.read_u64::<LittleEndian>()?;
            Ok(GgufKvValue::Uint64(val))
        }
        GgufValueType::Int64 => {
            let val = reader.read_i64::<LittleEndian>()?;
            Ok(GgufKvValue::Int64(val))
        }
        GgufValueType::Float32 => {
            let val = reader.read_f32::<LittleEndian>()?;
            Ok(GgufKvValue::Float32(val))
        }
        GgufValueType::Float64 => {
            let val = reader.read_f64::<LittleEndian>()?;
            Ok(GgufKvValue::Float64(val))
        }
        GgufValueType::Bool => {
            let val = reader.read_u8()? != 0;
            Ok(GgufKvValue::Bool(val))
        }
        _ => Err(GgufError::InvalidValueType(value_type as u32)),
    }
}

/// Read tensor info for v3 practical format - uses u64 for name length
fn read_tensor_info_v3<R>(reader: &mut R) -> Result<GgufTensorInfo, GgufError>
where
    R: Read + std::io::Seek,
{
    // Per llama.cpp gguf_reader.py _get_tensor_info_field():
    // - Name length: u64 LE (NOT u32!)
    // - Name: raw bytes
    // - N dims: u32 LE
    // - Shape: n_dims * u64 LE
    // - Dtype: u32 LE
    // - Offset: u64 LE

    // 1. Read tensor name length (u64 LE)
    let name_len = reader.read_u64::<LittleEndian>()? as usize;
    if name_len == 0 || name_len > 1024 * 1024 {
        return Err(GgufError::Io(format!(
            "tensor name length {} out of range",
            name_len
        )));
    }

    // 2. Read tensor name
    let name_bytes = read_bytes(reader, name_len)?;
    let name = String::from_utf8(name_bytes).map_err(GgufError::Utf8)?;

    // 3. Read number of dimensions (u32 LE)
    let n_dims = reader.read_u32::<LittleEndian>()?;

    // 4. Read shape array (n_dims * u64 LE)
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }

    // 5. Read data type (u32 LE)
    let dtype = reader.read_u32::<LittleEndian>()?;

    // 6. Read data offset (u64 LE)
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(GgufTensorInfo { name, shape, offset, dtype })
}

/// Read tensor info for v2 format (keys use u32 lengths)
fn read_tensor_info_v2<R: Read + Seek>(reader: &mut R) -> Result<GgufTensorInfo, GgufError> {
    // Tensor name: u32 length + raw bytes
    let name_len = reader.read_u32::<LittleEndian>()? as usize;
    if name_len == 0 || name_len > 1024 * 1024 {
        return Err(GgufError::Io(format!(
            "tensor name length {} out of range",
            name_len
        )));
    }

    let name_bytes = read_bytes(reader, name_len)?;
    let name = String::from_utf8(name_bytes).map_err(GgufError::Utf8)?;

    // Shape: u32 count + shape array (u64 each)
    let n_dims = reader.read_u32::<LittleEndian>()? as usize;
    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        shape.push(reader.read_u64::<LittleEndian>()?);
    }

    // Dtype: u32
    let dtype = reader.read_u32::<LittleEndian>()?;

    // Offset: u64
    let offset = reader.read_u64::<LittleEndian>()?;

    Ok(GgufTensorInfo {
        name,
        shape,
        offset,
        dtype,
    })
}

fn read_alignment_from_kv(kv_pairs: &[GgufKvPair]) -> Option<u64> {
    kv_pairs
        .iter()
        .find(|p| p.key == "general.alignment")
        .and_then(|p| p.value.as_u64())
}

fn read_bytes<R>(reader: &mut R, len: usize) -> Result<Vec<u8>, GgufError>
where
    R: Read,
{
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn compute_data_section_start(
    version: u32,
    kv_pairs: &[GgufKvPair],
    tensors: &[GgufTensorInfo],
    data_alignment: Option<u64>,
) -> u64 {
    // Header base: magic (4) + version (4) + tensor_count (8) + kv_count (8) = 24 bytes
    let header_base: u64 = 4 + 4 + 8 + 8;

    let kv_size: u64 = match version {
        3 => kv_pairs.iter().map(|p| p.raw_byte_size_v3() as u64).sum(),
        _ => kv_pairs.iter().map(|p| p.raw_byte_size() as u64).sum(),
    };

    let tensor_size: u64 = tensors.iter().map(|t| t.raw_byte_size() as u64).sum();
    let mut data_section = header_base
        .checked_add(kv_size)
        .and_then(|v| v.checked_add(tensor_size))
        .unwrap_or(u64::MAX);

    // Apply alignment padding for v3
    if version == 3 {
        if let Some(alignment) = data_alignment {
            if alignment > 0 {
                let remainder = data_section % alignment;
                if remainder != 0 {
                    data_section += alignment - remainder;
                }
            }
        }
    }

    data_section
}

pub fn extract_tensor_bytes_from_path(
    path: &std::path::Path,
    file_offset: u64,
    stored_size: usize,
) -> Result<Vec<u8>, GgufError> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    let mut file = File::open(path).map_err(|e| GgufError::Io(format!("open gguf: {}", e)))?;
    file.seek(SeekFrom::Start(file_offset))
        .map_err(|e| GgufError::Io(format!("seek: {}", e)))?;
    let mut buffer = vec![0u8; stored_size];
    file.read_exact(&mut buffer)
        .map_err(|e| GgufError::Io(format!("read: {}", e)))?;
    Ok(buffer)
}

pub fn extract_tensor_bytes<R>(
    reader: &mut R,
    dtype: u32,
    element_count: u64,
    _offset: u64,
    _data_section_start: u64,
) -> Result<Vec<u8>, GgufError>
where
    R: Read,
{
    let _ = (dtype, _offset, _data_section_start); // placeholder
    let size = tensor_bytes_for_dtype(dtype, element_count);
    let mut buffer = vec![0u8; size];
    reader.read_exact(&mut buffer)?;
    Ok(buffer)
}

pub fn extract_tensor_bytes_from<R>(
    reader: &mut R,
    dtype: u32,
    element_count: u64,
    _offset: u64,
    _data_section_start: u64,
) -> Result<Vec<u8>, GgufError>
where
    R: Read,
{
    let _ = (dtype, _offset, _data_section_start); // placeholder
    let size = tensor_bytes_for_dtype(dtype, element_count);
    let mut buffer = vec![0u8; size];
    reader.read_exact(&mut buffer)?;
    Ok(buffer)
}

pub fn tensor_bytes_for_dtype(dtype: u32, element_count: u64) -> usize {
    // Placeholder - actual implementation depends on quantization type
    let _ = dtype;
    element_count as usize * 4 // rough guess
}

#[cfg(test)]
mod tests_real_file {
    use super::*;
    use crate::tests::conformance_corpus_path;

    #[test]
    #[ignore = "Requires conformance corpus files"]
    fn test_parse_conformance_corpus_qwen2_5() {
        let path = conformance_corpus_path("qwen2.5-0.5b-instruct-q4_k_m.gguf");

        // Should parse without error
        let header = parse_gguf(&path).expect("Failed to parse real GGUF file");

        eprintln!("Header version: {}", header.version);
        assert_eq!(header.version, 3);

        // Should have KV pairs
        assert!(
            header.kv_pairs.len() > 0,
            "Should have KV pairs"
        );
        eprintln!("KV pair count: {}", header.kv_pairs.len());

        // Check a specific key exists
        let has_architecture = header
            .kv_pairs
            .iter()
            .any(|p| p.key == "general.architecture");
        assert!(has_architecture, "Should have general.architecture KV pair");

        // Find and print the architecture value
        if let Some(arch_pair) = header
            .kv_pairs
            .iter()
            .find(|p| p.key == "general.architecture")
        {
            eprintln!("Architecture value: {:?}", arch_pair.value);
        }

        // Should have tensors
        assert!(header.tensors.len() > 0, "Should have tensors");
        eprintln!("Tensor count: {}", header.tensors.len());

        eprintln!("SUCCESS: Real GGUF file parsed correctly!");
    }
}
