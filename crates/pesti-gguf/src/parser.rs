/// Little-endian byte-order helpers for GGUF parsing.
///
/// GGUF is always little-endian, so we use explicit byte-reinterpretation
/// instead of generic trait-based readers like byteorder::LittleEndian.
mod le_bytes {
    /// Read u8 (no endianness needed)
    #[inline]
    pub fn read_u8<R: std::io::Read>(reader: &mut R) -> Result<u8, std::io::Error> {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    /// Read u16 LE
    #[inline]
    pub fn read_u16_le<R: std::io::Read>(reader: &mut R) -> Result<u16, std::io::Error> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read u32 LE
    #[inline]
    pub fn read_u32_le<R: std::io::Read>(reader: &mut R) -> Result<u32, std::io::Error> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read u64 LE
    #[inline]
    pub fn read_u64_le<R: std::io::Read>(reader: &mut R) -> Result<u64, std::io::Error> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read f32 LE (via u32)
    #[inline]
    pub fn read_f32_le<R: std::io::Read>(reader: &mut R) -> Result<f32, std::io::Error> {
        let bits = read_u32_le(reader)?;
        Ok(f32::from_bits(bits))
    }

    /// Read f64 LE (via u64)
    #[inline]
    pub fn read_f64_le<R: std::io::Read>(reader: &mut R) -> Result<f64, std::io::Error> {
        let bits = read_u64_le(reader)?;
        Ok(f64::from_bits(bits))
    }

    /// Read i8 (no endianness needed)
    #[inline]
    pub fn read_i8<R: std::io::Read>(reader: &mut R) -> Result<i8, std::io::Error> {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        Ok(buf[0] as i8)
    }

    /// Read i16 LE
    #[inline]
    pub fn read_i16_le<R: std::io::Read>(reader: &mut R) -> Result<i16, std::io::Error> {
        let bytes = read_u16_le(reader)?;
        Ok(bytes as i16)
    }

    /// Read i32 LE
    #[inline]
    pub fn read_i32_le<R: std::io::Read>(reader: &mut R) -> Result<i32, std::io::Error> {
        let bytes = read_u32_le(reader)?;
        Ok(bytes as i32)
    }

    /// Read i64 LE
    #[inline]
    pub fn read_i64_le<R: std::io::Read>(reader: &mut R) -> Result<i64, std::io::Error> {
        let bytes = read_u64_le(reader)?;
        Ok(bytes as i64)
    }
}

use le_bytes::{
    read_f32_le, read_f64_le, read_i16_le, read_i32_le, read_i64_le, read_i8, read_u16_le,
    read_u32_le, read_u64_le, read_u8,
};
use std::io::{Read, Seek};
use std::path::Path;

use crate::error::GgufError;
use crate::types::{GgufHeader, GgufKvPair, GgufKvValue, GgufTensorInfo, GgufValueType, GgufVersion, GgufWireFormat, IntWidth};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
// const GGUF_VERSION_1/2/3 removed - use GgufVersion enum instead

// Constants from llama.cpp reference implementation
const GGUF_DEFAULT_ALIGNMENT: u32 = 32;
const GGUF_MAX_KEY_LENGTH: u64 = 1024 * 1024; // 1 MiB (was 1 GiB - unrealistic for metadata keys)
const GGUF_MAX_TENSOR_NAME_LENGTH: u64 = 1024 * 1024; // 1 MiB (was 1 GiB - realistic max is ~256 bytes)
const GGUF_MAX_COUNT: usize = 10_000_000; // 10M elements - sanity limit for KV pairs, tensors, array elements

/// Semantic wrapper: read string/key lengths with format-dependent width  
fn read_length<R: Read>(reader: &mut R, width: IntWidth) -> Result<usize, GgufError> {
    let val = read_int(reader, width)?;
    if val == 0 || val > GGUF_MAX_KEY_LENGTH {
        return Err(GgufError::InvalidMetadata(format!(
            "length {} is invalid (must be > 0 and ≤ {})",
            val, GGUF_MAX_KEY_LENGTH
        )));
    }
    val.try_into().map_err(|_| {
        GgufError::InvalidMetadata(format!(
            "length {} exceeds maximum usize ({})",
            val, usize::MAX
        ))
    })
}

pub fn parse_gguf(path: &Path) -> Result<GgufHeader, GgufError> {
    use std::io::BufReader;
    
    let file = std::fs::File::open(path)
        .map_err(|e| GgufError::Io(format!("open {}: {}", path.display(), e)))?;
    let mut reader = BufReader::new(file);
    parse_gguf_reader(&mut reader)
}

pub fn parse_gguf_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufHeader, GgufError> {
    // Check magic number - GGUF files are always little-endian
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    
    if &magic_bytes != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(
            String::from_utf8_lossy(&magic_bytes).to_string(),
        ));
    }

    let version = read_u32_le(reader)?;
    let format = match GgufVersion::from_u32(version) {
        Some(v) => v.wire_format(),
        None => return Err(GgufError::UnsupportedVersion(version)),
    };

    // V1 has different header layout (u32 counts, tensors come after KV pairs)
    if version == 1 {
        return parse_gguf_v1(reader, &format);
    }

    // V2/V3 share the same layout
    parse_gguf_v2_v3(reader, version, &format)
}

fn parse_gguf_v1<R: Read + Seek>(reader: &mut R, format: &GgufWireFormat) -> Result<GgufHeader, GgufError> {
    let _tensor_count = read_u32_le(reader)?;
    let kv_count = read_int(reader, format.header_count_width)? as usize;
    validate_count(kv_count, "KV pair")?;

    let mut kv_pairs = Vec::with_capacity(kv_count);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair(reader, format)?);
    }

    build_header(1, kv_pairs, vec![], reader, format)
}

fn parse_gguf_v2_v3<R: Read + Seek>(reader: &mut R, version: u32, format: &GgufWireFormat) -> Result<GgufHeader, GgufError> {
    let tensor_count = read_int(reader, format.header_count_width)? as usize;
    let kv_count = read_int(reader, format.header_count_width)? as usize;
    validate_count(kv_count, "KV pair")?;
    validate_count(tensor_count, "tensor")?;

    let mut kv_pairs = Vec::with_capacity(kv_count);
    for _ in 0..kv_count {
        kv_pairs.push(read_kv_pair(reader, format)?);
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_info(reader, format)?);
    }

    build_header(version, kv_pairs, tensors, reader, format)
}

fn validate_count(count: usize, name: &str) -> Result<(), GgufError> {
    if count > GGUF_MAX_COUNT {
        return Err(GgufError::InvalidMetadata(format!(
            "{} count {} exceeds maximum {}", name, count, GGUF_MAX_COUNT
        )));
    }
    Ok(())
}

fn build_header<R: Read + Seek>(
    version: u32,
    kv_pairs: Vec<GgufKvPair>,
    tensors: Vec<GgufTensorInfo>,
    _reader: &mut R,
    format: &GgufWireFormat,
) -> Result<GgufHeader, GgufError> {
    let alignment = read_alignment_from_kv(&kv_pairs)?;
    let data_section_start = compute_data_section_start(
        format,
        &kv_pairs,
        &tensors,
        Some(alignment),
    );

    Ok(GgufHeader {
        version,
        kv_pairs,
        tensors,
        data_alignment: Some(alignment),
        data_section_start,
    })
}

/// Read a u32 or u64 from the reader based on the wire format's integer width.
fn read_int<R: Read>(reader: &mut R, width: IntWidth) -> Result<u64, GgufError> {
    match width {
        IntWidth::U32 => Ok(read_u32_le(reader)? as u64),
        IntWidth::U64 => Ok(read_u64_le(reader)?),
    }
}

/// Read a KV pair using the wire format to determine field widths.
fn read_kv_pair<R: Read + Seek>(
    reader: &mut R,
    format: &GgufWireFormat,
) -> Result<GgufKvPair, GgufError> {
    let key_len = read_length(reader, format.key_width)?;
    let key_bytes = read_bytes(reader, key_len)?;
    let key = String::from_utf8(key_bytes).map_err(GgufError::Utf8)?;

    let value_type_raw = read_u32_le(reader)?;
    let value_type = GgufValueType::from_u32(value_type_raw)
        .ok_or(GgufError::InvalidValueType(value_type_raw))?;

    let value = read_kv_value_generic(
        reader,
        value_type,
        format.string_width,
        format.array_count_width,
    )?;

    Ok(GgufKvPair {
        key,
        value_type,
        value,
    })
}

/// Read tensor info using the wire format to determine name length width.
fn read_tensor_info<R: Read + Seek>(
    reader: &mut R,
    format: &GgufWireFormat,
) -> Result<GgufTensorInfo, GgufError> {
    let name_len = read_length(reader, format.tensor_name_width)?;
    if name_len > GGUF_MAX_TENSOR_NAME_LENGTH as usize {
        return Err(GgufError::TensorNameLengthOutOfRange(name_len as u64));
    }

    let name_bytes = read_bytes(reader, name_len)?;
    let name = String::from_utf8(name_bytes).map_err(GgufError::Utf8)?;

    // n_dims, shape, dtype, offset are always u32/u64 across all versions
    let n_dims = read_u32_le(reader)?;
    // Clamp to 8 dimensions (llama.cpp limit) to prevent DoS via massive allocations
    if n_dims > 8 {
        return Err(GgufError::InvalidTensor(
            "Too many tensor dimensions (max 8)".into()
        ));
    }
    let shape = (0..n_dims)
        .map(|_| read_u64_le(reader))
        .collect::<Result<Vec<u64>, _>>()?;
    let dtype = read_u32_le(reader)?;
    let offset = read_u64_le(reader)?;

    Ok(GgufTensorInfo { name, shape, offset, dtype })
}
/// Read a single scalar KV value (shared across all GGUF versions).
fn read_kv_scalar<R: Read>(reader: &mut R, value_type: GgufValueType) -> Result<GgufKvValue, GgufError> {
    match value_type {
        GgufValueType::Int8 => Ok(GgufKvValue::Int8(read_i8(reader)?)),
        GgufValueType::Uint16 => Ok(GgufKvValue::Uint16(read_u16_le(reader)?)),
        GgufValueType::Int16 => Ok(GgufKvValue::Int16(read_i16_le(reader)?)),
        GgufValueType::Uint32 => Ok(GgufKvValue::Uint32(read_u32_le(reader)?)),
        GgufValueType::Int32 => Ok(GgufKvValue::Int32(read_i32_le(reader)?)),
        GgufValueType::Uint64 => Ok(GgufKvValue::Uint64(read_u64_le(reader)?)),
        GgufValueType::Int64 => Ok(GgufKvValue::Int64(read_i64_le(reader)?)),
        GgufValueType::Float32 => Ok(GgufKvValue::Float32(read_f32_le(reader)?)),
        GgufValueType::Float64 => Ok(GgufKvValue::Float64(read_f64_le(reader)?)),
        GgufValueType::Bool => Ok(GgufKvValue::Bool(read_u8(reader)? != 0)),
        GgufValueType::Bfloat16 => {
            let val = read_u16_le(reader)?;
            Ok(GgufKvValue::Bfloat16(f32::from_bits((val as u32) << 16)))
        }
        GgufValueType::Float16 => Ok(GgufKvValue::Float16(read_u16_le(reader)?)),
        GgufValueType::Uint8 => Ok(GgufKvValue::Uint8(read_u8(reader)?)),
        _ => Err(GgufError::InvalidValueType(value_type as u32)),
    }
}

/// Read a single array element (shared across all GGUF versions).
fn read_kv_array_element<R: Read>(
    reader: &mut R,
    elem_type: GgufValueType,
    elem_type_raw: u32,
    string_width: IntWidth,
) -> Result<GgufKvValue, GgufError> {
    match elem_type {
        GgufValueType::String => {
            let str_len = read_length(reader, string_width)?;
            let bytes = read_bytes(reader, str_len)?;
            Ok(GgufKvValue::String(
                String::from_utf8(bytes).map_err(GgufError::Utf8)?,
            ))
        }
        GgufValueType::Uint32 => Ok(GgufKvValue::Uint32(read_u32_le(reader)?)),
        GgufValueType::Int8 => Ok(GgufKvValue::Int8(read_i8(reader)?)),
        GgufValueType::Uint16 => Ok(GgufKvValue::Uint16(read_u16_le(reader)?)),
        GgufValueType::Int16 => Ok(GgufKvValue::Int16(read_i16_le(reader)?)),
        GgufValueType::Int32 => Ok(GgufKvValue::Int32(read_i32_le(reader)?)),
        GgufValueType::Uint64 => Ok(GgufKvValue::Uint64(read_u64_le(reader)?)),
        GgufValueType::Int64 => Ok(GgufKvValue::Int64(read_i64_le(reader)?)),
        GgufValueType::Float32 => Ok(GgufKvValue::Float32(read_f32_le(reader)?)),
        GgufValueType::Bool => Ok(GgufKvValue::Bool(read_u8(reader)? != 0)),
        GgufValueType::Bfloat16 => {
            let val = read_u16_le(reader)?;
            Ok(GgufKvValue::Bfloat16(f32::from_bits((val as u32) << 16)))
        }
        GgufValueType::Float16 => Ok(GgufKvValue::Float16(read_u16_le(reader)?)),
        GgufValueType::Uint8 => Ok(GgufKvValue::Uint8(read_u8(reader)?)),
        GgufValueType::Int8Array | GgufValueType::Uint8Array => {
            Err(GgufError::UnsupportedArrayElementType(elem_type_raw))
        }
        _ => Err(GgufError::UnsupportedArrayElementType(elem_type_raw)),
    }
}

/// Generic KV value reader — parameterized by wire format integer widths.
fn read_kv_value_generic<R: Read + Seek>(
    reader: &mut R,
    value_type: GgufValueType,
    string_width: IntWidth,
    array_count_width: IntWidth,
) -> Result<GgufKvValue, GgufError> {
    match value_type {
        GgufValueType::String => {
            let str_len = read_length(reader, string_width)?;
            let bytes = read_bytes(reader, str_len)?;
            Ok(GgufKvValue::String(
                String::from_utf8(bytes).map_err(GgufError::Utf8)?,
            ))
        }
        GgufValueType::Array => {
            let elem_type_raw = read_u32_le(reader)?;
            let elem_type = GgufValueType::from_u32(elem_type_raw)
                .ok_or(GgufError::InvalidValueType(elem_type_raw))?;
            let elem_count = read_int(reader, array_count_width)? as usize;
            validate_count(elem_count, "array element")?;
            let mut elements = Vec::with_capacity(elem_count);
            for _ in 0..elem_count {
                elements.push(read_kv_array_element(reader, elem_type, elem_type_raw, string_width)?);
            }
            Ok(GgufKvValue::Array(elements))
        }
        _ => read_kv_scalar(reader, value_type),
    }
}
fn read_alignment_from_kv(kv_pairs: &[GgufKvPair]) -> Result<u64, GgufError> {
    let alignment = kv_pairs
        .iter()
        .find(|p| p.key == "general.alignment")
        .and_then(|p| p.value.as_u64())
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT as u64);

    // Validate alignment is a non-zero power of two (per llama.cpp spec)
    if alignment == 0 {
        return Err(GgufError::AlignmentOutOfRange(0));
    }
    
    // Check power of two: (n & (n-1)) == 0 for powers of two
    if (alignment & (alignment - 1)) != 0 {
        return Err(GgufError::AlignmentOutOfRange(alignment as u32));
    }

    // Sanity check: alignment should be reasonable (max 64KB)
    // Real GGUF files use 32, 256, or rarely 4096 bytes
    const MAX_ALIGNMENT: u64 = 65536; // 64KB
    if alignment > MAX_ALIGNMENT {
        return Err(GgufError::AlignmentOutOfRange(alignment as u32));
    }

    Ok(alignment)
}
/// Read bytes from reader with better error handling
fn read_bytes<R>(reader: &mut R, len: usize) -> Result<Vec<u8>, GgufError>
where
    R: Read,
{
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            // Provide more context about what we were trying to read
            GgufError::UnexpectedEof {
                expected: len,
                actual: 0, // We didn't read any bytes
            }
        } else {
            GgufError::Binary(e)
        }
    })?;
    Ok(buf)
}

pub fn compute_data_section_start(
    format: &GgufWireFormat,
    kv_pairs: &[GgufKvPair],
    tensors: &[GgufTensorInfo],
    data_alignment: Option<u64>,
) -> u64 {
    // Header base: magic (4) + version (4) + tensor_count + kv_count (format-dependent widths)
    let mut header_base: u64 = 4 + 4;
    
    // Add count fields based on format
    match format.header_count_width {
        IntWidth::U32 => header_base += 8, // 2 x u32 counts
        IntWidth::U64 => header_base += 16, // 2 x u64 counts
    }

    // Use format-aware byte size calculation for KV pairs and tensors
    let kv_size: u64 = kv_pairs.iter().map(|p| p.raw_byte_size_for_format(format) as u64).sum();
    let tensor_size: u64 = tensors.iter().map(|t| t.raw_byte_size_for_format(format) as u64).sum();
    
    let mut data_section = header_base
        .checked_add(kv_size)
        .and_then(|v| v.checked_add(tensor_size))
        .unwrap_or(u64::MAX);

    // Apply alignment padding (all versions)
    if let Some(alignment) = data_alignment {
        if alignment > 0 {
            let remainder = data_section % alignment;
            if remainder != 0 {
                // Saturate on overflow instead of wrapping
                data_section = data_section.saturating_add(alignment - remainder);
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
    // Delegate to stored_size which has correct per-type calculations
    let info = GgufTensorInfo {
        name: String::new(),
        shape: vec![element_count],
        offset: 0,
        dtype,
    };
    info.stored_size().unwrap_or(0) as usize
}

#[cfg(test)]
mod tests_real_file {
    use super::*;
    use crate::tests::conformance_corpus_path;

    // Note: This test requires conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf
    #[test]
    #[ignore = "Requires conformance corpus file"]
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

    #[test]
    fn test_bfloat16_bit_reinterpretation() {
        // Test that BF16 values are correctly reinterpreted as bit patterns, not numeric conversions
        // 0x3F80 in BF16 represents 1.0 (not 16256.0 which would be from numeric conversion)
        
        let bf16_bits: u16 = 0x3F80;
        // The correct interpretation: shift bits to high position and reinterpret as f32
        let expected_value = f32::from_bits((bf16_bits as u32) << 16);
        
        assert_eq!(expected_value, 1.0, "BF16 0x3F80 should represent 1.0");
        
        // Test zero value: 0x0000 represents 0.0 in BF16
        let bf16_zero: u16 = 0x0000;
        let zero_value = f32::from_bits((bf16_zero as u32) << 16);
        assert_eq!(zero_value, 0.0, "BF16 0x0000 should represent 0.0");
        
        // Test that numeric conversion would give wrong result for 0x3F80
        let wrong_conversion = bf16_bits as f32;
        assert_eq!(wrong_conversion, 16256.0, "Numeric conversion of 0x3F80 gives 16256.0 (WRONG)");
    }
}
