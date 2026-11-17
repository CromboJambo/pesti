use serde::{Deserialize, Serialize};
use crate::error::GgufError;

/// GGUF file format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufVersion {
    V1,
    V2,
    V3,
}

impl GgufVersion {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            _ => None,
        }
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }

    /// Returns the wire format for this GGUF version.
    pub fn wire_format(self) -> GgufWireFormat {
        match self {
            Self::V1 => GgufWireFormat::V1,
            Self::V2 => GgufWireFormat::V2,
            Self::V3 => GgufWireFormat::V3,
        }
    }
}

/// Integer width for wire-format fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntWidth {
    /// 4-byte little-endian integer.
    U32,
    /// 8-byte little-endian integer.
    U64,
}

impl IntWidth {
    pub const fn is_u32(self) -> bool {
        matches!(self, Self::U32)
    }

    pub const fn is_u64(self) -> bool {
        matches!(self, Self::U64)
    }
}

/// Describes the wire-format encoding differences between GGUF versions.
///
/// Instead of branching on version numbers, the parser and writer
/// are parameterized by this struct. The version-specific code becomes
/// trivial: select the right `GgufWireFormat` constant, then call the
/// generic parser/writer.
///
/// The key insight: GGUF v1/v2/v3 differ **only** in integer widths
/// for specific fields. The grammar (what fields exist, in what order)
/// is the same across all versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufWireFormat {
    /// Width of tensor_count and kv_count in the file header.
    pub header_count_width: IntWidth,
    /// Width of KV pair key length prefix.
    pub key_width: IntWidth,
    /// Width of string value length prefix (top-level and in arrays).
    pub string_width: IntWidth,
    /// Width of array element count.
    pub array_count_width: IntWidth,
    /// Width of tensor info name length prefix.
    pub tensor_name_width: IntWidth,
}

impl GgufWireFormat {
    /// V1: u32 counts, u32 keys, u32 strings, u32 array counts, u32 tensor names.
    pub const V1: Self = Self {
        header_count_width: IntWidth::U32,
        key_width: IntWidth::U32,
        string_width: IntWidth::U32,
        array_count_width: IntWidth::U32,
        tensor_name_width: IntWidth::U32,
    };

    /// V2: u64 counts, u32 keys, u32 strings, u64 array counts, u32 tensor names.
    pub const V2: Self = Self {
        header_count_width: IntWidth::U64,
        key_width: IntWidth::U32,
        string_width: IntWidth::U32,
        array_count_width: IntWidth::U64,
        tensor_name_width: IntWidth::U32,
    };

    /// V3: u64 counts, u64 keys, u64 strings, u64 array counts, u64 tensor names.
    pub const V3: Self = Self {
        header_count_width: IntWidth::U64,
        key_width: IntWidth::U64,
        string_width: IntWidth::U64,
        array_count_width: IntWidth::U64,
        tensor_name_width: IntWidth::U64,
    };
}

impl From<GgufVersion> for GgufWireFormat {
    fn from(v: GgufVersion) -> Self {
        v.wire_format()
    }
}

/// GGUF key-value value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Uint64,
    Int64,
    Bool,
    String,
    Array,
    Int8Array,
    Uint8Array,
    Float64,
    Bfloat16,
    Float16,
}

/// GGUF key-value pair value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GgufKvValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Bool(bool),
    String(String),
    Array(Vec<GgufKvValue>),
    Int8Array(Vec<i8>),
    Uint8Array(Vec<u8>),
    Float64(f64),
    Bfloat16(f32), // Stored as f32, but represents bfloat16 bit pattern
    Float16(u16),
}

impl GgufKvValue {
    /// Compute the serialized byte size of this value, given the string length prefix width in bytes.
    /// String and array string elements use `str_len_bytes` for their length prefix.
    pub fn raw_byte_size_with_str_width(&self, str_len_bytes: usize) -> usize {
        match self {
            Self::Uint8(..) | Self::Int8(..) | Self::Bool(..) => 1,
            Self::Uint16(..) | Self::Int16(..) | Self::Bfloat16(..) | Self::Float16(..) => 2,
            Self::Uint32(..) | Self::Int32(..) | Self::Float32(..) => 4,
            Self::Uint64(..) | Self::Int64(..) | Self::Float64(..) => 8,
            Self::String(s) => str_len_bytes + s.len(),
            Self::Int8Array(arr) => 8 + arr.len(),
            Self::Uint8Array(arr) => 8 + arr.len(),
            Self::Array(arr) => {
                let elem_size = match arr.first().map(|v| v.value_type()) {
                    Some(GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool
                    | GgufValueType::Int8Array | GgufValueType::Uint8Array) => 1,
                    Some(GgufValueType::Uint16 | GgufValueType::Int16
                    | GgufValueType::Float16) => 2,
                    Some(GgufValueType::Uint32 | GgufValueType::Int32
                    | GgufValueType::Float32) => 4,
                    Some(GgufValueType::Uint64 | GgufValueType::Int64) => 8,
                    Some(GgufValueType::String) => {
                        return arr.iter().map(|v| match v {
                            Self::String(s) => str_len_bytes + s.len(),
                            _ => 0,
                        }).sum::<usize>() + 4 + 8;
                    }
                    _ => 4,
                };
                // Array wire format: elem_type (u32) + count (u64) + elements.
                4 + 8 + arr.len() * elem_size
            }
        }
    }

    pub fn value_type(&self) -> GgufValueType {
        match self {
            Self::Uint8(_) => GgufValueType::Uint8,
            Self::Int8(_) => GgufValueType::Int8,
            Self::Uint16(_) => GgufValueType::Uint16,
            Self::Int16(_) => GgufValueType::Int16,
            Self::Uint32(_) => GgufValueType::Uint32,
            Self::Int32(_) => GgufValueType::Int32,
            Self::Float32(_) => GgufValueType::Float32,
            Self::Uint64(_) => GgufValueType::Uint64,
            Self::Int64(_) => GgufValueType::Int64,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Array(_) => GgufValueType::Array,
            Self::Int8Array(_) => GgufValueType::Int8Array,
            Self::Uint8Array(_) => GgufValueType::Uint8Array,
            Self::Float64(_) => GgufValueType::Float64,
            Self::Bfloat16(_) => GgufValueType::Bfloat16,
            Self::Float16(_) => GgufValueType::Float16,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint8(v) => Some(*v as u32),
            Self::Int8(v) => Some(*v as u32),
            Self::Uint16(v) => Some(*v as u32),
            Self::Int16(v) => Some(*v as u32),
            Self::Uint32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            Self::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Int8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Int16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            Self::Bfloat16(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(*v as f64),
            Self::Float64(v) => Some(*v),
            _ => None,
        }
    }
}

impl GgufValueType {
    pub const fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            13 => Some(Self::Int8Array),
            14 => Some(Self::Uint8Array),
            15 => Some(Self::Bfloat16),
            16 => Some(Self::Float16),
            _ => None,
        }
    }

    pub const fn to_u32(self) -> u32 {
        match self {
            Self::Uint8 => 0,
            Self::Int8 => 1,
            Self::Uint16 => 2,
            Self::Int16 => 3,
            Self::Uint32 => 4,
            Self::Int32 => 5,
            Self::Float32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::Uint64 => 10,
            Self::Int64 => 11,
            Self::Float64 => 12,
            Self::Int8Array => 13,
            Self::Uint8Array => 14,
            Self::Bfloat16 => 15,
            Self::Float16 => 16,
        }
    }

    pub const fn element_size(self) -> Option<usize> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool | Self::Int8Array | Self::Uint8Array => Some(1),
            Self::Uint16 | Self::Int16 | Self::Bfloat16 | Self::Float16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// A single key-value pair from the GGUF header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufKvPair {
    pub key: String,
    pub value_type: GgufValueType,
    pub value: GgufKvValue,
}

impl GgufKvPair {
    /// Compute serialized byte size using the wire format to determine key/string widths.
    pub fn raw_byte_size_for_format(&self, format: &GgufWireFormat) -> usize {
        let key_bytes = self.key.len();
        let key_len_bytes: usize = match format.key_width {
            IntWidth::U32 => 4,
            IntWidth::U64 => 8,
        };
        let str_len_bytes: usize = match format.string_width {
            IntWidth::U32 => 4,
            IntWidth::U64 => 8,
        };
        let value_bytes = self.value.raw_byte_size_with_str_width(str_len_bytes);
        key_len_bytes + key_bytes + 4 + value_bytes
    }
}

/// GGUF tensor data type.
/// 
/// Note: Uses snake_case to match llama.cpp's ggml.h naming convention,
/// even though Rust typically uses PascalCase for enum variants.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
    IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS,
    I8, I16, I32, I64, F64, IQ1_M, BF16, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0,
    Unknown(u32),
}

impl GgufDtype {
    /// Map a raw GGML type id to a variant.
    ///
    /// IDs are kept in lockstep with llama.cpp's `enum ggml_type` in `ggml.h`
    /// (authoritative). IDs 4, 5, 31, 32, 33, 36, 37, 38 were removed from
    /// ggml and have no variant; they resolve to `Unknown`.
    pub const fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            40 => Self::NVFP4,
            41 => Self::Q1_0,
            42 => Self::Q2_0,
            _ => Self::Unknown(v),
        }
    }

    /// Inverse of [`from_u32`]. `Unknown(v)` round-trips to `v`.
    pub const fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::IQ2_XXS => 16,
            Self::IQ2_XS => 17,
            Self::IQ3_XXS => 18,
            Self::IQ1_S => 19,
            Self::IQ4_NL => 20,
            Self::IQ3_S => 21,
            Self::IQ2_S => 22,
            Self::IQ4_XS => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::IQ1_M => 29,
            Self::BF16 => 30,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 35,
            Self::MXFP4 => 39,
            Self::NVFP4 => 40,
            Self::Q1_0 => 41,
            Self::Q2_0 => 42,
            Self::Unknown(v) => v,
        }
    }

    /// True for every quantized dtype (all Q*/IQ* families).
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1
                | Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K
                | Self::IQ2_XXS | Self::IQ2_XS | Self::IQ3_XXS | Self::IQ1_S | Self::IQ4_NL
                | Self::IQ3_S | Self::IQ2_S | Self::IQ4_XS | Self::IQ1_M | Self::TQ1_0
                | Self::TQ2_0 | Self::MXFP4 | Self::NVFP4 | Self::Q1_0 | Self::Q2_0
        )
    }

    /// Number of elements per quantization block.
    ///
    /// Non-quantized types (F32, F16, BF16, I8, I16, I32, I64, F64) use a block
    /// size of 1 (one scalar per block). Quantized types use the block size from
    /// llama.cpp's `ggml_type_traits` table. `Unknown` types have no known block
    /// size and return `None`.
    pub const fn block_size(self) -> Option<usize> {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::I16 | Self::I32
            | Self::I64 | Self::F64 => Some(1),
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1
            | Self::IQ4_NL | Self::MXFP4 | Self::Q1_0 => Some(32),
            Self::NVFP4 => Some(64),
            Self::Q2_0 => Some(64),
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K
            | Self::IQ2_XXS | Self::IQ2_XS | Self::IQ3_XXS | Self::IQ1_S | Self::IQ3_S
            | Self::IQ2_S | Self::IQ4_XS | Self::IQ1_M | Self::TQ1_0 | Self::TQ2_0 => Some(256),
            Self::Unknown(_) => None,
        }
    }

    /// Bytes stored per quantization block.
    ///
    /// Non-quantized types store one scalar per block (bytes = scalar size).
    /// Quantized types use the block byte size from llama.cpp's `ggml_type_traits`
    /// table (verified against `sizeof(block_*)` in the ggml headers).
    pub const fn bytes_per_block(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::I8 => Some(1),
            Self::I32 => Some(4),
            Self::F64 => Some(8),
            Self::I64 => Some(8),
            Self::Q4_0 => Some(18),
            Self::Q4_1 => Some(20),
            Self::Q5_0 => Some(22),
            Self::Q5_1 => Some(24),
            Self::Q8_0 => Some(34),
            Self::Q8_1 => Some(36),
            Self::Q2_K => Some(84),
            Self::Q3_K => Some(110),
            Self::Q4_K => Some(144),
            Self::Q5_K => Some(176),
            Self::Q6_K => Some(210),
            Self::Q8_K => Some(292),
            Self::IQ2_XXS => Some(66),
            Self::IQ2_XS => Some(74),
            Self::IQ3_XXS => Some(98),
            Self::IQ1_S => Some(50),
            Self::IQ4_NL => Some(18),
            Self::IQ3_S => Some(110),
            Self::IQ2_S => Some(82),
            Self::IQ4_XS => Some(136),
            Self::IQ1_M => Some(56),
            Self::TQ1_0 => Some(54),
            Self::TQ2_0 => Some(66),
            Self::MXFP4 => Some(17),
            Self::NVFP4 => Some(36),
            Self::Q1_0 => Some(18),
            Self::Q2_0 => Some(18),
            Self::Unknown(_) => None,
        }
    }

    /// Bytes per scalar element for non-quantized types.
    ///
    /// Returns `None` for quantized types (their storage is block-based, not
    /// scalar-based — use [`block_size`] and [`bytes_per_block`] instead).
    pub const fn bytes_per_element(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::I8 => Some(1),
            Self::I32 => Some(4),
            Self::F64 => Some(8),
            Self::I64 => Some(8),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
            Self::NVFP4 => "NVFP4",
            Self::Q1_0 => "Q1_0",
            Self::Q2_0 => "Q2_0",
            Self::Unknown(_) => "unknown",
        }
    }
}

/// A single tensor's metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub dtype: u32,
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> u64 {
        self.shape.iter().copied().product::<u64>()
    }

    /// Safe element count that returns None on overflow.
    pub fn element_count_checked(&self) -> Option<u64> {
        self.shape.iter().copied().try_fold(1u64, |acc, d| acc.checked_mul(d))
    }

    pub fn ndims(&self) -> u32 {
        self.shape.len() as u32
    }

    /// Compute the actual stored byte size on disk with overflow checking.
    ///
    /// Mirrors ggml's `ggml_row_size`: `type_size * ne / blck_size`, which
    /// requires the element count to be a multiple of the block size. Valid
    /// GGUF quantized tensors always satisfy this (ggml asserts it), so a
    /// non-aligned count is reported as an invalid tensor rather than guessed.
    pub fn stored_size(&self) -> Result<u64, GgufError> {
        let n = self.element_count();
        let dtype = GgufDtype::from_u32(self.dtype);
        let (block_elems, block_bytes) = match (dtype.block_size(), dtype.bytes_per_block()) {
            (Some(be), Some(bb)) => (be, bb),
            _ => {
                return Err(GgufError::InvalidTensor(format!(
                    "dtype {} ({}) has no known block size",
                    self.dtype,
                    dtype.name()
                )))
            }
        };
        if n % block_elems as u64 != 0 {
            return Err(GgufError::InvalidTensor(format!(
                "tensor '{}' has {} elements, not a multiple of the {}-element block for dtype {} ({})",
                self.name, n, block_elems, self.dtype, dtype.name()
            )));
        }
        let blocks = n / block_elems as u64;
        blocks
            .checked_mul(block_bytes as u64)
            .ok_or_else(|| GgufError::InvalidTensor(format!("tensor '{}' stored size overflow", self.name)))
    }

    /// Compute serialized byte size using the wire format to determine name length width.
    pub fn raw_byte_size_for_format(&self, format: &GgufWireFormat) -> usize {
        let name_len_bytes: usize = match format.tensor_name_width {
            IntWidth::U32 => 4,
            IntWidth::U64 => 8,
        };
        // n_dims(u32) + shape(n * u64) + dtype(u32) + offset(u64) are always the same
        name_len_bytes + self.name.len() + 4 + (self.shape.len() * 8) + 4 + 8
    }
}

/// Parsed GGUF header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufHeader {
    pub version: u32,
    pub kv_pairs: Vec<GgufKvPair>,
    pub tensors: Vec<GgufTensorInfo>,
    #[serde(default = "default_data_alignment")]
    pub data_alignment: Option<u64>,
    pub data_section_start: u64,
}

fn default_data_alignment() -> Option<u64> { Some(32) }

impl GgufHeader {
    pub fn get_kv<T: From<GgufKvValue>>(&self, key: &str) -> Option<T> {
        self.kv_pairs.iter().find(|p| p.key == key).map(|p| T::from(p.value.clone()))
    }

    pub fn get_kv_str(&self, key: &str) -> Option<&str> {
        self.kv_pairs.iter().find(|p| p.key == key).and_then(|p| p.value.as_str())
    }

    pub fn get_kv_u32(&self, key: &str) -> Option<u32> {
        self.kv_pairs.iter().find(|p| p.key == key).and_then(|p| p.value.as_u32())
    }

    pub fn architecture(&self) -> Option<&str> {
        self.get_kv_str("general.architecture").or_else(|| self.get_kv_str("arch"))
    }

    pub fn context_length(&self) -> Option<u32> {
        self.get_kv_u32("llama.context_length")
            .or_else(|| self.get_kv_u32("n_ctx"))
            // Then try architecture-specific key: {arch}.context_length
            .or_else(|| {
                self.architecture().and_then(|arch| {
                    let arch_key = format!("{}.context_length", arch);
                    self.get_kv_u32(&arch_key)
                })
            })
    }

    pub fn embedding_length(&self) -> Option<u32> {
        // Try standard llama.cpp keys first
        self.get_kv_u32("llama.embedding_length")
            .or_else(|| self.get_kv_u32("n_embd"))
            // Then try architecture-specific key: {arch}.embedding_length
            .or_else(|| {
                self.architecture().and_then(|arch| {
                    let arch_key = format!("{}.embedding_length", arch);
                    self.get_kv_u32(&arch_key)
                })
            })
    }

    pub fn block_count(&self) -> Option<u32> {
        self.get_kv_u32("llama.block_count")
            .or_else(|| self.get_kv_u32("n_layer"))
            // Then try architecture-specific key: {arch}.block_count
            .or_else(|| {
                self.architecture().and_then(|arch| {
                    let arch_key = format!("{}.block_count", arch);
                    self.get_kv_u32(&arch_key)
                })
            })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    pub fn total_tensor_bytes_f32(&self) -> u64 {
        // Return actual byte count for F32 tensors (element_count * 4 bytes per F32)
        self.tensors.iter().map(|t| t.element_count() * 4).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_roundtrip_all() {
        // Every dtype id that maps to a named variant in ggml.h.
        // Removed ids (4, 5, 31, 32, 33, 36, 37, 38) resolve to Unknown and are
        // not expected to round-trip to a named variant.
        for v in [
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 34, 35, 39, 40, 41, 42,
        ] {
            let dt = GgufDtype::from_u32(v);
            assert_eq!(dt.to_u32(), v, "roundtrip failed for {v}");
        }
        // Removed ids must resolve to Unknown.
        for v in [4, 5, 31, 32, 33, 36, 37, 38] {
            assert!(matches!(GgufDtype::from_u32(v), GgufDtype::Unknown(_)), "id {v} should be Unknown");
        }
    }

    #[test]
    fn test_dtype_integer_types() {
        // dtype 24-28 are I8/I16/I32/I64/F64 per official ggml.h
        assert_eq!(GgufDtype::from_u32(24), GgufDtype::I8);
        assert_eq!(GgufDtype::from_u32(25), GgufDtype::I16);
        assert_eq!(GgufDtype::from_u32(26), GgufDtype::I32);
        assert_eq!(GgufDtype::from_u32(27), GgufDtype::I64);
        assert_eq!(GgufDtype::from_u32(28), GgufDtype::F64);
    }

    #[test]
    fn test_stored_size_i8() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 24 };
        assert_eq!(info.stored_size().unwrap(), 100);
    }

    #[test]
    fn test_stored_size_i16() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 25 };
        assert_eq!(info.stored_size().unwrap(), 200);
    }

    #[test]
    fn test_stored_size_i32() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 26 };
        assert_eq!(info.stored_size().unwrap(), 400);
    }

    #[test]
    fn test_stored_size_i64() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 27 };
        assert_eq!(info.stored_size().unwrap(), 800);
    }

    #[test]
    fn test_stored_size_f64() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![100], offset: 0, dtype: 28 };
        assert_eq!(info.stored_size().unwrap(), 800);
    }

    #[test]
    fn test_stored_size_overflow() {
        let info = GgufTensorInfo { name: "t".to_string(), shape: vec![u64::MAX / 4 + 1], offset: 0, dtype: 0 };
        assert!(info.stored_size().is_err());
    }

    #[test]
    fn test_total_tensor_bytes_f32() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![],
            tensors: vec![
                GgufTensorInfo { name: "a".into(), shape: vec![100], offset: 0, dtype: 0 }, // F32
                GgufTensorInfo { name: "b".into(), shape: vec![50], offset: 0, dtype: 0 },   // F32
            ],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        // (100 + 50) elements * 4 bytes per F32 = 600 bytes
        assert_eq!(header.total_tensor_bytes_f32(), 600);
    }

    #[test]
    fn test_v2_alignment_computation() {
        use crate::parser::compute_data_section_start;
        
        // Test that alignment is applied for v2 files with custom alignment
        let kv_pairs: Vec<GgufKvPair> = vec![];
        let tensors: Vec<GgufTensorInfo> = vec![];
        
        // Compute with 256-byte alignment - should align header size to 256
        let data_start =
            compute_data_section_start(&GgufWireFormat::V2, &kv_pairs, &tensors, Some(256));
        assert_eq!(
            data_start % 256,
            0,
            "Data section start should be aligned to 256 bytes"
        );

        // With default alignment (32)
        let data_start_32 =
            compute_data_section_start(&GgufWireFormat::V2, &kv_pairs, &tensors, Some(32));
        assert_eq!(
            data_start_32 % 32,
            0,
            "Data section start should be aligned to 32 bytes"
        );
    }

    #[test]
    fn test_embedding_length_architecture_specific() {
        // Test that embedding_length looks for architecture-specific keys
        let mut kv_pairs: Vec<GgufKvPair> = vec![];
        
        // Add architecture key
        kv_pairs.push(GgufKvPair {
            key: "general.architecture".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("qwen2".to_string()),
        });
        
        // Add architecture-specific embedding_length (should be found)
        kv_pairs.push(GgufKvPair {
            key: "qwen2.embedding_length".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(4096),
        });
        
        let header = GgufHeader {
            version: 3,
            kv_pairs,
            tensors: vec![],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        
        // Should find qwen2.embedding_length
        assert_eq!(header.embedding_length(), Some(4096));
    }

    #[test]
    fn test_embedding_length_fallback_order() {
        // Test fallback order: llama.embedding_length > n_embd > {arch}.embedding_length
        
        let mut kv_pairs: Vec<GgufKvPair> = vec![];
        
        // Add all three keys - should prefer llama.embedding_length
        kv_pairs.push(GgufKvPair {
            key: "general.architecture".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("mistral".to_string()),
        });
        kv_pairs.push(GgufKvPair {
            key: "llama.embedding_length".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(5120),
        });
        kv_pairs.push(GgufKvPair {
            key: "n_embd".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(4096),
        });
        kv_pairs.push(GgufKvPair {
            key: "mistral.embedding_length".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(3072),
        });
        
        let header = GgufHeader {
            version: 3,
            kv_pairs,
            tensors: vec![],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        
        // Should prefer llama.embedding_length (5120) over all others
        assert_eq!(header.embedding_length(), Some(5120));

        // Test fallback to n_embd when llama.embedding_length is missing
        let mut kv_pairs2: Vec<GgufKvPair> = vec![];
        kv_pairs2.push(GgufKvPair {
            key: "general.architecture".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("mistral".to_string()),
        });
        kv_pairs2.push(GgufKvPair {
            key: "n_embd".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(4096),
        });
        kv_pairs2.push(GgufKvPair {
            key: "mistral.embedding_length".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(3072),
        });
        
        let header2 = GgufHeader {
            version: 3,
            kv_pairs: kv_pairs2,
            tensors: vec![],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        
        // Should prefer n_embd (4096) over mistral.embedding_length (3072)
        assert_eq!(header2.embedding_length(), Some(4096));

        // Test fallback to architecture-specific key when both standard keys are missing
        let mut kv_pairs3: Vec<GgufKvPair> = vec![];
        kv_pairs3.push(GgufKvPair {
            key: "general.architecture".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("mistral".to_string()),
        });
        kv_pairs3.push(GgufKvPair {
            key: "mistral.embedding_length".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(3072),
        });
        
        let header3 = GgufHeader {
            version: 3,
            kv_pairs: kv_pairs3,
            tensors: vec![],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        
        // Should fall back to architecture-specific key (3072)
        assert_eq!(header3.embedding_length(), Some(3072));

        // Test returns None when no embedding_length key exists
        let mut kv_pairs4: Vec<GgufKvPair> = vec![];
        kv_pairs4.push(GgufKvPair {
            key: "general.architecture".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("mistral".to_string()),
        });
        
        let header4 = GgufHeader {
            version: 3,
            kv_pairs: kv_pairs4,
            tensors: vec![],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        
        // Should return None when no embedding_length key exists
        assert_eq!(header4.embedding_length(), None);

        // Test returns None when architecture is unknown (can't form arch-specific key)
        let mut kv_pairs5: Vec<GgufKvPair> = vec![];
        kv_pairs5.push(GgufKvPair {
            key: "n_embd".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(4096),
        });
        
        let header5 = GgufHeader {
            version: 3,
            kv_pairs: kv_pairs5,
            tensors: vec![],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        
        // Should still find n_embd even without architecture key
        assert_eq!(header5.embedding_length(), Some(4096));
    }
}