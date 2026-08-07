use serde::{Deserialize, Serialize};

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
}

/// GGUF key-value value type.
///
/// Maps to GGUF spec value types (per actual llama.cpp implementation):
/// UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5,
/// UINT64=6, INT64=7, FLOAT32=8, BOOL=9, STRING=10, ARRAY=11, BFLOAT16=15
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32, // llama.cpp uses 6 for FLOAT32
    Uint64,
    Int64,
    Bool,
    String,
    Array,
    Int8Array,
    Uint8Array,
    Float64, // llama.cpp uses 12 for FLOAT64
    Bfloat16,
    Float16,
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
            6 => Some(Self::Float32), // llama.cpp uses 6 for FLOAT32
            7 => Some(Self::Bool),     // llama.cpp uses 7 for BOOL
            8 => Some(Self::String),   // llama.cpp uses 8 for STRING
            9 => Some(Self::Array),    // llama.cpp uses 9 for ARRAY
            10 => Some(Self::Uint64),  // llama.cpp uses 10 for UINT64
            11 => Some(Self::Int64),   // llama.cpp uses 11 for INT64
            12 => Some(Self::Float64), // llama.cpp uses 12 for FLOAT64 (extra type)
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
            Self::Float32 => 6, // llama.cpp uses 6 for FLOAT32
            Self::Bool => 7,     // llama.cpp uses 7 for BOOL
            Self::String => 8,   // llama.cpp uses 8 for STRING
            Self::Array => 9,    // llama.cpp uses 9 for ARRAY
            Self::Uint64 => 10,  // llama.cpp uses 10 for UINT64
            Self::Int64 => 11,   // llama.cpp uses 11 for INT64
            Self::Float64 => 12, // llama.cpp uses 12 for FLOAT64
            Self::Int8Array => 13,
            Self::Uint8Array => 14,
            Self::Bfloat16 => 15,
            Self::Float16 => 16,
        }
    }

    pub const fn to_u8(self) -> u8 {
        self.to_u32() as u8
    }

    pub fn is_array(self) -> bool {
        self == Self::Array
    }

    pub fn element_size(self) -> Option<usize> {
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
    /// Total byte size of this KV pair in the GGUF file (key_len + key + type + value).
    /// For GGUF v2 and earlier: key_len is u32 (4 bytes).
    pub fn raw_byte_size(&self) -> usize {
        self.raw_byte_size_for_version(2)
    }

    /// Total byte size of this KV pair in GGUF v3 wire format (u64 key length).
    pub fn raw_byte_size_v3(&self) -> usize {
        self.raw_byte_size_for_version(3)
    }

    /// Compute raw byte size for a specific GGUF version.
    fn raw_byte_size_for_version(&self, _version: u32) -> usize {
        let key_bytes = self.key.len();
        // llama.cpp v3 uses u32 for key lengths (despite spec saying u64), only string values/tensor names are u64
        let key_len_bytes = 4; 
        let value_bytes = match &self.value {
            GgufKvValue::Uint8(..)
            | GgufKvValue::Int8(..)
            | GgufKvValue::Bool(..) => 1,
            GgufKvValue::Uint16(..) | GgufKvValue::Int16(..) | GgufKvValue::Bfloat16(..) | GgufKvValue::Float16(..) => 2,
            GgufKvValue::Uint32(..)
            | GgufKvValue::Int32(..)
            | GgufKvValue::Float32(..) => 4,
            GgufKvValue::Uint64(..)
            | GgufKvValue::Int64(..)
            | GgufKvValue::Float64(..) => 8,
            GgufKvValue::String(s) => 8 + s.len(),
            GgufKvValue::Int8Array(arr) => 8 + arr.len(),
            GgufKvValue::Uint8Array(arr) => 8 + arr.len(),
            GgufKvValue::Array(arr) => {
                let elem_size = match arr.first().map(|v| v.value_type()) {
                    Some(GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool | GgufValueType::Int8Array | GgufValueType::Uint8Array) => 1,
                    Some(GgufValueType::Uint16 | GgufValueType::Int16 | GgufValueType::Float16) => 2,
                    Some(GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32) => 4,
                    Some(GgufValueType::Uint64 | GgufValueType::Int64) => 8,
                    Some(GgufValueType::String) => {
                        return arr.iter().map(|v| match v {
                            GgufKvValue::String(s) => 8 + s.len(),
                            _ => 0,
                        }).sum::<usize>() + 1 + 8;
                    }
                    _ => 4,
                };
                1 + 8 + arr.len() * elem_size
            }
        };
        key_len_bytes + key_bytes + 4 + value_bytes
    }
}

/// GGUF tensor data type (stored on disk).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgufDtype {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    Q1_K,
    Q4_K_M,
    Q5_K_M,
    Q6_K_S,
    Q8_K_M,
    Q2_K_S,
    Q3_K_S,
    Q4_K_S,
    Q5_K_S,
    Q2_K_M,
    Unknown(u32),
}

impl GgufDtype {
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
            16 => Self::Unknown(v), // IQ2_XXS - newer type
            17 => Self::Unknown(v), // IQ2_XS - newer type
            18 => Self::Unknown(v), // IQ3_XXS - newer type
            19 => Self::Unknown(v), // IQ1_S - newer type
            20 => Self::Unknown(v), // IQ4_NL - newer type
            21 => Self::Unknown(v), // IQ3_S - newer type
            22 => Self::Unknown(v), // IQ2_S - newer type
            23 => Self::Unknown(v), // IQ4_XS - newer type
            24 => Self::I8,         // I8
            25 => Self::I16,        // I16
            26 => Self::I32,        // I32
            27 => Self::I64,        // I64
            28 => Self::F64,        // F64
            29 => Self::Unknown(v), // IQ1_M - newer type
            30 => Self::BF16,       // BF16
            31 => Self::Unknown(v), // Q4_0_4_4 - removed from gguf files
            32 => Self::Unknown(v), // Q4_0_4_8 - not yet used
            33 => Self::Unknown(v), // Q4_0_8_8 - not yet used
            34 => Self::Unknown(v), // TQ1_0 - newer type
            35 => Self::Unknown(v), // TQ2_0 - newer type
            36 => Self::Unknown(v), // IQ4_NL_4_4 - not yet used
            37 => Self::Unknown(v), // IQ4_NL_4_8 - not yet used
            38 => Self::Unknown(v), // IQ4_NL_8_8 - not yet used
            39 => Self::Unknown(v), // MXFP4 - newer type
            40 => Self::Unknown(v), // NVFP4 - newer type
            41 => Self::Unknown(v), // Q1_0 - newer type
            42 => Self::Unknown(v), // Q2_0 - newer type
            _ => Self::Unknown(v),
        }
    }

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
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::BF16 => 30,
            Self::Q1_K => 20,
            Self::Q4_K_M => 21,
            Self::Q5_K_M => 22,
            Self::Q6_K_S => 23,
            Self::Q8_K_M => 24,
            Self::Q2_K_S => 25,
            Self::Q3_K_S => 26,
            Self::Q4_K_S => 27,
            Self::Q5_K_S => 28,
            Self::Q2_K_M => 29,
            Self::Unknown(v) => v,
        }
    }

    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
                | Self::Q8_K
                | Self::Q1_K
                | Self::Q4_K_M
                | Self::Q5_K_M
                | Self::Q6_K_S
                | Self::Q8_K_M
                | Self::Q2_K_S
                | Self::Q3_K_S
                | Self::Q4_K_S
                | Self::Q5_K_S
                | Self::Q2_K_M
        )
    }

    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q8_0 | Self::Q8_1 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::F64 => 8,
            Self::BF16 => 2,
            Self::Q4_0 | Self::Q4_1 | Self::Q1_K | Self::Q5_0 | Self::Q5_1 | Self::Q4_K_M => 0,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q5_K_S | Self::Q5_K_M | Self::Q6_K | Self::Q6_K_S | Self::Q8_K | Self::Q8_K_M | Self::Q2_K_M | Self::Q2_K_S | Self::Q3_K_S | Self::Q4_K_S => 0,
            Self::Unknown(_) => 0,
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
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::BF16 => "BF16",
            Self::Q1_K => "Q1_K",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q5_K_M => "Q5_K_M",
            Self::Q6_K_S => "Q6_K_S",
            Self::Q8_K_M => "Q8_K_M",
            Self::Q2_K_S => "Q2_K_S",
            Self::Q3_K_S => "Q3_K_S",
            Self::Q4_K_S => "Q4_K_S",
            Self::Q5_K_S => "Q5_K_S",
            Self::Q2_K_M => "Q2_K_M",
            Self::Unknown(_) => "unknown",
        }
    }
}

/// A single tensor's metadata (name, shape, dtype, offset).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub dtype: u32,
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn ndims(&self) -> u32 {
        self.shape.len() as u32
    }

    /// Compute the actual stored byte size on disk.
    ///
    /// For quantized tensors, this is much smaller than element_count * 2 (F16).
    pub fn stored_size(&self) -> u64 {
        let n = self.element_count();
        let dtype = GgufDtype::from_u32(self.dtype);
        match dtype {
            GgufDtype::F32 => n * 4,
            GgufDtype::F16 | GgufDtype::BF16 => n * 2,
            GgufDtype::Q8_0 => {
                let full_blocks = n / 32;
                let remaining = n % 32;
                full_blocks * 34 + if remaining > 0 { 2 + remaining } else { 0 }
            }
            GgufDtype::Q8_1 => {
                let full_blocks = n / 32;
                let remaining = n % 32;
                full_blocks * 36 + if remaining > 0 { 4 + remaining } else { 0 }
            }
            GgufDtype::Q4_0 => {
                let full_blocks = n / 32;
                let remaining = n % 32;
                full_blocks * 18 + if remaining > 0 { 2 + remaining.div_ceil(2) } else { 0 }
            }
            GgufDtype::Q4_1 => {
                let full_blocks = n / 32;
                let remaining = n % 32;
                full_blocks * 20 + if remaining > 0 { 4 + remaining.div_ceil(2) } else { 0 }
            }
            GgufDtype::Q5_0 => n / 2 + 32 + 16,
            GgufDtype::Q5_1 => n / 2 + 64 + 16,
            GgufDtype::Q2_K => n / 4 + n * 6 / 32 + 8,
            GgufDtype::Q3_K => n / 8 + n * 6 / 32 + 16,
            GgufDtype::Q4_K => n / 4 + n * 6 / 32 + 16 + 32,
            GgufDtype::Q5_K => n / 4 + n * 6 / 32 + 16 + 32 + 16,
            GgufDtype::Q6_K => n / 2 + n / 4 + 256,
            GgufDtype::Q8_K => n / 2 + n * 6 / 32 + 256,
            GgufDtype::Q1_K => n / 8 + n * 6 / 32 + 96,
            GgufDtype::Q4_K_M | GgufDtype::Q5_K_M | GgufDtype::Q8_K_M => n / 4 + n * 6 / 32 + 48,
            GgufDtype::Q2_K_S | GgufDtype::Q3_K_S | GgufDtype::Q4_K_S | GgufDtype::Q5_K_S | GgufDtype::Q6_K_S | GgufDtype::Q2_K_M => n / 4 + n * 6 / 32 + 24,
            GgufDtype::I8 => n,
            GgufDtype::I16 => n * 2,
            GgufDtype::I32 => n * 4,
            GgufDtype::I64 => n * 8,
            GgufDtype::F64 => n * 8,
            GgufDtype::Unknown(_) => n * 2,
        }
    }

    /// Total byte size of this tensor info in the GGUF file (name_len + name + dims + shape + dtype + offset).
    pub fn raw_byte_size(&self) -> usize {
        8 + self.name.len() + 4 + (self.shape.len() * 8) + 4 + 8
    }
}

/// Parsed key-value value (runtime representation).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GgufKvValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufKvValue>),
    Int8Array(Vec<i8>),
    Uint8Array(Vec<u8>),
    Float64(f64), // llama.cpp uses 12 for FLOAT64
    Bfloat16(f32),
    Float16(u16),
}

impl GgufKvValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufKvValue::Uint8(v) => Some(*v as u64),
            GgufKvValue::Uint16(v) => Some(*v as u64),
            GgufKvValue::Uint32(v) => Some(*v as u64),
            GgufKvValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            GgufKvValue::Int8(v) => Some(*v as i64),
            GgufKvValue::Int16(v) => Some(*v as i64),
            GgufKvValue::Int32(v) => Some(*v as i64),
            GgufKvValue::Int64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufKvValue::Uint8(v) => Some(*v as u32),
            GgufKvValue::Uint16(v) => Some(*v as u32),
            GgufKvValue::Uint32(v) => Some(*v),
            GgufKvValue::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            GgufKvValue::Int8(v) => Some(*v as i32),
            GgufKvValue::Int16(v) => Some(*v as i32),
            GgufKvValue::Int32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufKvValue::Float32(v) => Some(*v),
            GgufKvValue::Bfloat16(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufKvValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufKvValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<GgufKvValue>> {
        match self {
            GgufKvValue::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn value_type(&self) -> GgufValueType {
        match self {
            GgufKvValue::Uint8(..) => GgufValueType::Uint8,
            GgufKvValue::Int8(..) => GgufValueType::Int8,
            GgufKvValue::Uint16(..) => GgufValueType::Uint16,
            GgufKvValue::Int16(..) => GgufValueType::Int16,
            GgufKvValue::Uint32(..) => GgufValueType::Uint32,
            GgufKvValue::Int32(..) => GgufValueType::Int32,
            GgufKvValue::Uint64(..) => GgufValueType::Uint64,
            GgufKvValue::Int64(..) => GgufValueType::Int64,
            GgufKvValue::String(..) => GgufValueType::String,
            GgufKvValue::Float32(..) => GgufValueType::Float32,
            GgufKvValue::Bool(..) => GgufValueType::Bool,
            GgufKvValue::Array(..) => GgufValueType::Array,
            GgufKvValue::Int8Array(..) => GgufValueType::Int8Array,
            GgufKvValue::Uint8Array(..) => GgufValueType::Uint8Array,
            GgufKvValue::Float64(..) => GgufValueType::Float64,
            GgufKvValue::Float16(..) => GgufValueType::Float16,
            GgufKvValue::Bfloat16(..) => GgufValueType::Bfloat16,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            GgufKvValue::Uint8(_) => "u8",
            GgufKvValue::Int8(_) => "i8",
            GgufKvValue::Uint16(_) => "u16",
            GgufKvValue::Int16(_) => "i16",
            GgufKvValue::Uint32(_) => "u32",
            GgufKvValue::Int32(_) => "i32",
            GgufKvValue::Uint64(_) => "u64",
            GgufKvValue::Int64(_) => "i64",
            GgufKvValue::String(_) => "str",
            GgufKvValue::Float32(_) => "f32",
            GgufKvValue::Bool(_) => "bool",
            GgufKvValue::Array(_) => "array",
            GgufKvValue::Int8Array(_) => "i8[]",
            GgufKvValue::Uint8Array(_) => "u8[]",
            GgufKvValue::Float64(_) => "f64",
            GgufKvValue::Float16(_) => "f16",
            GgufKvValue::Bfloat16(_) => "bf16",
        }
    }
}

/// Parsed GGUF header (everything before tensor data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufHeader {
    pub version: u32,
    pub kv_pairs: Vec<GgufKvPair>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Data alignment in bytes (default 32 for llama.cpp).
    #[serde(default = "default_data_alignment")]
    pub data_alignment: Option<u64>,
    pub data_section_start: u64,
}

fn default_data_alignment() -> Option<u64> {
    Some(32)
}

impl GgufHeader {
    pub fn get_kv<T: From<GgufKvValue>>(&self, key: &str) -> Option<T> {
        self.kv_pairs
            .iter()
            .find(|p| p.key == key)
            .map(|p| T::from(p.value.clone()))
    }

    pub fn get_kv_str(&self, key: &str) -> Option<&str> {
        self.kv_pairs
            .iter()
            .find(|p| p.key == key)
            .and_then(|p| p.value.as_str())
    }

    pub fn get_kv_u32(&self, key: &str) -> Option<u32> {
        self.kv_pairs
            .iter()
            .find(|p| p.key == key)
            .and_then(|p| p.value.as_u32())
    }

    pub fn get_kv_i32(&self, key: &str) -> Option<i32> {
        self.kv_pairs
            .iter()
            .find(|p| p.key == key)
            .and_then(|p| p.value.as_i32())
    }

    pub fn get_kv_u64(&self, key: &str) -> Option<u64> {
        self.kv_pairs
            .iter()
            .find(|p| p.key == key)
            .and_then(|p| p.value.as_u64())
    }

    pub fn get_kv_f32(&self, key: &str) -> Option<f32> {
        self.kv_pairs
            .iter()
            .find(|p| p.key == key)
            .and_then(|p| p.value.as_f32())
    }

    pub fn get_kv_bool(&self, key: &str) -> Option<bool> {
        self.kv_pairs
            .iter()
            .find(|p| p.key == key)
            .and_then(|p| p.value.as_bool())
    }

    pub fn get_kv_array(&self, key: &str) -> Option<&Vec<GgufKvValue>> {
        self.kv_pairs
            .iter()
            .find(|p| p.key == key)
            .and_then(|p| p.value.as_array())
    }

    pub fn to_config_map(&self) -> std::collections::HashMap<String, GgufKvValue> {
        self.kv_pairs
            .iter()
            .map(|p| (p.key.clone(), p.value.clone()))
            .collect()
    }

    /// Extract architecture name (e.g., "llama", "mistral", "qwen2").
    pub fn architecture(&self) -> Option<&str> {
        self.get_kv_str("general.architecture")
            .or_else(|| self.get_kv_str("arch"))
    }

    /// Extract file type string (e.g., "F16", "Q4_0", "Q8_0").
    pub fn file_type(&self) -> Option<String> {
        self.get_kv_str("general.file_type")
            .map(|s| s.to_string())
            .or_else(|| self.get_kv_str("ft").map(|s| s.to_string()))
            .or_else(|| self.get_kv_u32("general.file_type").map(|v| v.to_string()))
            .or_else(|| self.get_kv_u32("ft").map(|v| v.to_string()))
    }

    /// Extract context length (n_ctx).
    pub fn context_length(&self) -> Option<u32> {
        self.get_kv_u32("llama.context_length")
            .or_else(|| self.get_kv_u32("context_length"))
            .or_else(|| self.get_kv_u32("n_ctx"))
    }

    /// Extract embedding/vector dimension.
    pub fn embedding_length(&self) -> Option<u32> {
        // Try llama-style keys first (most common)
        self.get_kv_u32("llama.embedding_length")
            .or_else(|| self.get_kv_u32("embedding_length"))
            .or_else(|| self.get_kv_u32("n_embd"))
            // Then try qwen2-specific key
            .or_else(|| self.get_kv_u32("qwen2.embedding_length"))
    }

    /// Extract block count (number of layers).
    pub fn block_count(&self) -> Option<u32> {
        self.get_kv_u32("llama.block_count")
            .or_else(|| self.get_kv_u32("block_count"))
            .or_else(|| self.get_kv_u32("n_layer"))
            .or_else(|| self.get_kv_u32("qwen2.block_count"))
    }

    /// Extract attention head count.
    pub fn attention_head_count(&self) -> Option<u32> {
        self.get_kv_u32("llama.attention.head_count")
            .or_else(|| self.get_kv_u32("attention.head_count"))
            .or_else(|| self.get_kv_u32("n_head"))
    }

    /// Extract attention head count for KV (QKV projection).
    pub fn attention_head_count_kv(&self) -> Option<u32> {
        self.get_kv_u32("llama.attention.head_count_kv")
            .or_else(|| self.get_kv_u32("attention.head_count_kv"))
    }

    /// Extract rope dimension count.
    pub fn rope_dimension_count(&self) -> Option<i32> {
        self.get_kv_i32("llama.rope.dimension_count")
            .or_else(|| self.get_kv_i32("rope.dimension_count"))
            .or_else(|| self.get_kv_i32("rope_dim"))
    }

    /// Extract feed-forward dimension.
    pub fn feed_forward_length(&self) -> Option<u32> {
        self.get_kv_u32("llama.feed_forward_length")
            .or_else(|| self.get_kv_u32("feed_forward_length"))
            .or_else(|| self.get_kv_u32("n_ff"))
    }

    /// Extract rope scaling parameters.
    pub fn rope_scaling(&self) -> Option<&Vec<GgufKvValue>> {
        self.get_kv_array("rope.scaling")
    }

    /// Extract rope scaling type (e.g., "linear", "yarn").
    pub fn rope_scaling_type(&self) -> Option<&str> {
        self.get_kv_str("rope.scaling.type")
            .or_else(|| self.get_kv_str("rope_type"))
    }

    /// Extract token embedding length (vocabulary size).
    pub fn vocab_size(&self) -> Option<u32> {
        self.get_kv_u32("tokenizer.ggml.tokens")
            .or_else(|| self.get_kv_u32("vocab_size"))
            .or_else(|| self.get_kv_u32("n_vocab"))
    }

    /// Extract normalization epsilon.
    pub fn normalization_epsilon(&self) -> Option<f32> {
        self.get_kv_f32("llama.attention.layer_norm_rms_epsilon")
            .or_else(|| self.get_kv_f32("attention.layer_norm_epsilon"))
            .or_else(|| self.get_kv_f32("layer_norm_epsilon"))
            .or_else(|| self.get_kv_f32("rms_norm_eps"))
    }

    /// Extract quantization description if present.
    pub fn quantization_description(&self) -> Option<&str> {
        self.get_kv_str("general.quantization_version")
            .or_else(|| self.get_kv_str("quantization"))
    }

    /// Get tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|t| t.name == name)
    }

    /// Total tensor data size in bytes (sum of all tensor sizes assuming f32).
    /// Actual size depends on quantization — this is an upper bound.
    pub fn total_tensor_bytes_f32(&self) -> u64 {
        self.tensors.iter().map(|t| t.element_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_version_from_u32() {
        assert_eq!(GgufVersion::from_u32(1), Some(GgufVersion::V1));
        assert_eq!(GgufVersion::from_u32(2), Some(GgufVersion::V2));
        assert_eq!(GgufVersion::from_u32(3), Some(GgufVersion::V3));
        assert_eq!(GgufVersion::from_u32(4), None);
    }

    #[test]
    fn test_gguf_version_to_u32() {
        assert_eq!(GgufVersion::V1.to_u32(), 1);
        assert_eq!(GgufVersion::V2.to_u32(), 2);
        assert_eq!(GgufVersion::V3.to_u32(), 3);
    }

    #[test]
    fn test_value_type_from_u32() {
        assert_eq!(GgufValueType::from_u32(0), Some(GgufValueType::Uint8));
        assert_eq!(GgufValueType::from_u32(6), Some(GgufValueType::Float32)); // llama.cpp uses 6 for FLOAT32
        assert_eq!(GgufValueType::from_u32(7), Some(GgufValueType::Bool));
        assert_eq!(GgufValueType::from_u32(8), Some(GgufValueType::String)); // llama.cpp practical format
        assert_eq!(GgufValueType::from_u32(9), Some(GgufValueType::Array));
        assert_eq!(GgufValueType::from_u32(10), Some(GgufValueType::Uint64));
        assert_eq!(GgufValueType::from_u32(11), Some(GgufValueType::Int64));
        assert_eq!(GgufValueType::from_u32(12), Some(GgufValueType::Float64));
        assert_eq!(GgufValueType::from_u32(13), Some(GgufValueType::Int8Array));
        assert_eq!(GgufValueType::from_u32(14), Some(GgufValueType::Uint8Array));
        assert_eq!(GgufValueType::from_u32(15), Some(GgufValueType::Bfloat16));
        assert_eq!(GgufValueType::from_u32(16), Some(GgufValueType::Float16));
    }

    #[test]
    fn test_value_type_element_size() {
        assert_eq!(GgufValueType::Uint8.element_size(), Some(1));
        assert_eq!(GgufValueType::Float32.element_size(), Some(4));
        assert_eq!(GgufValueType::String.element_size(), None);
        assert_eq!(GgufValueType::Array.element_size(), None);
    }

    #[test]
    fn test_tensor_info_element_count() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![2, 3, 4],
            offset: 0,
            dtype: 0,
        };
        assert_eq!(info.element_count(), 24);
        assert_eq!(info.ndims(), 3);
    }

    #[test]
    fn test_gguf_header_helpers() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "general.architecture".to_string(),
                    value_type: GgufValueType::String,
                    value: GgufKvValue::String("llama".to_string()),
                },
                GgufKvPair {
                    key: "llama.context_length".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(4096),
                },
                GgufKvPair {
                    key: "llama.embedding_length".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(4096),
                },
                GgufKvPair {
                    key: "llama.attention.layer_norm_rms_epsilon".to_string(),
                    value_type: GgufValueType::Float32,
                    value: GgufKvValue::Float32(1e-5),
                },
            ],
            tensors: vec![
                GgufTensorInfo {
                    name: "token_embd.weight".to_string(),
                    shape: vec![4096],
                    offset: 0,
                    dtype: 1,
                },
            ],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.architecture(), Some("llama"));
        assert_eq!(header.context_length(), Some(4096));
        assert_eq!(header.embedding_length(), Some(4096));
        assert_eq!(header.has_tensor("token_embd.weight"), true);
        assert_eq!(header.has_tensor("missing"), false);
    }

    #[test]
    fn test_empty_header() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.architecture(), None);
        assert_eq!(header.context_length(), None);
        assert_eq!(header.get_kv_str("anything"), None);
        assert_eq!(header.get_kv_u32("anything"), None);
        assert_eq!(header.get_kv_i32("anything"), None);
        assert_eq!(header.get_kv_f32("anything"), None);
        assert_eq!(header.get_kv_bool("anything"), None);
        assert!(header.get_kv_array("anything").is_none());
        assert!(header.get_tensor("anything").is_none());
        assert!(!header.has_tensor("anything"));
        assert_eq!(header.file_type(), None);
        assert_eq!(header.rope_scaling_type(), None);
        assert_eq!(header.quantization_description(), None);
        assert_eq!(header.vocab_size(), None);
        assert_eq!(header.normalization_epsilon(), None);
        assert_eq!(header.feed_forward_length(), None);
        assert_eq!(header.rope_dimension_count(), None);
        assert_eq!(header.block_count(), None);
        assert_eq!(header.attention_head_count(), None);
        assert_eq!(header.attention_head_count_kv(), None);
    }

    #[test]
    fn test_kv_pair_serialization() {
        let kv = GgufKvPair {
            key: "test.key".to_string(),
            value_type: GgufValueType::Uint32,
            value: GgufKvValue::Uint32(42),
        };
        let json = serde_json::to_string(&kv).unwrap();
        let deserialized: GgufKvPair = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.key, "test.key");
        assert_eq!(deserialized.value_type, GgufValueType::Uint32);
        assert_eq!(deserialized.value, GgufKvValue::Uint32(42));
    }

    #[test]
    fn test_string_kv_pair_raw_byte_size() {
        let kv = GgufKvPair {
            key: "arch".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("llama".to_string()),
        };
        assert_eq!(kv.raw_byte_size(), 25);
    }

    #[test]
    fn test_string_kv_pair_raw_byte_size_empty() {
        let kv = GgufKvPair {
            key: "k".to_string(),
            value_type: GgufValueType::String,
            value: GgufKvValue::String("".to_string()),
        };
        assert_eq!(kv.raw_byte_size(), 17);
    }

    #[test]
    fn test_array_kv_pair_raw_byte_size() {
        let kv = GgufKvPair {
            key: "arr".to_string(),
            value_type: GgufValueType::Array,
            value: GgufKvValue::Array(vec![
                GgufKvValue::Uint32(1),
                GgufKvValue::Uint32(2),
                GgufKvValue::Uint32(3),
            ]),
        };
        assert_eq!(kv.raw_byte_size(), 32);
    }

    #[test]
    fn test_kv_value_conversions() {
        let u8_val = GgufKvValue::Uint8(255);
        assert_eq!(u8_val.as_u64(), Some(255));
        assert_eq!(u8_val.as_u32(), Some(255));
        assert!(u8_val.as_i64().is_none());
        assert!(u8_val.as_f32().is_none());
        assert!(u8_val.as_bool().is_none());
        assert!(u8_val.as_str().is_none());
        assert!(u8_val.as_array().is_none());

        let i8_val = GgufKvValue::Int8(-128);
        assert_eq!(i8_val.as_i64(), Some(-128));
        assert!(i8_val.as_u64().is_none());

        let f32_val = GgufKvValue::Float32(3.14);
        assert_eq!(f32_val.as_f32(), Some(3.14));

        let f32_val = GgufKvValue::Float32(3.14159);
        assert_eq!(f32_val.as_f32(), Some(3.14159f32));

        let bool_val = GgufKvValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));

        let str_val = GgufKvValue::String("hello".to_string());
        assert_eq!(str_val.as_str(), Some("hello"));
    }

    #[test]
    fn test_dtype_roundtrip_all() {
        for v in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 30] {
            let dt = GgufDtype::from_u32(v);
            assert_eq!(dt.to_u32(), v, "roundtrip failed for {v}");
        }
        // Values that remain unmapped and should be Unknown
        for v in [4, 5, 16, 17, 18, 19, 31, 32, 33, 34, 35, 100] {
            let dt = GgufDtype::from_u32(v);
            if let GgufDtype::Unknown(val) = dt {
                assert_eq!(val, v);
            } else {
                panic!("expected Unknown({v}) for input {v}");
            }
        }
    }

    #[test]
    fn test_is_quantized_all() {
        let quantized = [
            GgufDtype::Q4_0, GgufDtype::Q4_1, GgufDtype::Q5_0, GgufDtype::Q5_1,
            GgufDtype::Q8_0, GgufDtype::Q8_1, GgufDtype::Q2_K, GgufDtype::Q3_K,
            GgufDtype::Q4_K, GgufDtype::Q5_K, GgufDtype::Q6_K, GgufDtype::Q8_K,
            GgufDtype::Q1_K, GgufDtype::Q4_K_M, GgufDtype::Q5_K_M, GgufDtype::Q6_K_S,
            GgufDtype::Q8_K_M, GgufDtype::Q2_K_S, GgufDtype::Q3_K_S, GgufDtype::Q4_K_S,
            GgufDtype::Q5_K_S, GgufDtype::Q2_K_M,
        ];
        for dt in &quantized {
            assert!(dt.is_quantized(), "{dt:?} should be quantized");
        }

        let unquantized = [
            GgufDtype::F32, GgufDtype::F16, GgufDtype::I8, GgufDtype::I16,
            GgufDtype::I32, GgufDtype::I64, GgufDtype::F64, GgufDtype::BF16,
        ];
        for dt in &unquantized {
            assert!(!dt.is_quantized(), "{dt:?} should not be quantized");
        }
    }

    #[test]
    fn test_stored_size_quantized_variants() {
        let q8 = GgufTensorInfo {
            name: "t".to_string(), shape: vec![32], offset: 0, dtype: 8,
        };
        assert_eq!(q8.stored_size(), 34);

        let q8_2 = GgufTensorInfo {
            name: "t".to_string(), shape: vec![64], offset: 0, dtype: 8,
        };
        assert_eq!(q8_2.stored_size(), 68);

        let q8_3 = GgufTensorInfo {
            name: "t".to_string(), shape: vec![33], offset: 0, dtype: 8,
        };
        assert_eq!(q8_3.stored_size(), 37);

        let q4 = GgufTensorInfo {
            name: "t".to_string(), shape: vec![32], offset: 0, dtype: 2,
        };
        assert_eq!(q4.stored_size(), 18);

        let q4_2 = GgufTensorInfo {
            name: "t".to_string(), shape: vec![64], offset: 0, dtype: 2,
        };
        assert_eq!(q4_2.stored_size(), 36);

        let q2 = GgufTensorInfo {
            name: "t".to_string(), shape: vec![1], offset: 0, dtype: 10,
        };
        assert!(q2.stored_size() > 0);

        let q6 = GgufTensorInfo {
            name: "t".to_string(), shape: vec![256], offset: 0, dtype: 14,
        };
        assert!(q6.stored_size() > 0);

        let q8k = GgufTensorInfo {
            name: "t".to_string(), shape: vec![256], offset: 0, dtype: 15,
        };
        assert!(q8k.stored_size() > 0);
    }

    #[test]
    fn test_stored_size_integer_types() {
        let i8_t = GgufTensorInfo {
            name: "t".to_string(), shape: vec![100], offset: 0, dtype: 24,
        };
        assert_eq!(i8_t.stored_size(), 100);

        let i16_t = GgufTensorInfo {
            name: "t".to_string(), shape: vec![100], offset: 0, dtype: 25,
        };
        assert_eq!(i16_t.stored_size(), 200);

        let i32_t = GgufTensorInfo {
            name: "t".to_string(), shape: vec![100], offset: 0, dtype: 26,
        };
        assert_eq!(i32_t.stored_size(), 400);

        let i64_t = GgufTensorInfo {
            name: "t".to_string(), shape: vec![100], offset: 0, dtype: 27,
        };
        assert_eq!(i64_t.stored_size(), 800);

        let f64_t = GgufTensorInfo {
            name: "t".to_string(), shape: vec![100], offset: 0, dtype: 28,
        };
        assert_eq!(f64_t.stored_size(), 800);
    }

    #[test]
    fn test_tensor_info_serialization() {
        let tensor = GgufTensorInfo {
            name: "test.weight".to_string(),
            shape: vec![128, 256],
            offset: 4096,
            dtype: 1,
        };
        let json = serde_json::to_string(&tensor).unwrap();
        let deserialized: GgufTensorInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test.weight");
        assert_eq!(deserialized.shape, vec![128u64, 256u64]);
        assert_eq!(deserialized.offset, 4096);
        assert_eq!(deserialized.dtype, 1);
    }

    #[test]
    fn test_header_serialization() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "general.architecture".to_string(),
                    value_type: GgufValueType::String,
                    value: GgufKvValue::String("llama".to_string()),
                },
            ],
            tensors: vec![
                GgufTensorInfo {
                    name: "token_embd.weight".to_string(),
                    shape: vec![4096],
                    offset: 0,
                    dtype: 1,
                },
            ],
            data_alignment: None,
            data_section_start: 1024,
        };
        let json = serde_json::to_string(&header).unwrap();
        let deserialized: GgufHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.version, 3);
        assert_eq!(deserialized.kv_pairs.len(), 1);
        assert_eq!(deserialized.tensors.len(), 1);
        // Note: serde deserializes null as None, not the default value
        // The default only applies when the field is missing from JSON
        assert_eq!(deserialized.data_alignment, None);
        assert_eq!(deserialized.data_section_start, 1024);
    }

    #[test]
    fn test_to_config_map_identity() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "a".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(1),
                },
                GgufKvPair {
                    key: "b".to_string(),
                    value_type: GgufValueType::String,
                    value: GgufKvValue::String("two".to_string()),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        let map = header.to_config_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map["a"], GgufKvValue::Uint32(1));
        assert_eq!(map["b"], GgufKvValue::String("two".to_string()));
    }

    #[test]
    fn test_get_kv_with_type_conversion() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "general.architecture".to_string(),
                    value_type: GgufValueType::String,
                    value: GgufKvValue::String("llama".to_string()),
                },
                GgufKvPair {
                    key: "general.file_type".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(6),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };

        let arch = header.get_kv_str("general.architecture");
        assert_eq!(arch, Some("llama"));

        let ft = header.get_kv_u32("general.file_type");
        assert_eq!(ft, Some(6));

        let missing = header.get_kv_str("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_file_type_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "ft".to_string(),
                    value_type: GgufValueType::String,
                    value: GgufKvValue::String("Q4_0".to_string()),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.file_type(), Some("Q4_0".to_string()));
    }

    #[test]
    fn test_context_length_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "n_ctx".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(8192),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.context_length(), Some(8192));
    }

    #[test]
    fn test_embedding_length_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "n_embd".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(4096),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.embedding_length(), Some(4096));
    }

    #[test]
    fn test_block_count_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "n_layer".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(32),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.block_count(), Some(32));
    }

    #[test]
    fn test_attention_head_count_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "n_head".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(32),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.attention_head_count(), Some(32));
    }

    #[test]
    fn test_rope_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "rope_type".to_string(),
                    value_type: GgufValueType::String,
                    value: GgufKvValue::String("yarn".to_string()),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.rope_scaling_type(), Some("yarn"));
    }

    #[test]
    fn test_normalization_epsilon_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "rms_norm_eps".to_string(),
                    value_type: GgufValueType::Float32,
                    value: GgufKvValue::Float32(1e-6),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.normalization_epsilon(), Some(1e-6));
    }

    #[test]
    fn test_vocab_size_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "n_vocab".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(32000),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.vocab_size(), Some(32000));
    }

    #[test]
    fn test_quantization_description_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "quantization".to_string(),
                    value_type: GgufValueType::String,
                    value: GgufKvValue::String("Q4_0".to_string()),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.quantization_description(), Some("Q4_0"));
    }

    #[test]
    fn test_architecture_fallback_arch_key() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "arch".to_string(),
                    value_type: GgufValueType::String,
                    value: GgufKvValue::String("mistral".to_string()),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.architecture(), Some("mistral"));
    }

    #[test]
    fn test_file_type_from_u32_key() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "general.file_type".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(7),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.file_type(), Some("7".to_string()));
    }

    #[test]
    fn test_file_type_from_ft_u32_key() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "ft".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(8),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.file_type(), Some("8".to_string()));
    }

    #[test]
    fn test_feed_forward_length_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "n_ff".to_string(),
                    value_type: GgufValueType::Uint32,
                    value: GgufKvValue::Uint32(11008),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.feed_forward_length(), Some(11008));
    }

    #[test]
    fn test_rope_dimension_count_fallback_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![
                GgufKvPair {
                    key: "rope_dim".to_string(),
                    value_type: GgufValueType::Int32,
                    value: GgufKvValue::Int32(128),
                },
            ],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.rope_dimension_count(), Some(128));
    }

    #[test]
    fn test_value_type_is_array() {
        assert!(GgufValueType::Array.is_array());
        assert!(!GgufValueType::Uint32.is_array());
        assert!(!GgufValueType::String.is_array());
    }

    #[test]
    fn test_value_type_to_u32_roundtrip() {
        let types = vec![
            GgufValueType::Uint8, GgufValueType::Int8, GgufValueType::Uint16,
            GgufValueType::Int16, GgufValueType::Uint32, GgufValueType::Int32,
            GgufValueType::Uint64, GgufValueType::Int64, GgufValueType::String,
            GgufValueType::Float32, GgufValueType::Bool,
            GgufValueType::Array, GgufValueType::Int8Array, GgufValueType::Uint8Array,
            GgufValueType::Bfloat16, GgufValueType::Float16,
        ];
        for t in types {
            let raw = t.to_u32();
            assert_eq!(GgufValueType::from_u32(raw), Some(t), "failed for {:?}", t);
        }
    }

    #[test]
    fn test_value_type_from_u32_unmapped_reserved() {
        assert!(GgufValueType::from_u32(17).is_none());
        // Value 16 is now Float16, so it should return Some
        assert_eq!(GgufValueType::from_u32(16), Some(GgufValueType::Float16));
    }

    #[test]
    fn test_gguf_version_from_u32_unmapped() {
        assert!(GgufVersion::from_u32(0).is_none());
        assert!(GgufVersion::from_u32(4).is_none());
        assert!(GgufVersion::from_u32(99).is_none());
    }

    #[test]
    fn test_tensor_stored_size_q4_0_partial() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![32],
            offset: 0,
            dtype: 2, // Q4_0
        };
        // Q4_0: 32 elements = one partial block
        // full_blocks=0, remaining=32 => 2 + 32/2 = 18
        assert_eq!(info.stored_size(), 18);
    }

    #[test]
    fn test_total_tensor_bytes_f32_empty() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![],
            tensors: vec![],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.total_tensor_bytes_f32(), 0);
    }

    #[test]
    fn test_total_tensor_bytes_f32_single() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![],
            tensors: vec![GgufTensorInfo {
                name: "t".to_string(),
                shape: vec![100, 200],
                offset: 0,
                dtype: 0,
            }],
            data_alignment: None,
            data_section_start: 0,
        };
        assert_eq!(header.total_tensor_bytes_f32(), 100 * 200);
    }
}
