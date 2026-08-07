use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(String),

    #[error("invalid magic number: {0}")]
    InvalidMagic(String),

    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("invalid value type: {0}")]
    InvalidValueType(u32),

    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("binary read error: {0}")]
    Binary(#[from] std::io::Error),

    #[error("unexpected end of file - expected {expected} bytes, got {actual}")]
    UnexpectedEof { expected: usize, actual: usize },

    #[error("key length out of range: {0} (max 1GB)")]
    KeyLengthOutOfRange(u64),

    #[error("tensor name length out of range: {0} (max 1GB)")]
    TensorNameLengthOutOfRange(u64),

    #[error("invalid tensor data: {0}")]
    InvalidTensor(String),

    #[error("quantization not supported: {0}")]
    QuantizationNotSupported(String),

    #[error("alignment value out of range: {0} (must be power of 2, max 1GB)")]
    AlignmentOutOfRange(u32),

    #[error("array length out of range: {0} (max 1B elements)")]
    ArrayLengthOutOfRange(u64),

    #[error("array too large to fit in memory: {0} elements")]
    ArrayTooLarge(usize),
}
