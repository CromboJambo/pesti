# Implementation Plan: Close GGUF Parser Gaps vs. Reference

## Priority 1: Byte-Order Detection

### Current Code (parser.rs lines 15-25)
```rust
pub fn parse<R: Read + Seek>(mut reader: R) -> Result<GgufMetadata, GgufError> {
    let magic = reader.read_u32::<LittleEndian>()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = reader.read_u32::<LittleEndian>()?;
```

### Reference Implementation (Python lines 130-145)
```python
temp_version = self._get(offs, np.uint32)
if temp_version[0] & 65535 == 0:
    # If we get 0 here that means it's a GGUF file created for
    # the opposite byte order of the machine this script is running on.
    self.byte_order = 'S'  # Swapped!
```

### Implementation Strategy
1. Read version as raw bytes first
2. Check if `version & 0xFFFF == 0` (indicates swapped endianness)
3. Set a flag to swap all subsequent reads
4. Apply byte swapping for u32, u64 reads

### Code Changes Needed
```rust
// Add to GgufMetadata struct
pub byte_order: ByteOrder, // LittleEndian or BigEndian

// Modify parse function
let version_raw = reader.read_u32::<LittleEndian>()?;
let version = if version_raw & 0xFFFF == 0 {
    // Swapped endianness detected
    ByteOrder::BigEndian
} else {
    ByteOrder::LittleEndian
};

// Create helper method for conditional swapping
fn read_with_byte_order<R: Read + Seek>(
    reader: &mut R, 
    value: u32, 
    byte_order: ByteOrder
) -> u32 {
    match byte_order {
        ByteOrder::LittleEndian => value,
        ByteOrder::BigEndian => value.to_le(), // or to_be() depending on read
    }
}
```

---

## Priority 2: Alignment Validation

### Current Code (parser.rs lines 70-90)
```rust
// Read KV pairs
for _ in 0..kv_count {
    let kv = read_kv_pair_v3(&mut reader)?;
    metadata.kv.push(kv);
}

// Jump to tensor section
let tensor_start = reader.read_u64::<LittleEndian>()?;
```

### Reference Implementation (Python lines 175-185)
```python
new_align = self.fields.get('general.alignment')
if new_align is not None:
    self.alignment = new_align.parts[-1][0]
    # Ensure alignment is a non-zero power of two
    if self.alignment == 0 or (self.alignment & (self.alignment - 1)) != 0:
        raise ValueError('Invalid alignment: must be a non-zero power of two')

padding = offs % self.alignment
if padding != 0:
    offs += self.alignment - padding
```

### Implementation Strategy
1. After reading KV pairs, check for `general.alignment` key
2. Extract the alignment value (should be u32)
3. Validate it's a non-zero power of two
4. Calculate padding needed before tensor section
5. Skip to aligned position instead of raw tensor_start

### Code Changes Needed
```rust
// After reading KV pairs, find alignment
let mut alignment = 32u32; // Default per reference
for kv in &metadata.kv {
    if kv.key == "general.alignment" {
        if let GgufKvValue::UInt32(align) = &kv.value {
            alignment = *align;
            // Validate power of two
            if alignment == 0 || (alignment & (alignment - 1)) != 0 {
                return Err(GgufError::InvalidAlignment(alignment));
            }
        }
    }
}

// Read tensor count and calculate aligned position
let tensor_count = reader.read_u64::<LittleEndian>()?;
let kv_end_offset = reader.stream_position()?;

// Apply alignment padding
let padding = kv_end_offset % alignment as u64;
if padding != 0 {
    reader.seek_relative((alignment - padding) as i64)?;
}

// Now read tensor info from aligned position
```

---

## Priority 3: String Length Limits

### Current Code (parser.rs lines 430-440)
```rust
let len = reader.read_u64::<LittleEndian>()?;
if len == 0 || len > 1024 * 1024 {  // Only 1MB!
    return Err(GgufError::KeyLengthOutOfRange);
}
```

### Reference Implementation (C++ lines 18-19, 215-220)
```cpp
#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)  // 1GB
// ...
if (size > GGUF_MAX_STRING_LENGTH) {
    GGML_LOG_ERROR("string length %" PRIu64 " exceeds maximum", size);
    return false;
}
```

### Implementation Strategy
1. Increase limit from 1MB to 1GB
2. Add separate limits for keys vs tensor names
3. Consider adding runtime-configurable limits

### Code Changes Needed
```rust
pub const GGUF_MAX_KEY_LENGTH: u64 = 1024 * 1024 * 1024; // 1GB
pub const GGUF_MAX_STRING_LENGTH: u64 = 1024 * 1024 * 1024; // 1GB
pub const GGUF_MAX_TENSOR_NAME_LENGTH: u64 = 1024 * 1024 * 1024; // 1GB

// Update read_kv_pair_v3
let key_len = reader.read_u64::<LittleEndian>()?;
if key_len == 0 || key_len > GGUF_MAX_KEY_LENGTH {
    return Err(GgufError::KeyLengthOutOfRange);
}

// Update read_tensor_info_v3
let name_len = reader.read_u64::<LittleEndian>()?;
if name_len == 0 || name_len > GGUF_MAX_TENSOR_NAME_LENGTH {
    return Err(GgufError::TensorNameLengthOutOfRange);
}
```

---

## Priority 4: Array Element Limit

### Reference Implementation (C++ lines 19)
```cpp
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024) // 1B elements
```

### Current Code
No explicit limit on array lengths.

### Implementation Strategy
Add validation for array element counts to prevent memory exhaustion.

### Code Changes Needed
```rust
pub const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1024 * 1024 * 1024; // 1B

// In read_kv_value_v3
let array_len = reader.read_u64::<LittleEndian>()?;
if array_len > GGUF_MAX_ARRAY_ELEMENTS {
    return Err(GgufError::ArrayLengthOutOfRange);
}

// Also check for overflow when allocating Vec
if array_len > usize::MAX as u64 {
    return Err(GgufError::ArrayTooLarge);
}
```

---

## Priority 5: Enhanced Error Types

### Add to GgufError enum (parser.rs lines 10-30)
```rust
pub enum GgufError {
    // Existing variants...
    IoError(#[source] std::io::Error),
    Utf8Error(#[from] std::string::FromUtf8Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    KeyLengthOutOfRange,
    TensorNameLengthOutOfRange,
    TruncatedFile,
    
    // New variants from reference comparison
    InvalidAlignment(u32),           // Priority 2
    ArrayLengthOutOfRange,           // Priority 4
    ArrayTooLarge,                   // Priority 4
    AlignmentPaddingError,           // Priority 2
}
```

---

## Testing Strategy

### Test Cases to Add
1. **Byte-order test**: Create swapped-endianness GGUF file (or find one)
2. **Alignment test**: Parse file with non-default alignment (e.g., 64-byte)
3. **String length test**: Try parsing 500MB string (should succeed now)
4. **Array limit test**: Verify 1B element arrays are rejected

### Integration Test
```rust
#[test]
fn test_parse_with_alignment() {
    let metadata = GgufMetadata::parse_file("models/test_aligned.gguf").unwrap();
    assert_eq!(metadata.alignment, 32); // Or whatever the file specifies
}

#[test]
fn test_string_length_limit() {
    // Try parsing a file with very long string (should succeed up to 1GB)
    let large_string = "x".repeat(100 * 1024 * 1024); // 100MB
    // ... create GGUF with this string ...
    // Should parse successfully
}
```

---

## Implementation Order

### Phase 1: Critical Fixes (Week 1)
1. ✅ String length limits (Priority 3) - Quick win, low risk
2. ⚠️ Alignment validation (Priority 2) - Affects correctness
3. ⚠️ Array element limit (Priority 4) - Memory safety

### Phase 2: Advanced Features (Week 2-3)
4. ⚡ Byte-order detection (Priority 1) - Complex, needs testing
5. 🎯 Enhanced error types (Priority 5) - Improves DX

### Phase 3: Validation (Week 4)
6. 🧪 Add conformance tests for new features
7. 📊 Benchmark against reference implementation

---

## Risk Assessment

| Change | Risk Level | Reason |
|--------|------------|--------|
| String length limits | Low | Just increasing limit, no logic change |
| Alignment validation | Medium | Could break existing parsers if file is malformed |
| Byte-order detection | High | Needs thorough testing with real files |
| Array element limit | Low | Simple validation, prevents OOM |

---

## Success Criteria

### Definition of "Reference-Aligned"
1. ✅ Parse all files that reference implementation parses
2. ✅ Reject all files that reference implementation rejects  
3. ✅ Same error messages (where applicable)
4. ✅ Handle edge cases identically (byte-order, alignment, etc.)

### Metrics
- **Conformance rate**: 100% on llama.cpp test suite
- **Error coverage**: All GgufError variants tested
- **Performance**: <5% overhead from additional checks

---

## References
- `/home/crombo/projects/pesti/GGUF_PARSER_SPECIFICATION.md` - Full gap analysis
- `ggml-org/llama.cpp/gguf-py/gguf/gguf_reader.py` - Reference implementation
- `ggml-org/llama.cpp/gguf/src/gguf.cpp` - C++ reference (57KB)

---

**Created**: 2026-08-07  
**Target Completion**: 4 weeks (1 month sprint)
