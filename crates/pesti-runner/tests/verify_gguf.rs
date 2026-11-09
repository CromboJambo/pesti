// Verify the test GGUF file byte-by-byte
use std::fs;

fn main() {
    let buf = fs::read("/tmp/test.gguf").expect("read /tmp/test.gguf");
    println!("File size: {}", buf.len());

    // Print bytes around position 1026 (where tensor 5's offset should be)
    println!("Bytes at position 1020-1034:");
    for i in 1020..1034 {
        println!("  [{}] = 0x{:02X}", i, buf[i]);
    }

    // Read the u64 at position 1026 (little-endian)
    use std::convert::TryInto;
    let offset_bytes: [u8; 8] = buf[1026..1034].try_into().unwrap();
    let offset = u64::from_le_bytes(offset_bytes);
    println!("u64 at position 1026: {}", offset);

    // Print tensor info entries
    let mut pos = 590; // 24 + kv_size
    println!("\nTensor info entries:");
    for i in 0..6 {
        let name_len = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let name = String::from_utf8_lossy(&buf[pos..pos + name_len as usize]);
        pos += name_len as usize;
        let n_dims = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let dims: Vec<u64> = (0..n_dims)
            .map(|_| {
                let d = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                pos += 8;
                d
            })
            .collect();
        let dtype = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let offset = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        println!(
            "  {}: name={}({} chars) dims={} shape={:?} dtype={} offset={}",
            i, name, name_len, n_dims, dims, dtype, offset
        );
    }
}
