use std::io::Cursor;

fn main() {
    // Generate minimal v3 GGUF bytes (copied from parser tests)
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(b"GGUF");

    // Version
    buf.extend_from_slice(&3u32.to_le_bytes());

    // Tensor count
    buf.extend_from_slice(&2u64.to_le_bytes());

    // KV count - v3 format uses u64 here (per spec)
    buf.extend_from_slice(&3u64.to_le_bytes());

    // First KV pair: general.alignment = 32 (must come first for data section calculation)
    let key = "general.alignment";
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32 type
    buf.extend_from_slice(&32u32.to_le_bytes());

    // KV pair 1: general.architecture = "llama" (string)
    let key = "general.architecture";
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes()); // STRING type (llama.cpp uses 8, not 10 per spec)
    buf.extend_from_slice(&5u64.to_le_bytes()); // "llama" length (v3 uses u64 for string values)
    buf.extend_from_slice(b"llama");

    // KV pair 2: general.file_type = 1 (F16) (uint32)
    let key = "general.file_type";
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32 type
    buf.extend_from_slice(&1u32.to_le_bytes());

    // Tensor 1: token_embd.weight (shape [4096], dtype F16, offset 0)
    let name = "token_embd.weight";
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes()); // v3 spec uses u64 for tensor names
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim (v3 spec: tensor ndims is u32)
    buf.extend_from_slice(&4096u64.to_le_bytes()); // shape[0]
    buf.extend_from_slice(&1u32.to_le_bytes()); // dtype F16
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Tensor 2: output.weight (shape [4096, 32000], dtype F16, offset after tensor 1)
    let name = "output.weight";
    buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
    buf.extend_from_slice(&4096u64.to_le_bytes()); // shape[0]
    buf.extend_from_slice(&32000u64.to_le_bytes()); // shape[1]
    buf.extend_from_slice(&1u32.to_le_bytes()); // dtype F16
    buf.extend_from_slice(&(4096 * 32000u64).to_le_bytes()); // offset (F16 = 2 bytes per element)

    println!("Generated {} bytes", buf.len());
    println!("\nFirst 100 bytes:");
    for i in 0..100.min(buf.len()) {
        print!("{:02X} ", buf[i]);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }

    // Now try to parse it with the gguf crate
    match testi_gguf::parser::parse_gguf_reader(Cursor::new(&buf)) {
        Ok(header) => {
            println!("\n✅ Successfully parsed GGUF v{}", header.version);
            println!("KV pairs: {}", header.kv_pairs.len());
            println!("Tensors: {}", header.tensors.len());
            
            if let Some(arch) = header.architecture() {
                println!("Architecture: {}", arch);
            }
            
            for kv in &header.kv_pairs {
                println!("  KV: {} = {:?}", kv.key, kv.value);
            }
            
            for tensor in &header.tensors {
                println!("  Tensor: {} shape={:?} dtype={} offset={}", 
                    tensor.name, tensor.shape, tensor.dtype, tensor.offset);
            }
        }
        Err(e) => {
            println!("\n❌ Parse error: {}", e);
        }
    }
}
