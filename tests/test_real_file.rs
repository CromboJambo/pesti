use std::io::{Read, Seek};

fn main() {
    let path = "/home/crombo/projects/llm-workspace/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("Opening: {}", path);
    
    match std::fs::File::open(path) {
        Ok(mut file) => {
            println!("Seeking to 0");
            file.seek(std::io::SeekFrom::Start(0)).expect("seek failed");
            
            let mut magic_buf = [0u8; 4];
            file.read_exact(&mut magic_buf).expect("read failed");
            
            println!("Magic bytes: {:?}", String::from_utf8_lossy(&magic_buf));
        }
        Err(e) => {
            eprintln!("Open error: {}", e);
        }
    }
}
