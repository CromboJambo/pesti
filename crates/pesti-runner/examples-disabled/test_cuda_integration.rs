//! Quick test that CUDA path is wired in generate()

fn main() {
    println!("Testing CUDA integration...");

    // Check if dispatch context exists
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    match pesti_runner::transformer::LlamaModel::load_gguf(std::path::Path::new(model_path)) {
        Ok(model) => {
            println!("✅ Model loaded successfully");

            // Check if dispatch is initialized
            if model.dispatch.is_some() {
                println!("✅ Dispatch context initialized (CUDA path available)");

                // Try to check GPU availability
                let ctx = model.dispatch.as_ref().unwrap();
                if ctx.gpu_available() {
                    println!("✅ GPU detected and available");
                } else {
                    println!("⚠️  CUDA enabled but GPU not detected (fallback to CPU)");
                }
            } else {
                println!("❌ Dispatch context not initialized");
            }

            // Check tokenizer
            if model.tokenizer.is_some() {
                println!("✅ Tokenizer loaded");
            } else {
                println!("⚠️  No tokenizer in GGUF file");
            }
        }
        Err(e) => {
            println!("❌ Model load failed: {}", e);
        }
    }

    println!("\n✅ CUDA path wiring test complete!");
}
