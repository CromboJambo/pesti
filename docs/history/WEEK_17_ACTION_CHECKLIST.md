# Week 17 - Immediate Action Checklist ✅

**Date**: August 18, 2026  
**Status**: Ready to execute

---

## 🚀 Phase 1: Baseline Profiling (Start Today)

### Step 1: Create profiling example
```bash
cd /home/crombo/projects/pesti/pesti-runner/examples
cat > profiling.rs << 'EOF'
//! Measure token generation throughput and timing breakdown

use pesti_runner::transformer::LlamaModel;
use std::time::{Duration, Instant};

fn main() {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "Once upon a time in the land of Rust,";
    let target_tokens = 100;

    println!("=== Baseline Profiling ===");
    println!("Model: {}", model_path);
    println!("Prompt: {}", prompt);
    println!("Target tokens: {}", target_tokens);
    println!();

    // Load model
    let start_load = Instant::now();
    let model = match LlamaModel::load_gguf(std::path::Path::new(model_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Load failed: {}", e);
            std::process::exit(1);
        }
    };
    println!("✅ Model loaded in {:.2}s", start_load.elapsed().as_secs_f64());
    println!();

    // Generate tokens with timing
    let start_gen = Instant::now();
    match model.generate(prompt, target_tokens) {
        Ok(tokens) => {
            let gen_time = start_gen.elapsed();
            let throughput = target_tokens as f64 / gen_time.as_secs_f64();

            println!("✅ Generated {} tokens in {:.2}s", tokens.len(), gen_time.as_secs_f64());
            println!("📊 Throughput: {:.2} tok/s", throughput);
            println!();

            // Decode first few tokens for verification
            if let Ok(decoded) = model.tokenizer.as_ref().unwrap().decode(&tokens[..10]) {
                println!("Sample output: \"{}\"", decoded);
            }
        }
        Err(e) => {
            eprintln!("Generation failed: {}", e);
            std::process::exit(1);
        }
    }

    println!();
    println!("=== Baseline Complete ===");
}
EOF
```

### Step 2: Run baseline test
```bash
cd /home/crombo/projects/pesti
cargo run --package pesti-runner --example profiling --features cuda 2>&1 | tee /tmp/baseline_result.txt
```

### Step 3: Repeat 5 times for consistency
```bash
for i in {1..5}; do
    echo "=== Run $i ===" >> /tmp/baseline_results.txt
    cargo run --package pesti-runner --example profiling --features cuda 2>&1 | grep -E "Throughput|tokens in" >> /tmp/baseline_results.txt
done
```

---

## 🔧 Phase 2: CUDA Verification (Day 2)

### Step 1: Add dispatch logging
**File**: `pesti-runner/src/transformer/model.rs`

Find line ~1416 and add:
```rust
// Pass through transformer layers - use GPU if dispatch is available
let logits_hidden = if self.dispatch.is_some() {
    tracing::debug!("Using GPU dispatch for forward pass");
    self.forward_with_dispatch(&hidden, pos)?
} else {
    tracing::warn!("Falling back to CPU (no CUDA)");
    self.forward_layers(&hidden, pos)?
};
```

### Step 2: Create CUDA test harness
**File**: `pesti-runner/examples/cuda_test.rs`

```rust
//! Verify CUDA dispatch is active

use pesti_runner::transformer::LlamaModel;

fn main() {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("=== CUDA Verification ===");
    
    let model = LlamaModel::load_gguf(std::path::Path::new(model_path)).unwrap();
    
    if let Some(dispatch) = &model.dispatch {
        println!("✅ Dispatch context initialized");
        
        // Check GPU availability
        match dispatch.gpu_available() {
            true => println!("✅ GPU detected and available"),
            false => println!("⚠️  CUDA enabled but GPU not detected (CPU fallback)"),
        }
    } else {
        println!("❌ Dispatch context not initialized");
    }
}
```

### Step 3: Run CUDA test
```bash
cargo run --package pesti-runner --example cuda_test --features cuda 2>&1 | grep "✅\|⚠️"
```

---

## 📊 Phase 3: Real Tokenizer Integration (Day 3-4)

### Step 1: Check current tokenizer usage
```bash
grep -r "fallback_tokenizer\|whitespace" pesti-runner/src/ | grep -v ".git"
```

### Step 2: Create integration wrapper
**File**: `pesti-runner/src/tokenizer/qwen2_bpe.rs`

```rust
//! Wrapper for qwen2-bpe crate

use pesti_runner::error::{Result, RunnerError};
use qwen2_bpe::Qwen2BPE;

pub struct Qwen2BpeTokenizer {
    inner: Qwen2BPE,
}

impl Qwen2BpeTokenizer {
    pub fn from_gguf(path: &std::path::Path) -> Result<Self> {
        let bpe = Qwen2BPE::from_gguf(path)?;
        Ok(Self { inner: bpe })
    }
}

impl crate::transformer::PestiTokenizer for Qwen2BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, RunnerError> {
        self.inner.encode(text).map_err(|e| e.into())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, RunnerError> {
        self.inner.decode(tokens).map_err(|e| e.into())
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}
```

### Step 3: Update Cargo.toml
**File**: `pesti-runner/Cargo.toml`

Add feature flag:
```toml
[features]
default = []
rust-tokenizer = ["qwen2-bpe"]
cuda = ["cudarc"]
```

---

## 📈 Phase 4: Documentation (Day 5-7)

### Step 1: Create results document
**File**: `WEEK_17_PROFILING_RESULTS.md`

Start with template from Week 16 completion doc, add new metrics.

### Step 2: Create comparison chart
```markdown
| Configuration | Throughput (tok/s) | Latency (ms/tok) | Notes |
|---------------|-------------------|------------------|-------|
| CPU-only      | X.XX              | XXX.X            | Baseline |
| CUDA enabled  | Y.YY              | YYY.Y            | Speedup: Z.Z× |
```

### Step 3: Write optimization TODO list
See Week 17 plan for template.

---

## ✅ Quick Wins (Do First)

1. **Create profiling example** - 5 minutes
2. **Run baseline test** - 2 minutes
3. **Add dispatch logging** - 3 minutes
4. **Document results** - 10 minutes

**Total time to get started**: ~20 minutes

---

## 🎯 Success Criteria for Today

- [ ] Baseline throughput measured (tok/s)
- [ ] Dispatch logging added to codebase
- [ ] CUDA verification test passes
- [ ] Results documented in temporary file

---

*Ready to execute: August 18, 2026*  
*First milestone: Baseline profiling complete by end of day*
