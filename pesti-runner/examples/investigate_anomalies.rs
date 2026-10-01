//! Anomaly investigation: linked_list_push and async_task across seeds.
//! Uses pesti-runner's synchronous llama API. Saves generated code for inspection.
//!
//! Matches original benchmark approach: simple code-signature prompts, SamplingConfig::precise().
//!
//! Usage: cargo run --example investigate_anomalies [model_path]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use pesti_runner::llama::{LlamaRunner, SamplingConfig};

// Same prompts as original benchmark — just the function signature to complete
const TASKS: &[(&str, &str)] = &[
    ("linked_list_push", "impl LinkedList {\n    fn push(&mut self, val: i32) {\n"),
    ("async_task", "async fn process(data: Vec<u8>) -> usize {\n"),
];

fn generate_code(runner: &LlamaRunner, prompt: &str, seed: i32) -> String {
    let mut sampling = SamplingConfig::precise();
    sampling.seed = seed as u32;
    match runner.generate(prompt, &sampling) {
        Ok(result) => result.text.trim().to_string(),
        Err(e) => {
            println!("Generation error: {}", e);
            String::new()
        }
    }
}

fn test_code(code: &str, out_dir: &PathBuf, name: &str, seed: i32) -> (&'static str, String) {
    let code_path = out_dir.join(format!("{}_seed{}.rs", name, seed));
    fs::write(&code_path, code).unwrap();

    // Compile
    let compile = std::process::Command::new("rustc")
        .args(["-o", "/tmp/test_prog", code_path.to_str().unwrap(), "--edition", "2021"])
        .output().unwrap();

    if !compile.status.success() {
        let err = String::from_utf8_lossy(&compile.stderr).to_string();
        return ("compile", err);
    }

    // Run
    let test = std::process::Command::new("/tmp/test_prog").output().unwrap();
    let result = String::from_utf8_lossy(&test.stdout).to_string();
    if result.contains("PASSED") {
        return ("ok", result);
    } else {
        return ("runtime", result);
    }
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "/home/crombo/projects/active/pesti/test_models/tinyllama-q4.gguf".to_string());
    println!("Model: {}", model_path);

    let start = Instant::now();

    for (task_name, prompt) in TASKS {
        println!("\n=== {} ===", task_name);
        println!("Prompt: {}", prompt.replace('\n', "\\n"));

        let out_dir = PathBuf::from("/tmp/anomaly_codes").join(task_name);
        fs::create_dir_all(&out_dir).unwrap();

        let mut passes = 0;
        for seed in 0..8 {
            // Fresh runner per seed to avoid KV cache issues
            let runner = LlamaRunner::builder(&model_path)
                .n_ctx(2048)
                .build()?;

            let code = generate_code(&runner, prompt, seed as i32);
            drop(runner);

            let (status, output) = test_code(&code, &out_dir, task_name, seed);
            if status == "ok" {
                passes += 1;
            }
            println!("  seed {:03}: {} ({})", seed, status, output.chars().take(40).collect::<String>());
        }

        println!("Pass rate: {}/8 ({:.0}%)", passes, passes as f64 / 8.0 * 100.0);
        println!("Code saved to: {}", out_dir.display());
    }

    println!("\nTotal time: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}
