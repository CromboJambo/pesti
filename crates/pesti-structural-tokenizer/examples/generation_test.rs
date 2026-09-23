//! Test whether structural tokens are sufficient for code generation.
//! Feeds structural representation to pesti, asks for Rust reconstruction.

use std::process::Command;

fn main() {
    // The structural representation from the fibonacci experiment
    let structural_input = r#"FUNCTION_DECL: fn add(a: i32, b: i32) -> i32 
  BLOCK_START
  ReturnExpr: return a + b;
  BLOCK_END
FUNCTION_DECL: fn multiply(a: i32, b: i32) -> i32 
  BLOCK_START
  ReturnExpr: return a * b;
  BLOCK_END
FUNCTION_DECL: fn main() 
  BLOCK_START
  LET: result = add(5, 3);
  IF: if result > 10 { print("big"); } else { print("small"); }
  BLOCK_END"#;

    let prompt = format!(
        "Reconstruct a complete, compilable Rust program from this structural representation. \
         Output only the Rust code, no explanation.\n\n{}",
        structural_input
    );

    // Write prompt to file for pesti-runner
    std::fs::write("/tmp/struct_prompt.txt", &prompt).unwrap();

    println!("Prompt written to /tmp/struct_prompt.txt");
    println!("Run: pesti-runner --model <your_model.gguf> --prompt-file /tmp/struct_prompt.txt");
}
