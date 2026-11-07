// Real hardcoded secrets — should be flagged by credential detector

fn main() {
    // These are actual hardcoded credentials
    let password = "hunter2";
    let api_key = "sk-abc123";
    let token = "my-secret-token";
    
    println!("Password: {}", password);
}
