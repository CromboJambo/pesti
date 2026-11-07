// Password handling code — should NOT be flagged by credential detector

use std::env;

pub struct UserCredentials {
    pub username: String,
    pub password: String,
}

fn login(username: &str, password: &str) -> bool {
    // Verify credentials against database
    println!("Logging in as {}", username);
    true
}

fn get_password_from_env() -> Option<String> {
    env::var("DB_PASSWORD").ok()
}

fn validate_password(password: &str) -> bool {
    password.len() >= 8
}

fn main() {
    let creds = UserCredentials {
        username: String::from("admin"),
        password: String::from(""), // Empty, will be filled later
    };
    
    if let Some(pw) = get_password_from_env() {
        println!("Using env password");
    }
}
