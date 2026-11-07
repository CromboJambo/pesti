// SQL injection patterns — should be detected

fn build_query_bad(user_id: String) -> String {
    // Vulnerable: string concatenation in query
    let query = "SELECT * FROM users WHERE id = " + user_id;
    query
}

fn build_query_good(user_id: i32) -> String {
    // Safe: parameterized (in real code this would use a DB library)
    format!("SELECT * FROM users WHERE id = {}", user_id)
}

fn main() {
    let query1 = build_query_bad("1".to_string());
    let query2 = build_query_good(1);
    println!("{}", query1);
    println!("{}", query2);
}
