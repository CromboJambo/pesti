//! Example: Fleet coordination using durable objects
//!
//! Demonstrates how crabjar uses pesti-durable-objects to manage agent task
//! state across the home-lab fleet, with persistence and versioning.

use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let object_id = pesti_durable_objects::ObjectId::generate();
    println!("Task object ID: {}", object_id);

    // Save task state snapshot
    let task_state = serde_json::json!({
        "task": "agent-deployment",
        "target_machine": "ftw3",
        "status": "running",
        "progress": 0.5,
        "retry_count": 2
    });

    let path = PathBuf::from("/tmp/fleet_task_snapshot.json");
    pesti_durable_objects::save_snapshot(&path, &task_state).await.unwrap();
    println!("Task state saved to snapshot");

    // Simulate crash/restart - load task state back
    let loaded: serde_json::Value = pesti_durable_objects::load_snapshot(&path).await.unwrap();
    println!("Restored task status: {}", loaded["status"]);
    println!("Progress: {}%", loaded["progress"].as_f64().unwrap() * 100.0);

    // Cleanup
    tokio::fs::remove_file(&path).await.unwrap();
}