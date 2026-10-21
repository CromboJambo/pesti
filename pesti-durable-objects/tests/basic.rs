use std::path::PathBuf;

#[tokio::test]
async fn test_save_and_load_snapshot() {
    let path = PathBuf::from("/tmp/test_durable_object.json");

    // Save a snapshot
    let data = serde_json::json!({
        "id": "test-obj-1",
        "name": "Test Object",
        "version": 1,
        "data": {"foo": "bar"}
    });

    pesti_durable_objects::save_snapshot(&path, &data).await.unwrap();

    // Load it back
    let loaded: serde_json::Value = pesti_durable_objects::load_snapshot(&path).await.unwrap();

    assert_eq!(loaded["id"], "test-obj-1");
    assert_eq!(loaded["name"], "Test Object");
    assert_eq!(loaded["version"], 1);

    // Cleanup
    tokio::fs::remove_file(&path).await.unwrap();
}

#[tokio::test]
async fn test_object_id_generation() {
    let id = pesti_durable_objects::ObjectId::generate();
    assert!(!id.as_str().is_empty());
}