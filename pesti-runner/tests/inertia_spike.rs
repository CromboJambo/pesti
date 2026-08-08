//! Spike test for computational inertia concept.
//!
//! Demonstrates that PESTI can log demand when GPU unavailable and replay later.

use pesti_runner::inertia::{InertiaManager, WorkResult};

fn main() {
    println!("=== Computational Inertia Spike Test ===\n");

    // Test 1: InertiaManager basic functionality
    test_inertia_manager_basic();

    // Test 2: InertiaManager with simulated GPU dropout
    test_gpu_dropout_scenario();

    println!("\n=== All spike tests passed ===");
}

fn test_inertia_manager_basic() {
    println!("[Test 1] Basic inertia manager functionality");

    let mut manager = InertiaManager::new(100);

    // GPU available: work executes immediately
    manager.set_gpu_available(true);
    let result = manager.request_work(pesti_runner::inertia::WorkType::Gemm {
        m: 128,
        n: 128,
        k: 4096,
        alpha: 1.0,
        beta: 0.0,
    });
    assert!(matches!(result, WorkResult::ReadyForExecution(_)));
    println!("  ✓ GPU available → ReadyForExecution");

    // GPU unavailable: work logged
    manager.set_gpu_available(false);
    let result = manager.request_work(pesti_runner::inertia::WorkType::Gemm {
        m: 256,
        n: 256,
        k: 8192,
        alpha: 1.0,
        beta: 0.0,
    });
    assert!(matches!(result, WorkResult::LoggedForLater));
    println!("  ✓ GPU unavailable → LoggedForLater");

    // GPU returns: drain queue
    manager.set_gpu_available(true);
    let pending = manager.get_pending_for_execution();
    assert_eq!(pending.len(), 1);
    println!(
        "  ✓ GPU returned → drained {} pending work item(s)",
        pending.len()
    );

    // Check stats
    let stats = manager.stats();
    assert_eq!(stats.total_work_logged, 2);
    assert_eq!(stats.total_work_executed, 1);
    println!(
        "  ✓ Stats: logged={}, executed={}",
        stats.total_work_logged, stats.total_work_executed
    );

    println!("[Test 1] PASSED\n");
}

fn test_gpu_dropout_scenario() {
    println!("[Test 2] Simulated GPU dropout/recovery scenario");

    let mut manager = InertiaManager::new(50); // small queue for backpressure test

    // Phase 1: Normal operation (GPU available)
    manager.set_gpu_available(true);
    for _ in 0..10 {
        manager.request_work(pesti_runner::inertia::WorkType::Gemm {
            m: 64,
            n: 64,
            k: 2048,
            alpha: 1.0,
            beta: 0.0,
        });
    }
    println!("  Phase 1: GPU available → 10 works executed immediately");

    // Phase 2: GPU dropout (simulated hardware failure)
    manager.set_gpu_available(false);
    let mut logged_count = 0;
    for _ in 0..20 {
        let result = manager.request_work(pesti_runner::inertia::WorkType::Attention {
            query_seq_len: 1,
            num_heads: 8,
            head_dim: 64,
            cache_seq_len: 128,
        });
        match result {
            WorkResult::LoggedForLater => logged_count += 1,
            WorkResult::Dropped => println!("  (backpressure: work dropped)"),
            _ => panic!("Expected LoggedForLater or Dropped"),
        }
    }
    assert!(logged_count > 0);
    println!(
        "  Phase 2: GPU dropout → {} works logged to queue",
        logged_count
    );

    // Phase 3: GPU recovery
    manager.set_gpu_available(true);
    let pending = manager.get_pending_for_execution();
    println!(
        "  Phase 3: GPU recovered → {} pending items available for replay",
        pending.len()
    );

    // Check stats show the full lifecycle
    let stats = manager.stats();
    assert_eq!(stats.total_work_logged, 30); // 10 + 20
    println!(
        "  ✓ Total logged: {}, pending drained: {}",
        stats.total_work_logged,
        pending.len()
    );

    println!("[Test 2] PASSED\n");
}
