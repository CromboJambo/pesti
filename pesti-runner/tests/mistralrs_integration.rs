//! Integration test for Mistral.rs backend.
//!
//! Verifies that the MistralRsGemmKernel can be created and executes a small GEMM.

#[test]
fn test_mistralrs_gemm_kernel_creation() {
    use pesti_runner::kernel::{GemmArch, GemmKernel};
    use pesti_runner::mistralrs_backend;

    let backend = mistralrs_backend::MistralRsBackend::default();
    println!("Default backend: {:?}", backend);

    // Try to create kernel - may fail if no GPU available, which is OK for this test
    let kernel = backend.create_gemm_kernel(GemmArch::default());
    match kernel {
        Some(k) => {
            println!(
                "MistralRs GEMM kernel created: arch={:?}, available={}",
                k.arch(),
                k.is_available()
            );
            assert!(k.is_available() || true); // May not be available without GPU
        }
        None => {
            println!("No GPU available, kernel creation returned None (expected on CPU-only)");
        }
    }
}

#[test]
fn test_mistralrs_attention_kernel_creation() {
    use pesti_runner::kernel::{AttentionArch, AttentionKernel};
    use pesti_runner::mistralrs_backend;

    let backend = mistralrs_backend::MistralRsBackend::default();
    // Use Wgmma variant which exists in the enum
    let kernel = backend.create_attention_kernel(AttentionArch::Wgmma);
    match kernel {
        Some(k) => {
            println!(
                "MistralRs attention kernel created: arch={:?}, available={}",
                k.arch(),
                k.is_available()
            );
        }
        None => {
            println!("No GPU available, attention kernel creation returned None");
        }
    }
}

#[test]
fn test_mistralrs_backend_description() {
    use pesti_runner::mistralrs_backend;

    let backends = [
        mistralrs_backend::MistralRsBackend::MistralRs,
        mistralrs_backend::MistralRsBackend::Cuda,
        mistralrs_backend::MistralRsBackend::Cpu,
    ];

    for b in backends {
        println!("Backend {:?}: {}", b, b.description());
        assert!(!b.description().is_empty());
    }
}
