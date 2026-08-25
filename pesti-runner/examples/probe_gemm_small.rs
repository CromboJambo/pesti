//! Small fast GEMM probe to isolate the last-tile / last-chunk column bug.
//!
//! Diagonal structure: A = ones[m x k], B[k][j] = 1.0 iff j == k else 0.0.
//! Expected: C[0][j] = 1.0 for j < k, 0.0 for j >= k.
//!
//! The real output-head case (m=1, k=896, n=151936) loses exactly columns
//! 892-895 (local cols 4-7 of the last diagonal tile = t=2,3 threads, last
//! k-chunk). This probe reproduces the same structure at tiny scale so we can
//! iterate without the 259MB allocation.
use half::f16;

fn run(m: usize, k: usize, n: usize) {
    let ctx = pesti_runner::kernel::DispatchContext::new();
    eprintln!(
        "  gpu_available={} arch={}",
        ctx.gpu_available(),
        ctx.gemm_arch().name()
    );
    let a: Vec<f16> = vec![f16::from_f32(1.0); m * k];
    let b: Vec<f16> = (0..k * n)
        .map(|i| {
            let kk = i / n;
            let j = i % n;
            f16::from_f32(if j == kk { 1.0 } else { 0.0 })
        })
        .collect();
    let c = ctx
        .dispatch_gemm(&a, &b, None, m, n, k, 1.0, 0.0)
        .expect("dispatch_gemm");
    let mut bad: Vec<(usize, f32, f32)> = Vec::new();
    for (j, &v) in c.iter().enumerate() {
        let exp = if j < k { 1.0f32 } else { 0.0 };
        if (v - exp).abs() > 1e-3 {
            bad.push((j, v, exp));
        }
    }
    let shown: Vec<(usize, f32, f32)> = bad.iter().take(16).cloned().collect();
    eprintln!(
        "m={} k={} n={} : bad_cols={} first16={:?}",
        m, k, n, bad.len(), shown
    );
}

fn main() {
    eprintln!("=== small diagonal GEMM probes ===");
    // 1 k-chunk, 8 tiles. Diagonal in cols 0-15.
    run(1, 16, 64);
    // 2 k-chunks, 8 tiles.
    run(1, 32, 64);
    // 1 chunk, 2 tiles (n==k).
    run(1, 16, 16);
    // Same k as real output-head (896), small n (12 tiles). Diagonal in cols 0-895
    // but n=96 so only cols 0-95 exist; last diagonal tile is tile 11 (cols 88-95).
    run(1, 896, 96);
}
