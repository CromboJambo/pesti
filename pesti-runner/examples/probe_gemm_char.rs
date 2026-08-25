//! Characterize the mma.sync GEMM bug precisely.
//!
//! The 2x2 probe returned all zeros while 1x4 worked. The output-head probe
//! returned 892/896 correct with the last 4 diagonal columns lost. This
//! probe runs a series of controlled GEMMs to find the exact failure mode:
//!
//!   1. 1x4 (known good) — control
//!   2. 2x2 (known bad, all zeros)
//!   3. 4x4 (m=4, k=4, n=4)
//!   4. 16x8 (exactly one mma tile, m=16, k=16, n=8)
//!   5. 16x16 (one mma tile in m, two in n)
//!   6. 8x8 (m < 16, k < 16, n < 8 — all tail)
//!   7. 1x16 (m=1, k=16, n=16)
//!   8. 1x1 (m=1, k=1, n=1 — degenerate)
//!
//! Each uses A = ones, B = identity-like so C[j] = 1 for j < min(k, n)
//! (diagonal hit) and 0 otherwise. Actually for ones*identity:
//! C[i][j] = sum_k A[i][k]*B[k][j] = 1 if j == i (for square) else 0...
//! No, with A=ones[m x k] and B=identity[k x n]: C[i][j] = sum_k 1*B[k][j]
//! = number of k where B[k][j]=1 = 1 if j < k (diagonal in B) else 0.
//! So C[i][j] = 1.0 for j < k, 0.0 for j >= k, for ALL rows i.
//!
//! This is the same pattern as the output-head probe. The failure mode
//! (last 4 cols lost at n=151936, all zeros at 2x2) will be visible.
//!
//! Build+run:
//!   cargo run -p pesti-runner --release --features cuda --example probe_gemm_char
use half::f16;

fn run_case(
    ctx: &pesti_runner::kernel::DispatchContext,
    name: &str,
    m: usize,
    n: usize,
    k: usize,
) {
    // A = ones[m x k], B = identity[k x n] (B[k][j] = 1 if k==j else 0)
    let a: Vec<f16> = vec![f16::from_f32(1.0); m * k];
    let b: Vec<f16> = (0..k * n)
        .map(|i| {
            let kk = i / n;
            let j = i % n;
            f16::from_f32(if j == kk { 1.0 } else { 0.0 })
        })
        .collect();
    let fb0 = ctx.gpu_fallback_count();
    let c = match ctx.dispatch_gemm(&a, &b, None, m, n, k, 1.0, 0.0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[{name}] m={m} n={n} k={k}: ERROR {e}");
            return;
        }
    };
    let fb1 = ctx.gpu_fallback_count();
    let fell_back = fb1 - fb0 > 0;

    // Expected: C[i][j] = 1.0 if j < k else 0.0 (for all i)
    let mut bad = 0usize;
    let mut first_bad = None;
    for i in 0..m {
        for j in 0..n {
            let exp = if j < k { 1.0f32 } else { 0.0 };
            let got = c[i * n + j];
            if (got - exp).abs() > 1e-3 {
                bad += 1;
                if first_bad.is_none() {
                    first_bad = Some((i, j, got, exp));
                }
            }
        }
    }
    let sum: f32 = c.iter().sum();
    let exp_sum = m as f32 * k.min(n) as f32;
    let status = if bad == 0 && !fell_back {
        "GPU OK"
    } else if bad == 0 && fell_back {
        "CPU FALLBACK (correct result)"
    } else {
        "WRONG"
    };
    eprintln!(
        "[{name}] m={m} n={n} k={k}: sum={sum:.1} (exp {exp_sum:.1}) bad={bad} first_bad={first_bad:?} fallback={fell_back} => {status}"
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = pesti_runner::kernel::DispatchContext::new();
    eprintln!(
        "gpu_available={} gemm_arch={}",
        ctx.gpu_available(),
        ctx.gemm_arch().name()
    );
    eprintln!("(A=ones, B=identity; expected C[i][j]=1 if j<k else 0)");
    eprintln!();

    run_case(&ctx, "control-1x4", 1, 4, 4);
    run_case(&ctx, "bad-2x2", 2, 2, 2);
    run_case(&ctx, "4x4", 4, 4, 4);
    run_case(&ctx, "8x8-all-tail", 8, 8, 8);
    run_case(&ctx, "16x8-one-tile", 16, 8, 16);
    run_case(&ctx, "16x16", 16, 16, 16);
    run_case(&ctx, "1x16", 1, 16, 16);
    run_case(&ctx, "1x1-degenerate", 1, 1, 1);
    run_case(&ctx, "3x3-sub-tile", 3, 3, 3);
    run_case(&ctx, "24x24-mid", 24, 24, 24);

    Ok(())
}
