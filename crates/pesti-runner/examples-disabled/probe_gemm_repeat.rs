//! In-process repeat probe with failure-pattern detail.
//!
//! For each failing launch, prints the bad-column positions (capped) so the
//! pattern (last-tile columns? first columns? random?) is visible.
use half::f16;

fn run_once(
    ctx: &pesti_runner::kernel::DispatchContext,
    m: usize,
    k: usize,
    n: usize,
) -> (usize, bool, Vec<usize>) {
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
    let mut bad: Vec<usize> = Vec::new();
    let mut all_zero = true;
    for (j, &v) in c.iter().enumerate() {
        let exp = if j < k { 1.0f32 } else { 0.0 };
        if (v - exp).abs() > 1e-3 {
            bad.push(j);
        }
        if v != 0.0 {
            all_zero = false;
        }
    }
    (bad.len(), all_zero, bad)
}

fn main() {
    let ctx = pesti_runner::kernel::DispatchContext::new();
    eprintln!(
        "gpu_available={} arch={}",
        ctx.gpu_available(),
        ctx.gemm_arch().name()
    );
    let iters = 30usize;
    for (m, k, n) in [
        (1usize, 16usize, 64usize),
        (1usize, 32usize, 64usize),
        (1usize, 896usize, 96usize),
    ] {
        let mut correct = 0usize;
        let mut all_zero = 0usize;
        let mut partial = 0usize;
        let mut shown = 0usize;
        for iter in 0..iters {
            let (bad, az, cols) = run_once(&ctx, m, k, n);
            if az {
                all_zero += 1;
                if shown < 3 {
                    eprintln!(
                        "  [{}] k={} n={} iter={} ALL-ZERO ({} cols)",
                        m, k, n, iter, n
                    );
                    shown += 1;
                }
            } else if bad == 0 {
                correct += 1;
            } else {
                partial += 1;
                if shown < 6 {
                    let first: Vec<usize> = cols.iter().take(20).cloned().collect();
                    eprintln!(
                        "  [{}] k={} n={} iter={} PARTIAL bad={} first20={:?}",
                        m, k, n, iter, bad, first
                    );
                    shown += 1;
                }
            }
        }
        eprintln!(
            "m={} k={} n={} : correct={} all_zero={} partial={} (of {})",
            m, k, n, correct, all_zero, partial, iters
        );
    }
    eprintln!("fallback_count={}", ctx.gpu_fallback_count());
}
