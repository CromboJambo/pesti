//! Systematically probe the mma.sync GEMM k-tail handling.
//!
//! For each k in a sweep, run m=1, n=k (square-ish), A=ones, B=identity so
//! expected C[0][j] = 1 for j<k else 0. Reports which columns are wrong.
//!
//! Hypothesis: partial k-tails (k % 16 != 0) drop in-bounds elements because
//! the PTX zeroes the whole fragment register pair when only the SECOND f16
//! of the pair is out of bounds.
use half::f16;

fn run(
    ctx: &pesti_runner::kernel::DispatchContext,
    m: usize,
    n: usize,
    k: usize,
) -> (usize, Vec<usize>, bool) {
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
    for i in 0..m {
        for j in 0..n {
            let exp = if j < k { 1.0f32 } else { 0.0 };
            if (c[i * n + j] - exp).abs() > 1e-3 {
                bad.push(j);
            }
            if c[i * n + j] != 0.0 {
                all_zero = false;
            }
        }
    }
    (bad.len(), bad, all_zero)
}

fn main() {
    let ctx = pesti_runner::kernel::DispatchContext::new();
    eprintln!("gpu={} arch={}", ctx.gpu_available(), ctx.gemm_arch().name());
    // Sweep k values: multiples of 16 (full tiles) and partial tails.
    let ks: Vec<usize> = (1..=48)
        .chain([64usize, 80, 896, 1024])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut fails = 0usize;
    for k in ks {
        let n = k; // square
        let (bad, cols, az) = run(&ctx, 1, n, k);
        let tag = if bad == 0 {
            "ok "
        } else if az {
            "ZERO"
        } else {
            "PART"
        };
        if bad > 0 {
            fails += 1;
        }
        let kmod = k % 16;
        let first: Vec<usize> = cols.iter().take(12).cloned().collect();
        eprintln!(
            "k={k:4} (k%16={kmod:2}) n={n:4} bad={bad:3} {tag} first={first:?}"
        );
    }
    eprintln!("total failing k values: {fails}");
    eprintln!("fallback_count={}", ctx.gpu_fallback_count());
}
