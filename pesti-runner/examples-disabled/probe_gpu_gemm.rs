//! Minimal probe: does the raw GPU GEMM path (DispatchContext::dispatch_gemm)
//! actually compute a correct result, or does it silently return zeros?
//!
//! This isolates the output-head path that `forward_with_dispatch` uses
//! (ctx.dispatch_gemm), which is the ONLY thing exercising the raw cudarc
//! GEMM kernel — the per-layer linears go through dispatch_linear, which is
//! gated on candle_bridge::bridge_is_cuda() (false here) and thus falls back
//! to CPU. So if the GPU GEMM is broken, the output head is where it shows.
//!
//! Prints: gpu_available / backend / prefer_gpu, then a 2x2 GEMM and a
//! 1x8 GEMM (m=1 row-vector, like the output head) vs the expected f32 result.
//!
//! Build+run:
//!   cargo run -p pesti-runner --release --features cuda --example probe_gpu_gemm
use half::f16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = pesti_runner::kernel::DispatchContext::new();
    eprintln!(
        "gpu_available={} gemm_arch={}",
        ctx.gpu_available(),
        ctx.gemm_arch().name()
    );

    // 2x2 GEMM: A=[[1,2],[3,4]] B=[[5,6],[7,8]] -> C=[[19,22],[43,50]]
    let a: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let b: Vec<f16> = vec![5.0, 6.0, 7.0, 8.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let c = ctx.dispatch_gemm(&a, &b, None, 2, 2, 2, 1.0, 0.0)?;
    let exp2x2 = [19.0f32, 22.0, 43.0, 50.0];
    let ok2x2: Vec<bool> = c
        .iter()
        .zip(exp2x2.iter())
        .map(|(x, e)| (x - e).abs() < 1e-3)
        .collect();
    eprintln!("2x2 GEMM: got={:?} expected={:?} ok={:?}", c, exp2x2, ok2x2);

    // 1x8 GEMM (m=1 row-vector, k=4): A=[[1,2,3,4]] B=[[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    // C[0,j] = sum_k A[0,k]*B[k,j]
    // C[0,0]=1*5+2*9+3*13+4*17=5+18+39+68=130
    // C[0,1]=1*6+2*10+3*14+4*18=6+20+42+72=140
    // C[0,2]=1*7+2*11+3*15+4*19=7+22+45+76=150
    // C[0,3]=1*8+2*12+3*16+4*20=8+24+48+80=160
    // ... only 4 cols here (n=4). Let's do n=4, k=4.
    let a1: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let b1: Vec<f16> = vec![
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    ]
    .into_iter()
    .map(f16::from_f32)
    .collect();
    let c1 = ctx.dispatch_gemm(&a1, &b1, None, 1, 4, 4, 1.0, 0.0)?;
    let exp1x4 = [130.0f32, 140.0, 150.0, 160.0];
    let ok1x4: Vec<bool> = c1
        .iter()
        .zip(exp1x4.iter())
        .map(|(x, e)| (x - e).abs() < 1e-3)
        .collect();
    eprintln!(
        "1x4 GEMM: got={:?} expected={:?} ok={:?}",
        c1, exp1x4, ok1x4
    );

    let all_ok = ok2x2.iter().all(|&x| x) && ok1x4.iter().all(|&x| x);
    eprintln!(
        "VERDICT(small): {}",
        if all_ok {
            "GPU GEMM path computes correctly"
        } else {
            "GPU GEMM path is BROKEN (zeros/garbage)"
        }
    );

    // ── Output-head-size GEMM: m=1, k=896, n=151936 (Qwen2.5-0.5B) ────────
    // A = ones[1x896], B = identity-like [896 x 151936] where B[k, j] = 1.0 if
    // j == k else 0.0. Then C[0, j] = sum_k A[0,k]*B[k,j] = 1.0 for j<896
    // (the diagonal hit) and 0.0 for j>=896. This checks that the FULL 151936
    // output columns are actually written (a grid/launch bug at large n would
    // leave a tail of columns unwritten = 0).
    let (m, k, n) = (1usize, 896usize, 151936usize);
    let a_oh: Vec<f16> = vec![f16::from_f32(1.0); m * k];
    let b_oh: Vec<f16> = (0..k * n)
        .map(|i| {
            let kk = i / n;
            let j = i % n;
            f16::from_f32(if j == kk { 1.0 } else { 0.0 })
        })
        .collect();
    eprintln!(
        "[probe] allocating output-head GEMM: A={}KB B={}MB C={}KB",
        a_oh.len() * 2 / 1024,
        b_oh.len() * 2 / 1024 / 1024,
        m * n * 4 / 1024
    );
    let t0 = std::time::Instant::now();
    let c_oh = ctx.dispatch_gemm(&a_oh, &b_oh, None, m, n, k, 1.0, 0.0)?;
    let dt = t0.elapsed();
    // Expected: c_oh[j] == 1.0 for j in 0..896, 0.0 for j in 896..151936
    let mut bad = 0usize;
    let mut first_bad = None;
    for (j, &v) in c_oh.iter().enumerate() {
        let exp = if j < k { 1.0f32 } else { 0.0 };
        if (v - exp).abs() > 1e-3 {
            bad += 1;
            if first_bad.is_none() {
                first_bad = Some((j, v, exp));
            }
        }
    }
    let sum: f32 = c_oh.iter().sum();
    eprintln!(
        "[probe] output-head GEMM: n={} elapsed={:?} sum={:.1} (expect 896.0) bad_cols={} first_bad={:?}",
        n, dt, sum, bad, first_bad
    );
    let oh_ok = bad == 0;
    eprintln!(
        "VERDICT(output-head size): {}",
        if oh_ok {
            "OK — full 151936-col GEMM writes all columns correctly"
        } else {
            "BROKEN — some columns unwritten/wrong at n=151936"
        }
    );
    let _ = all_ok;
    Ok(())
}
