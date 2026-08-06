//! Isolate: does gemm crash with heap-allocated f32 buffer of same size?
use std::io::Write;

fn main() {
    let m = 1usize;
    let k = 500usize;
    let n = 500usize;

    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
    let w: Vec<f32> = vec![0.0f32; n * k];
    let mut output = vec![0.0f32; m * n];

    eprintln!("gemm: m={}, n={}, k={}, w.len={}, x.len={}, output.len={}", m, n, k, w.len(), x.len(), output.len());
    std::io::stderr().flush().unwrap();

    unsafe {
        gemm::gemm(
            m, n, k,
            output.as_mut_ptr(),
            1_isize,        // dst_cs
            n as isize,     // dst_rs
            false,          // read_dst
            x.as_ptr(),     // lhs
            k as isize,     // lhs_cs
            1_isize,        // lhs_rs
            w.as_ptr(),     // rhs
            k as isize,     // rhs_cs
            1_isize,        // rhs_rs
            1.0f32,
            0.0f32,
            false, false, false,
            gemm::Parallelism::Rayon(0),
        );
    }

    eprintln!("Done! output[0]={}", output[0]);
    std::io::stderr().flush().unwrap();
    println!("All zero: {}", output.iter().all(|&v| v == 0.0));
}
