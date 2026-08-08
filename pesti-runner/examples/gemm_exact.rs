use std::io::Write;

fn main() {
    // Test m=1 only (inference case, no auto-transpose)
    let m = 1usize;
    let n = 896usize;
    let k = 896usize;

    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.001).collect();
    let weight: Vec<f32> = (0..n * k).map(|i| (i as f32 + 1.0) * 0.001).collect();
    let mut output = vec![0.0f32; m * n];

    eprintln!(
        "Calling gemm with m={}, n={}, k={}, x.len={}, w.len={}, out.len={}",
        m,
        n,
        k,
        x.len(),
        weight.len(),
        output.len()
    );
    std::io::stderr().flush().unwrap();

    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            output.as_mut_ptr(),
            1_isize,    // dst_cs
            n as isize, // dst_rs
            false,      // read_dst
            x.as_ptr(),
            k as isize, // lhs_cs
            1_isize,    // lhs_rs
            weight.as_ptr(),
            k as isize, // rhs_cs
            1_isize,    // rhs_rs
            1.0f32,
            0.0f32,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }

    let sum: f32 = output.iter().sum();
    let max: f32 = output.iter().cloned().fold(0.0f32, f32::max);
    println!(
        "output: sum={:.4}, max={:.4}, output[0]={:.6}",
        sum, max, output[0]
    );
}
