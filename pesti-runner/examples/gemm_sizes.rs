use std::io::Write;

struct W {
    weight: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl W {
    fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let m = batch_size;
        let n = self.out_features;
        let k = self.in_features;
        let mut output = vec![0.0f32; m * n];
        if m == 0 || n == 0 || k == 0 { return output; }
        unsafe {
            gemm::gemm(
                m, n, k,
                output.as_mut_ptr(),
                1_isize,
                n as isize,
                false,
                x.as_ptr(),
                k as isize,
                1_isize,
                self.weight.as_ptr(),
                k as isize,
                1_isize,
                1.0f32,
                0.0f32,
                false, false, false,
                gemm::Parallelism::Rayon(0),
            );
        }
        output
    }
}

fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    for &s in &sizes {
        let w = vec![0.0f32; s * s];
        let model = W { weight: w, in_features: s, out_features: s };
        let x: Vec<f32> = (0..s).map(|i| i as f32 * 0.01).collect();
        let result = model.forward(&x, 1);
        println!("1x{0}x{0}: OK (len={1})", s, result.len());
    }
}
