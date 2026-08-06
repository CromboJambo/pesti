use pesti_runner::transformer::linear::Linear;

#[test]
fn test_linear_gemm_basic() {
    // Simple 2x3 matrix multiply
    let weight = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ];
    let bias = Some(vec![0.1, 0.2]);
    
    let linear = Linear::new(weight, bias, 3, 2);
    
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let output = linear.forward(&x, 2);
    
    // Expected: C[0,0]=14.1, C[0,1]=32.2, C[1,0]=32.1, C[1,1]=77.2
    assert!((output[0] - 14.1).abs() < 1e-5);
    assert!((output[1] - 32.2).abs() < 1e-5);
    assert!((output[2] - 32.1).abs() < 1e-5);
    assert!((output[3] - 77.2).abs() < 1e-5);
}
