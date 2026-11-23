//! Behavioral invariants for the split-half RoPE implementation.
//!
//! These tests deliberately exercise the public API without requiring CUDA.
use pesti_runner::transformer::rope::RopeConfig;

fn squared_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum()
}

#[test]
fn position_zero_is_identity() {
    let config = RopeConfig::new(4, 10_000.0, 128);
    let original = vec![1.0, -2.0, 3.5, 4.0, -0.5, 2.25, 1.5, -3.0];
    let mut actual = original.clone();

    config.apply_single(&mut actual, 2, 1, 0);

    for (index, (actual, expected)) in actual.iter().zip(&original).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "position zero changed element {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn rotation_preserves_each_pair_norm() {
    let config = RopeConfig::new(4, 10_000.0, 128);
    let mut actual = vec![1.0, -2.0, 3.5, 4.0, -0.5, 2.25, 1.5, -3.0];
    let original = actual.clone();

    config.apply_single(&mut actual, 2, 1, 37);

    // This implementation stores the two coordinates of each rotary pair
    // in the first and second halves of each head vector.
    for head in 0..2 {
        for i in 0..2 {
            let first = head * 4 + i;
            let second = first + 2;
            let before = squared_norm(&[original[first], original[second]]);
            let after = squared_norm(&[actual[first], actual[second]]);

            assert!(
                (after - before).abs() < 1e-5,
                "rotation changed norm for head {head}, pair {i}: {before} -> {after}"
            );
        }
    }
}

#[test]
fn start_position_uses_absolute_token_positions() {
    let config = RopeConfig::new(4, 10_000.0, 128);
    let mut sequence = vec![
        1.0, 2.0, 3.0, 4.0, // position 0
        -1.0, 0.5, 2.5, -3.0, // position 1
    ];
    config.apply_single(&mut sequence, 1, 2, 7);

    let mut token_at_absolute_position_eight = vec![-1.0, 0.5, 2.5, -3.0];
    config.apply_single(&mut token_at_absolute_position_eight, 1, 1, 8);

    for (index, (from_sequence, standalone)) in sequence[4..]
        .iter()
        .zip(&token_at_absolute_position_eight)
        .enumerate()
    {
        assert!(
            (from_sequence - standalone).abs() < 1e-6,
            "offset position mismatch at element {index}: {from_sequence} != {standalone}"
        );
    }
}
