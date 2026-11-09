//! Pooling strategies for reducing sequence of hidden states to single embedding vector.

/// Strategy for pooling token embeddings into a single vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Mean pooling across all token embeddings (most common).
    Mean,
    /// Use the first token embedding ([CLS] style).
    First,
    /// Use the last token embedding.
    Last,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        PoolingStrategy::Mean
    }
}

/// Apply pooling strategy to sequence of hidden states.
///
/// # Arguments
/// * `hidden_states` - Sequence of vectors (seq_len x embed_dim)
/// * `strategy` - Pooling strategy to apply
///
/// # Returns
/// Single pooled embedding vector (embed_dim)
pub fn pool(hidden_states: &[Vec<f32>], strategy: PoolingStrategy) -> Vec<f32> {
    if hidden_states.is_empty() {
        return vec![];
    }

    match strategy {
        PoolingStrategy::First => hidden_states[0].clone(),
        PoolingStrategy::Last => hidden_states.last().unwrap().clone(),
        PoolingStrategy::Mean => mean_pool(hidden_states),
    }
}

/// Mean pooling across all token embeddings.
fn mean_pool(states: &[Vec<f32>]) -> Vec<f32> {
    let embed_dim = states[0].len();
    let mut result = vec![0.0f32; embed_dim];

    for state in states {
        for (r, s) in result.iter_mut().zip(state.iter()) {
            *r += s;
        }
    }

    let n = states.len() as f32;
    for r in result.iter_mut() {
        *r /= n;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pooling() {
        let states = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let pooled = pool(&states, PoolingStrategy::Mean);
        assert_eq!(pooled, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_first_pooling() {
        let states = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let pooled = pool(&states, PoolingStrategy::First);
        assert_eq!(pooled, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_last_pooling() {
        let states = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let pooled = pool(&states, PoolingStrategy::Last);
        assert_eq!(pooled, vec![4.0, 5.0, 6.0]);
    }
}
