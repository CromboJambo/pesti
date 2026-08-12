//! Low-level matrix operations for adapter computations.

/// Matrix A in LoRA: projects from input space to latent space.
/// Shape: [rank x in_features]
#[derive(Debug, Clone)]
pub struct MatA {
    data: Vec<f32>,
    rows: usize, // rank
    cols: usize, // in_features
}

impl MatA {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert!(
            data.len() == rows * cols,
            "MatA data length {} != rows * cols {}",
            data.len(),
            rows * cols
        );
        Self { data, rows, cols }
    }

    pub fn empty() -> Self {
        Self {
            data: vec![],
            rows: 0,
            cols: 0,
        }
    }

    /// Compute Ax where A is [rows x cols] and x is [batch_size, cols].
    /// Returns [batch_size, rows].
    pub fn matmul_transpose(
        &self,
        x: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, super::AdapterError> {
        if self.rows == 0 || self.cols == 0 {
            return Err(super::AdapterError::NotInitialized);
        }

        if self.cols != x.len() / batch_size {
            return Err(super::AdapterError::DimensionMismatch {
                expected: self.cols,
                actual: x.len() / batch_size,
            });
        }

        let mut output = vec![0.0f32; batch_size * self.rows];

        // For each batch element
        for b in 0..batch_size {
            let x_start = b * self.cols;
            let out_start = b * self.rows;

            // For each row of A
            for r in 0..self.rows {
                let a_row_start = r * self.cols;
                let mut sum = 0.0f32;

                // Dot product: sum_i(A[r,i] * x[b,i])
                for c in 0..self.cols {
                    sum += self.data[a_row_start + c] * x[x_start + c];
                }

                output[out_start + r] = sum;
            }
        }

        Ok(output)
    }

    pub fn zero(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0);
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Matrix B in LoRA: projects from latent space to output space.
/// Shape: [out_features x rank]
#[derive(Debug, Clone)]
pub struct MatB {
    data: Vec<f32>,
    rows: usize, // out_features
    cols: usize, // rank
}

impl MatB {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert!(
            data.len() == rows * cols,
            "MatB data length {} != rows * cols {}",
            data.len(),
            rows * cols
        );
        Self { data, rows, cols }
    }

    pub fn empty() -> Self {
        Self {
            data: vec![],
            rows: 0,
            cols: 0,
        }
    }

    /// Compute Bx where B is [rows x cols] and x is [batch_size, cols].
    /// Returns [batch_size, rows].
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        if self.rows == 0 || self.cols == 0 {
            return vec![];
        }

        let mut output = vec![0.0f32; batch_size * self.rows];

        // For each batch element
        for b in 0..batch_size {
            let x_start = b * self.cols;
            let out_start = b * self.rows;

            // For each row of B
            for r in 0..self.rows {
                let b_row_start = r * self.cols;
                let mut sum = 0.0f32;

                // Dot product: sum_i(B[r,i] * x[b,i])
                for c in 0..self.cols {
                    sum += self.data[b_row_start + c] * x[x_start + c];
                }

                output[out_start + r] = sum;
            }
        }

        output
    }

    /// Compute B @ A (matrix multiply).
    /// Returns [out_features x in_features] matrix.
    pub fn matmul(&self, a: &MatA) -> Vec<f32> {
        let out_features = self.rows;
        let in_features = a.cols;
        let rank = self.cols; // == a.rows

        let mut result = vec![0.0f32; out_features * in_features];

        for out_idx in 0..out_features {
            for (col, _in_idx) in (0..in_features).enumerate() {
                let mut sum = 0.0f32;
                for k in 0..rank {
                    let b_val = self.data[out_idx * rank + k];
                    let a_val = a.data[k * in_features + col];
                    sum += b_val * a_val;
                }
                result[out_idx * in_features + col] = sum;
            }
        }

        result
    }

    pub fn zero(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0);
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}
