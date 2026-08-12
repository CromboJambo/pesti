//! Dataset loading and batching.

use serde::{Deserialize, Serialize};

/// A batch of training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Batch {
    /// Input features [batch_size * seq_len] as f32 embeddings or token IDs converted to f32
    pub input_ids: Vec<f32>,
    /// Attention mask [batch_size, seq_len]
    pub attention_mask: Vec<u8>,
    /// Labels for loss computation (optional) - token IDs as u32
    pub labels: Option<Vec<u32>>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl Batch {
    /// Create a new batch.
    pub fn new(input_ids: Vec<f32>, attention_mask: Vec<u8>) -> Self {
        Self {
            input_ids,
            attention_mask,
            labels: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set labels for supervised learning.
    pub fn with_labels(mut self, labels: Vec<u32>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Get batch size (assuming uniform sequence length).
    pub fn batch_size(&self) -> usize {
        if self.input_ids.is_empty() {
            return 0;
        }
        let seq_len = self.input_ids.len();
        // Assuming all sequences are same length
        seq_len / self.attention_mask.len().max(1)
    }

    /// Get sequence length.
    pub fn seq_len(&self) -> usize {
        if self.attention_mask.is_empty() {
            return 0;
        }
        self.input_ids.len() / self.batch_size().max(1)
    }
}

/// Dataset trait for loading training data.
pub trait Dataset: Send + Sync {
    /// Get total number of samples.
    fn len(&self) -> usize;

    /// Check if dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a sample by index.
    fn get(&self, index: usize) -> Option<Batch>;

    /// Iterate over the dataset.
    fn iter(&self) -> DatasetIterator<'_, Self>
    where
        Self: Sized,
    {
        DatasetIterator::new(self, 0)
    }
}

/// Iterator over dataset samples.
pub struct DatasetIterator<'a, D: Dataset> {
    dataset: &'a D,
    index: usize,
}

impl<'a, D: Dataset> DatasetIterator<'a, D> {
    fn new(dataset: &'a D, index: usize) -> Self {
        Self { dataset, index }
    }
}

impl<'a, D: Dataset> Iterator for DatasetIterator<'a, D> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.dataset.len() {
            let item = self.dataset.get(self.index)?;
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

/// Dataset loader trait for loading datasets from various sources.
pub trait DatasetLoader: Send + Sync {
    /// Load dataset from source.
    fn load(
        &self,
        path: &str,
    ) -> Result<Box<dyn Dataset>, Box<dyn std::error::Error + Send + Sync>>;

    /// Create a dataset from in-memory data.
    fn from_memory(input_ids: Vec<Vec<f32>>, labels: Option<Vec<Vec<u32>>>) -> Box<dyn Dataset> {
        Box::new(InMemoryDataset { input_ids, labels })
    }
}

/// In-memory dataset implementation.
pub struct InMemoryDataset {
    pub input_ids: Vec<Vec<f32>>,
    pub labels: Option<Vec<Vec<u32>>>,
}

impl InMemoryDataset {
    /// Create a new in-memory dataset.
    pub fn new(input_ids: Vec<Vec<f32>>, labels: Option<Vec<Vec<u32>>>) -> Self {
        Self { input_ids, labels }
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.input_ids.len()
    }

    fn get(&self, index: usize) -> Option<Batch> {
        if index >= self.input_ids.len() {
            return None;
        }

        let input_ids = &self.input_ids[index];
        let attention_mask: Vec<u8> = vec![1; input_ids.len()];

        let batch = Batch::new(input_ids.clone(), attention_mask);

        if let Some(labels) = &self.labels
            && index < labels.len()
        {
            return Some(batch.with_labels(labels[index].clone()));
        }

        Some(batch)
    }
}
