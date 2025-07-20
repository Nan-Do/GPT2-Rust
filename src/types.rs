use burn::backend::{Wgpu, Autodiff};

// Type alias for the backend to use.
pub type MyBackend = Wgpu;
// Type alias for the audtodiff backend
pub type MyAutodiffBackend = Autodiff<MyBackend>;

pub struct TrainingOptions {
    pub max_seq_len: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub train_ratio: f32,
}
