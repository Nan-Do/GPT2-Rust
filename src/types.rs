use burn::backend::{Wgpu, Autodiff};

// Type alias for the backend to use.
pub type MyBackend = Wgpu;
// Type alias for the audtodiff backend
pub type MyAutodiffBackend = Autodiff<MyBackend>;
