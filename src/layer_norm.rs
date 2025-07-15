use burn::prelude::*;

#[derive(Config, Debug)]
pub struct LayerNormConfig {
    emb_dim: usize
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    eps: f32,
    emb_dim: usize,
    scale: Tensor<B, 1>,
    shift: Tensor<B, 1>
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        LayerNorm {
            eps: 10e-5,
            emb_dim: self.emb_dim,
            scale: Tensor::<B, 1>::ones(Shape::new([self.emb_dim]), device),
            shift: Tensor::<B, 1>::zeros(Shape::new([self.emb_dim]), device),
        }
    }
}

impl<B: Backend> LayerNorm<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let last_dim = input.dims().len() - 1;
        let mean = input.clone().mean_dim(last_dim);
        let var = input.clone().var(last_dim);
        let norm_x = (input.clone() - mean) / (var + self.eps).sqrt();

        let mut target_shape_dims = [1; D];
        target_shape_dims[D - 1] = self.emb_dim;
        let target_shape = Shape::new(target_shape_dims); 

        let scale_reshaped = self.scale.clone().reshape(target_shape.clone());
        let shift_reshaped = self.shift.clone().reshape(target_shape);

        (scale_reshaped * norm_x) + shift_reshaped
    }
}
