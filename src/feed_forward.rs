use burn::{
    nn::{
        Linear, LinearConfig, 
    },
    tensor::activation::gelu,
    prelude::*
};


#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    emb_dim: usize,
}


#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>
}


impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward{
            layer1: LinearConfig::new(self.emb_dim, self.emb_dim * 4).init(device),
            layer2: LinearConfig::new(self.emb_dim * 4, self.emb_dim).init(device),
        }
    }
}


impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let input = self.layer1.forward(input);
        let input = gelu(input);
        self.layer2.forward(input)
    }
}

