use burn::prelude::*;

use crate::transformer_block::{
    TransformerBlock, TransformerBlockConfig
};

#[derive(Config, Debug)]
pub struct TransformersBlocksSequenceConfig {
    emb_dim: usize,
    context_length: usize,
    num_heads: usize,
    drop_rate: f64,
    qkv_bias: bool,
    num_layers: usize
}


#[derive(Module, Debug)]
pub struct TransformersBlocksSequence<B: Backend> {
    trf_blocks: Vec<TransformerBlock<B>>,
}


impl TransformersBlocksSequenceConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformersBlocksSequence<B> {
        let mut blocks: Vec<TransformerBlock<B>> = Vec::new();

        for _ in 0..self.num_layers {
            blocks.push(TransformerBlockConfig::new(
                self.emb_dim, 
                self.emb_dim, 
                self.context_length, 
                self.emb_dim,  
                self.num_heads,  
                self.drop_rate, 
                self.qkv_bias).init(device));
        }

        TransformersBlocksSequence {
            trf_blocks: blocks
        }
    }
}

impl<B: Backend> TransformersBlocksSequence<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut data = input;
        for i in 0..self.trf_blocks.len() {
            data = self.trf_blocks[i].forward(data);
        }
        data
    }
}
