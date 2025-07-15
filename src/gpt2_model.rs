use burn::nn::{
    Dropout, DropoutConfig, 
    Linear, LinearConfig, 
    Embedding, EmbeddingConfig,
};
use burn::tensor::{Tensor, Int};
use burn::prelude::*;

use crate::{
    layer_norm::LayerNorm, layer_norm::LayerNormConfig,
    trf_blocks_sequence::TransformersBlocksSequence, 
    trf_blocks_sequence::TransformersBlocksSequenceConfig
};

#[derive(Config, Debug)]
pub struct GPTModelConfig {
    vocab_size: usize,
    context_length: usize,
    emb_dim: usize,
    num_heads: usize,
    n_layers: usize,
    drop_rate: f64,
    qkv_bias: bool,
}


#[derive(Module, Debug)]
pub struct GPTModel<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    drop_emb: Dropout,
    trf_blocks: TransformersBlocksSequence<B>,
    final_norm: LayerNorm<B>,
    out_head: Linear<B>,
}

impl GPTModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPTModel<B> {
        GPTModel{
            tok_emb: EmbeddingConfig::new(self.vocab_size, self.emb_dim).init::<B>(device),
            pos_emb: EmbeddingConfig::new(self.context_length, self.emb_dim).init::<B>(device),
            drop_emb: DropoutConfig::new(self.drop_rate).init(),
            trf_blocks: TransformersBlocksSequenceConfig::new(self.emb_dim, self.context_length, self.num_heads, self.drop_rate, self.qkv_bias, self.n_layers).init::<B>(device),
            final_norm: LayerNormConfig::new(self.emb_dim).init::<B>(device),
            out_head: LinearConfig::new(self.emb_dim, self.vocab_size).init::<B>(device)
        }
    }
}


impl<B: Backend> GPTModel<B> {
    pub fn forward(&self, in_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = in_idx.device();
        let [_batch_size, seq_len] = in_idx.dims();
        let seq_len = seq_len as i64;
        let tok_embeds = self.tok_emb.forward(in_idx);
        let tensor = Tensor::<B, 1, Int>::arange(0..seq_len, &device);
        let pos_embeds = self.pos_emb.forward(tensor.unsqueeze());
        
        let x = tok_embeds + pos_embeds;
        let x = self.drop_emb.forward(x);
        let x = self.trf_blocks.forward(x);
        let x = self.final_norm.forward(x);

        self.out_head.forward(x)
    }
}
