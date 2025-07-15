use burn::{nn::Dropout, nn::DropoutConfig, prelude::*};

use crate::{
    feed_forward::FeedForward, feed_forward::FeedForwardConfig,
    multi_head_attention::MultiHeadAttention, multi_head_attention::MultiHeadAttentionConfig, 
    layer_norm::LayerNorm, layer_norm::LayerNormConfig
};

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    d_in: usize,
    d_out: usize,
    context_length: usize,
    emb_dim: usize,
    num_heads: usize,
    dropout: f64,
    qkv_bias: bool,
}


#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    att: MultiHeadAttention<B>,
    ff: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    drop_shortcut: Dropout
}


impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock{
            att: MultiHeadAttentionConfig::new(self.d_in, self.d_out, self.num_heads)
                .with_add_bias(self.qkv_bias)
                .with_dropout(self.dropout)
                .init::<B>(device),
            ff: FeedForwardConfig::new(self.emb_dim).init::<B>(device),
            norm1: LayerNormConfig::new(self.emb_dim).init::<B>(device),
            norm2: LayerNormConfig::new(self.emb_dim).init::<B>(device),
            drop_shortcut: DropoutConfig::new(self.dropout).init()
        }
    }
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let shortcut = input.clone();
        let input = self.norm1.forward(input);
        let input = self.att.forward(input);
        let input = self.drop_shortcut.forward(input);
        let input = input + shortcut;

        let shortcut = input.clone();
        let input = self.norm2.forward(input);
        let input = self.ff.forward(input);
        let input = self.drop_shortcut.forward(input);
        
        input + shortcut
    }
}
