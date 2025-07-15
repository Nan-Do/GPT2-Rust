use burn::{
    nn::{
        Linear, LinearConfig, Dropout, DropoutConfig, 
    },
    prelude::*, tensor::activation::softmax, tensor::Bool
};

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    d_in: usize,
    d_out: usize,
    num_heads: usize,
    #[config(default=false)]
    add_bias: bool,
    #[config(default=0.0)]
    dropout: f64
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    d_out: usize,
    num_heads: usize,
    head_dim: usize,
    w_query: Linear<B>,
    w_key: Linear<B>,
    w_value: Linear<B>,
    out_proj: Linear<B>,
    dropout: Dropout,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        assert!(self.d_out % self.num_heads == 0, "d_out must be divisible by num_heads");

        MultiHeadAttention {
           d_out: self.d_out,
           num_heads: self.num_heads,
           head_dim: self.d_out / self.num_heads,
           w_query: LinearConfig::new(self.d_in, self.d_out).with_bias(self.add_bias).init(device),
           w_key: LinearConfig::new(self.d_in, self.d_out).with_bias(self.add_bias).init(device),
           w_value: LinearConfig::new(self.d_in, self.d_out).with_bias(self.add_bias).init(device),
           out_proj: LinearConfig::new(self.d_out, self.d_out).init(device),
           dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = input.device();
        let [batch_size, num_tokens, d_in] = input.dims();
        let d_in = d_in as f32;

        let keys = self.w_key.forward(input.clone());
        let values = self.w_value.forward(input.clone());
        let queries = self.w_query.forward(input); 

        let keys = keys.reshape([batch_size, num_tokens, self.num_heads, self.head_dim])
                                     .swap_dims(1, 2);
        let values = values.reshape([batch_size, num_tokens, self.num_heads, self.head_dim])
                                         .swap_dims(1, 2);
        let queries = queries.reshape([batch_size, num_tokens, self.num_heads, self.head_dim])
                                           .swap_dims(1, 2);

        let mask = Tensor::<B, 4, Bool>::tril_mask([1, 1, num_tokens, num_tokens], 0, &device);
        let attn_scores = queries.matmul(keys.transpose());
        let attn_scores = attn_scores.mask_fill(mask, f32::NEG_INFINITY);

        let last_dim = attn_scores.dims().len() - 1;
        let att_weights = softmax(attn_scores / d_in.sqrt(), last_dim);
        let att_weights = self.dropout.forward(att_weights);

        let context_vec = att_weights.matmul(values).swap_dims(1, 2);
        let context_vec = context_vec.reshape([batch_size, num_tokens, self.d_out]);

        self.out_proj.forward(context_vec)
    }
}
