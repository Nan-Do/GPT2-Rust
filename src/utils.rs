use std::cmp::max;
use burn::tensor::cast::ToElement;
use rand::Rng;
use burn::tensor::{Tensor, TensorData, Int};
use burn::tensor::activation::softmax;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};

use crate::gpt2_model::GPTModel;

pub fn generate_text<Backend: burn::prelude::Backend>(
    model: &GPTModel<Backend>,
    tokenizer: &Gpt2Tokenizer,
    input_text: &str,
    steps: usize,
    max_seq_len: usize,
    temperature: f32,
    top_k: usize,
) -> String {
    // Encode the input text
    let mut encoded_tokens = tokenizer
        .encode(
            input_text,
            None,
            input_text.len(),
            &TruncationStrategy::LongestFirst,
            0,
        )
        .token_ids;

    if encoded_tokens.len() > max_seq_len {
        let start_index = max(encoded_tokens.len() - max_seq_len, 0);
        encoded_tokens = encoded_tokens[start_index..].to_vec();
    }

    for _ in 0..steps {
        let input_tensor: Tensor<Backend, 2, Int> =
            TensorData::new(encoded_tokens.clone(), [1, encoded_tokens.len()]).into();

        let outputs = model.forward(input_tensor);
        let [batch_size, tokens, emb_size] = outputs.dims();

        let logits = outputs.slice([0..batch_size, tokens-1..tokens, 0..emb_size]);
        let mut logits = logits.squeeze_dims::<1>(&[0, 1]);

        if top_k > 0 {
            let top_logits = logits.clone().topk(top_k, 0);
            let min_val = top_logits.slice(top_k-1..top_k).into_scalar();
            let mask = logits.clone().lower_elem(min_val);
            logits = logits.mask_fill(mask, f32::NEG_INFINITY);
        }

        if temperature > 0.0 {
            let mut rng = rand::rng();
            logits = logits / temperature;
            let probs = softmax(logits, 0);

            let mut cum_sum: Vec<f32> = vec![];
            let mut partial_sum: f32 = 0.0;
            for val in probs.to_data().iter::<f32>() {
                partial_sum += val;
                cum_sum.push(partial_sum);
            }
            let random_val: f32 = rng.random();
            let mut next_token: i64 = 0;
            for (idx, val) in cum_sum.iter().enumerate() {
                if random_val < *val {
                    next_token = idx as i64;
                    break;
                }
            }
            #[cfg(debug_assertions)] 
            {
                let max_token = probs.argmax(0).into_scalar();
                println!("Token: {next_token}; Argmax Token: {max_token}"); 
            }
            encoded_tokens.push(next_token);
        } else {
            let next_token = logits.argmax(0).into_scalar();
            encoded_tokens.push(next_token.to_i64());
        }

        if encoded_tokens.len() > max_seq_len {
            encoded_tokens.remove(0);
        }
    }

    tokenizer.decode(&encoded_tokens, true, true)
}
