use std::cmp::max;
use burn::tensor::{Tensor, TensorData, Int};
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};

use crate::gpt2_model::GPTModel;
use crate::types::{MyBackend, MyAutodiffBackend};

pub fn _decode_text_from_tensor(tokenizer: &Gpt2Tokenizer, tensor: Tensor<MyBackend, 2, Int>) -> String{
    let data: Vec<i64> = tensor.to_data().to_vec().unwrap().into_iter().map(|x: i32| x as i64).collect(); 
    tokenizer.decode(&data, true, true)
}

pub fn generate_text(
    // model: &GPTModel<MyBackend>,
    model: &GPTModel<MyAutodiffBackend>,
    tokenizer: &Gpt2Tokenizer,
    input_text: &str,
    steps: usize,
    max_seq_len: usize,
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
        let input_tensor: Tensor<MyAutodiffBackend, 2, Int> =
            TensorData::new(encoded_tokens.clone(), [1, encoded_tokens.len()]).into();

        let output_logits = model.forward(input_tensor);

        let next_token_tensor = output_logits.argmax(2).squeeze::<2>(2);

        let last_dim_size = next_token_tensor.dims()[1];
        let index_tensor: Tensor<MyAutodiffBackend, 1, Int> = TensorData::from([(last_dim_size - 1)]).into();
        let last_token: i32 = next_token_tensor
            .select(1, index_tensor)
            .squeeze::<1>(1)
            .into_scalar();
        encoded_tokens.push(last_token as i64);

        if encoded_tokens.len() > max_seq_len {
            encoded_tokens.remove(0);
        }
    }

    tokenizer.decode(&encoded_tokens, true, true)
}
