use std::cmp::min;
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData, Int};
use burn::data::dataloader::batcher::Batcher;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use burn::data::dataset::Dataset;

use std::marker::PhantomData;

pub fn prepare_text_dataset(
    tokenizer: &Gpt2Tokenizer,
    raw_text: &str,
    max_seq_len: usize,
    stride: usize,
) -> Vec<(Vec<i64>, Vec<i64>)> {
    let mut dataset_pairs: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();

    // Encode the entire raw text into a single long sequence of token IDs.
    let encoded_text = tokenizer.encode(raw_text, None, raw_text.len(), &TruncationStrategy::LongestFirst, 0);
    let encoded_tokens = encoded_text.token_ids;

    let last_index = if encoded_tokens.len() >= max_seq_len { encoded_tokens.len() - max_seq_len } else { 0 };
    for i in (0..=last_index).step_by(stride) {
        let last_pos = min(encoded_tokens.len() - 2, i + max_seq_len - 2);

        let mut input_seq = Vec::new();
        // The input's content will be tokens from `i` up to `i + max_seq_len - 2`
        input_seq.extend_from_slice(&encoded_tokens[i..=last_pos]);

        let mut target_seq = Vec::new();
        // The target's content will be tokens from `i + 1` up to `i + max_seq_len - 1`
        target_seq.extend_from_slice(&encoded_tokens[(i + 1)..=last_pos+1]);

        dataset_pairs.push((input_seq, target_seq));
    }

    dataset_pairs
}

#[derive(Clone, Debug)]
pub struct TokenizedItem {
    pub inputs: Vec<i64>,
    pub targets: Vec<i64>
}

#[derive(Clone, Debug)]
pub struct TokenizedDataset {
    data: Vec<(Vec<i64>, Vec<i64>)>, // Now stores (input, target) pairs
}

#[derive(Clone, Debug)]
pub struct TokenizedBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone)]
pub struct TokenizedBatcher<B: Backend> {
    _phantom: PhantomData<B>,
}

impl<B: Backend> Default for TokenizedBatcher<B> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl TokenizedDataset {
    pub fn new(tokenizer: &Gpt2Tokenizer, raw_text: &str, max_seq_len: usize) -> Self {
        let data = prepare_text_dataset(tokenizer, raw_text, max_seq_len, max_seq_len);
        Self {
            data,
        }
    }
}

impl Dataset<TokenizedItem> for TokenizedDataset {
    fn get(&self, index: usize) -> Option<TokenizedItem> {
        let data = self.data.get(index).unwrap();
        Some(
            TokenizedItem {
            inputs: data.0.clone(),
            targets: data.1.clone()
            }
        )
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<B: Backend> Batcher<B, TokenizedItem, TokenizedBatch<B>> for TokenizedBatcher<B> { 
    fn batch(&self, items: Vec<TokenizedItem>, device: &B::Device) -> TokenizedBatch<B> {
        let l = items.get(0).unwrap().inputs.len();
        let inputs: Vec<_> = items
            .iter()
            .map(|item| TensorData::new(item.inputs.clone(), [1, l]))
            .map(|data| Tensor::<B, 2, Int>::from_data(data, device))
            .collect();

        let l = items.get(0).unwrap().targets.len();
        let targets: Vec<_> = items
            .iter()
            .map(|item| TensorData::new(item.targets.clone(), [l]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .collect();

        let inputs: Tensor<B, 2, Int> = Tensor::cat(inputs, 0);
        let targets: Tensor<B, 1, Int> = Tensor::cat(targets, 0);

        TokenizedBatch{
            inputs,
            targets
        }
    }
}
