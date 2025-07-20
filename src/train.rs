use burn::data::dataloader::{DataLoaderBuilder};
use burn::nn::loss::{CrossEntropyLoss};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use rust_tokenizers::tokenizer::Gpt2Tokenizer;

use crate::data::{TokenizedDataset, TokenizedBatcher};
use crate::gpt2_model::GPTModel;
use crate::types::MyAutodiffBackend;
use crate::utils::generate_text;

pub fn train(
    mut model: GPTModel<MyAutodiffBackend>,
    tokenizer: &Gpt2Tokenizer,
    text_data: String,
    max_seq_len: usize,
    num_epochs: usize,
    batch_size: usize,
    seed: u64,
    train_ratio: f32,
) -> GPTModel<MyAutodiffBackend> {
    let split_idx = (text_data.len() as f32 * train_ratio) as usize;
    let train_data = &text_data[0..split_idx];
    let val_data = &text_data[split_idx..text_data.len()];

    let train_dataset = TokenizedDataset::new(&tokenizer, &train_data, max_seq_len);
    let val_dataset = TokenizedDataset::new(&tokenizer, &val_data, max_seq_len);

    let batcher: TokenizedBatcher<MyAutodiffBackend> = TokenizedBatcher::default();
    let train_dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(1)
        .build(train_dataset);

    let val_dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(1)
        .build(val_dataset);


    let optimizer_config = AdamWConfig::new()
        .with_weight_decay(0.1);
    let mut optimizer = optimizer_config.init::<MyAutodiffBackend, GPTModel<MyAutodiffBackend>>();

    for epoch in 1..=num_epochs {
        for (iteration, batch) in train_dataloader.iter().enumerate() {
            let output = model.forward(batch.inputs);
            let [batch_size, seq_len, vocab_size] = output.dims();
            let loss = &CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone().reshape([batch_size * seq_len, vocab_size]), batch.targets.clone());

            println!(
                "\t[Train - Epoch {} - Iteration {}] Loss {:.3}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optimizer.step(0.0004, model, grads);
        }

        for (iteration, batch) in val_dataloader.iter().enumerate() {
            let output = model.forward(batch.inputs);
            let [batch_size, seq_len, vocab_size] = output.dims();
            let loss = &CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone().reshape([batch_size * seq_len, vocab_size]), batch.targets.clone());

            println!(
                "\t[Valid - Epoch {} - Iteration {}] Loss {:.3}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );

        }
        
        println!("\tText sample: {}", generate_text(&model, &tokenizer, "This is a test, so please continue this sentence", 25, max_seq_len, 0.0, 0));
    }

    model
}
