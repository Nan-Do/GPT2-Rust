#![recursion_limit = "256"]

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use argh::FromArgs;
use burn::prelude::Backend;
use rust_tokenizers::tokenizer::Gpt2Tokenizer;

use crate::gpt2_model::{GPTModelConfig, GPTModel};
use crate::train::train;
use crate::types::MyAutodiffBackend;
use crate::utils::generate_text;

mod data;
mod feed_forward;
mod gpt2_model;
mod layer_norm;
mod multi_head_attention;
mod train;
mod transformer_block;
mod trf_blocks_sequence;
mod types;
mod utils;


#[derive(FromArgs)]
/// Options to train the model.
struct GptOptions {
    /// context length for the GPT Model (1024 by default).
    #[argh(option, default = "1024")]
    context_length: usize,

    /// embedding dimension for the GPT Model (768 by default).
    #[argh(option, default = "768")]
    emb_dim: usize,

    /// number of layers in the Transformer Block (12 by default).
    #[argh(option, default = "12")]
    num_layers: usize,

    /// number of heads for the Multi Head Attention Block (12 by default).
    #[argh(option, default = "12")]
    num_heads: usize,

    /// number of epochs to train (10 by default).
    #[argh(option, default = "10")]
    epochs: usize,

    /// batch size (2 by default).
    #[argh(option, default = "2")]
    batch_size: usize,

    /// text file that will be used to train the model (the-verdict.txt by default)
    #[argh(option, default = "String::from(\"the-verdict.txt\")")]
    training_file_name: String,

    /// vocab file used with the BPE Tokenizer (vocab.json by default)
    #[argh(option, default = "String::from(\"vocab.json\")")]
    vocab_file: String,

    /// merges file used with the BPE Tokenizer (merges.txt by default)
    #[argh(option, default = "String::from(\"merges.txt\")")]
    merges_file: String,

    /// text to be continued by the model after training (Hello world! by default)
    #[argh(option, default = "String::from(\"Hello World!\")")]
    text_to_continue: String,

    /// random seed (123 by default).
    #[argh(option, default = "123")]
    seed: u64,
}

fn main() {
    let args: GptOptions = argh::from_env();

    println!("--- Tokenizer Summary ---");
    println!("Using vocabulary file: {}", args.vocab_file);
    println!("Using merges file: {}", args.merges_file);
    println!("--- Model Summary ---"); 
    println!("Vocabulary size: 50257");
    println!("Context length: {}", args.context_length);
    println!("Embeddings dimension: {}", args.emb_dim);
    println!("Number of layers in the transformer block : {}", args.num_layers);
    println!("Number of heads in the multi head attention block : {}", args.num_heads);

    println!("--- Training Summary ---");
    println!("Batch Size: {}", args.batch_size);
    println!("Number of epochs to train: {}", args.epochs);

    println!("--- Initializing Device ---");
    println!("Using seed {}", args.seed);
    MyAutodiffBackend::seed(args.seed);
    let device = &<MyAutodiffBackend as Backend>::Device::default();

    println!("--- Initializing Tokenizer ---");
    let tokenizer = Gpt2Tokenizer::from_file(
        args.vocab_file,
        args.merges_file,
        false).expect("Error loading the tokenizer");

    println!("--- Initializing Model ---");
    let mut gpt2_model: GPTModel<MyAutodiffBackend> = GPTModelConfig::new(
        50257,
        args.context_length,
        args.emb_dim,
        args.num_heads,
        args.num_layers,
        0.1,
        false).init(device);
    println!("--- Reading the given text file {} ---", args.training_file_name);
    let raw_text: String = fs::read_to_string(args.training_file_name).expect("Training file must exist");

    println!("--- Training Model ---");
    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    gpt2_model = train(gpt2_model, &tokenizer, raw_text,  args.context_length, args.epochs, args.batch_size, args.seed);
    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("Training took {:?}s", end-start);

    println!("--- Generating Text After Training ---");
    println!("{}", generate_text(&gpt2_model, &tokenizer, &args.text_to_continue, 25, args.context_length));
}
