#![recursion_limit = "256"]

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use argh::FromArgs;
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use rust_tokenizers::tokenizer::Gpt2Tokenizer;

use crate::gpt2_model::{GPTModelConfig, GPTModel};
use crate::train::train;
use crate::types::{MyAutodiffBackend, TrainingOptions};
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
    /// select the mode to run the tool (generate, train)
    #[argh(positional)]
    command: String,

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

    /// train ratio for training (0.9 by default).
    #[argh(option, default = "0.9")]
    train_ratio: f32,

    /// file path for the model weights (weights by default)
    #[argh(option)]
    weights: String,

}

fn train_command(args: GptOptions) {
    println!("--- Tokenizer Summary ---");
    println!("\tUsing vocabulary file: {}", args.vocab_file);
    println!("\tUsing merges file: {}", args.merges_file);

    println!("--- Model Summary ---"); 
    println!("\tVocabulary size: 50257");
    println!("\tContext length: {}", args.context_length);
    println!("\tEmbeddings dimension: {}", args.emb_dim);
    println!("\tNumber of layers in the transformer block : {}", args.num_layers);
    println!("\tNumber of heads in the multi head attention block : {}", args.num_heads);

    println!("--- Loading Training Text ---");
    println!("\tUsing file {} ", args.training_file_name);
    let raw_text: String = fs::read_to_string(args.training_file_name).expect("Training file must exist");
    let train_size = (raw_text.len() as f32 * args.train_ratio) as usize;
    println!("\tTraining size: {train_size}");
    println!("\tValidation size: {}", raw_text.len() - train_size);

    println!("--- Training Summary ---");
    println!("\tBatch Size: {}", args.batch_size);
    println!("\tNumber of epochs to train: {}", args.epochs);

    println!("--- Initializing Device ---");
    println!("\tUsing seed {}", args.seed);
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

    if !args.weights.is_empty() {
        let path = PathBuf::from(args.weights.clone() + ".mpk.gz");
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        if path.exists(){
            let basename_path = PathBuf::from(&args.weights);
            println!("\tLoading weights from {:?}", path);
            gpt2_model = gpt2_model
                            .load_file(basename_path, &recorder, device)
                            .expect("Should be able to load the model weights from the provided file")
        }
    }

    println!("--- Training Model ---");
    let training_options = TrainingOptions {
        max_seq_len: args.context_length,
        num_epochs: args.epochs,
        batch_size: args.batch_size,
        seed: args.seed,
        train_ratio: args.train_ratio,
    };
    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    gpt2_model = train(gpt2_model, &tokenizer, raw_text, training_options);
    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("--- Training Summary ---");
    println!("\tTraining took {:?}", end-start);

    println!("--- Generating Text After Training ---");
    println!("\t{}", generate_text(&gpt2_model, &tokenizer, &args.text_to_continue, 25, args.context_length, 0.8, 20));

    if !args.weights.is_empty() {
        println!("--- Saving Model Weights ---");
        println!("\tSaving weights to {:?}", path);
        let path = PathBuf::from(args.weights.clone() + ".mpk.gz");
        let basename_path = PathBuf::from(&args.weights);
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        gpt2_model
            .save_file(basename_path, &recorder)
            .expect("Should be able to save the model");
        }
}


fn generate_command(args: GptOptions) {
    println!("ToDo");
}

fn main() {
    let args: GptOptions = argh::from_env();

    if args.command == "train" {
        train_command(args);
    } else{
        generate_command(args);
    }

}
