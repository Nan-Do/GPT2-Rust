# GPT-2 from Scratch in Rust 🦀

This repository contains a from-scratch implementation of a GPT-2 style Large Language Model (LLM) written entirely in **Rust**. It utilizes the powerful and flexible [**Burn**](https://burn.dev/) deep learning framework.

The primary goal of this project is to serve as an educational resource, demonstrating the core components of a modern LLM in a performant, type-safe language. It is a Rust-based counterpart to the excellent Python/PyTorch implementation by Sebastian Raschka.

[![Rust](https://img.shields.io/badge/rust-1.88.0-orange.svg)](https://www.rust-lang.org/)

***

## ✨ Features

* ✅ **GPT-2 Architecture:** A clean implementation of the decoder-only transformer architecture from the ground up.
* 🧠 **Text Generation:** Generate new text from a given prompt.
* 🌡️ **Temperature Sampling:** Control the creativity and randomness of the output.
* 🔝 **Top-K Sampling:** Limit token selection to the *k* most likely next tokens to improve coherence.
* 🏋️ **Model Training:** Includes functionality to train the model on a custom text corpus.

***

## 📚 Inspiration & Credits

This work is heavily inspired by and serves as a Rust-based companion to the following amazing resources:

* **LLMs from Scratch by Sebastian Raschka:** The original Python/PyTorch guide that this project is based on.
    * GitHub Repository: [**rasbt/LLMs-from-scratch**](https://github.com/rasbt/LLMs-from-scratch)
* **The Burn Framework:** A modern, flexible, and efficient deep learning framework for Rust.
    * Official Website: [**burn.dev**](https://burn.dev/)

***

## ⚙️ Installation

To get started, you'll need the Rust tool-chain installed on your system. You can install it via [rustup.rs](https://rustup.rs/).

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Nan-Do/GPT2-Rust
    cd GPT2-Rust
    ```

2.  **Build the project in release mode:**
    Building in release mode is highly recommended for performance.
    ```sh
    cargo build --release
    ```

***

## 🚀 Usage

The tool provides two main subcommands: `generate` for creating text and `train` for training the model.

### Generating Text

You can generate text from a prompt using a pre-trained model. The generation process can be customized with several options.

**Command:**

```
cargo run --release -- generate [OPTIONS]

Options:
  --vocab-file      vocab file used with the BPE Tokenizer (vocab.json by default)
  --merges-file     merges file used with the BPE Tokenizer (merges.txt by default)
  --text            text to be continued by the model (Hello world! by default)
  --seed            random seed (123 by default).
  --weights         file path for the model weights (weights by default)
  --num-tokens      number of tokens to generate (25 by default).
  --top-k           top k tokens to consider when generating text (disabled by
                    default from 0 to 50257).
  --temperature     temperature used when generating text (disabled by default
                    from 0.0 to 1.0).
```

**Example:**
```
cargo run --release -- generate\
  --text "The last time I saw"\
  --num-tokens 100\
  --temperature 0.7\
  --top-k 150\
  --weights weights
```


### Training the Model

You can train the model from scratch on your own dataset (e.g., a single large .txt file).

**Command:**

```
cargo run --release -- train [OPTIONS]

Options:
  --context-length  context length for the GPT Model (1024 by default).
  --emb-dim         embedding dimension for the GPT Model (768 by default).
  --num-layers      number of layers in the Transformer Block (12 by default).
  --num-heads       number of heads for the Multi Head Attention Block (12 by default).
  --epochs          number of epochs to train (10 by default).
  --batch-size      batch size (2 by default).
  --training-file-name
                    text file that will be used to train the model
                    (the-verdict.txt by default)
  --vocab-file      vocab file used with the BPE Tokenizer (vocab.json by default)
  --merges-file     merges file used with the BPE Tokenizer (merges.txt by default)
  --seed            random seed (123 by default).
  --train-ratio     train ratio for training (0.9 by default).
  --weights         file path for the model weights (weights by default)
```


**Example:**
```
cargo run --release -- train\
  --context-length 256\
  --epochs 5\
  --batch-size 4\
  --weights weights
```

