# microGPT2

A decoder-only transformer language model built entirely from scratch in PyTorch — including a hand-written attention mechanism, positional encoding, and custom optimizer implementations (Adam, AdamW, RMSprop, AdaGrad, SGD). No `torch.nn.TransformerDecoder` or `torch.optim` — the core building blocks are implemented manually to understand exactly how each piece works.

Trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset to generate short, simple English stories.

## Features

- **Custom decoder-only transformer** — multi-head self-attention with causal masking, positional encoding, feedforward blocks, and layer normalization, all implemented from scratch
- **Custom tokenizer training** — Byte-level BPE tokenizer trained directly on the target corpus
- **Memory-efficient data pipeline** — tokenizes large corpora in chunks and stores tokens as a disk-backed `numpy` memmap (`uint16`) instead of loading the entire dataset into RAM, so training scales to full-size corpora (400M+ tokens) without memory issues
- **Custom optimizers written from scratch**: Adam, AdamW (with decoupled weight decay), RMSprop, AdaGrad, and SGD with momentum
- **Noam-style learning rate schedule** with linear warmup followed by inverse-square-root decay
- **Gradient clipping**, checkpointing, and loss curve visualization built into the training loop
- **Top-k sampling with temperature control** for text generation

## Project structure

```
microgpt2/
├── main.py             # training entry point (CLI args, config, training loop)
├── generate.py          # text generation from a saved checkpoint
├── data.py              # tokenizer training, chunked tokenization, batching, optimizer config
├── models.py            # DecoderOnlyModel definition and training loop
├── building_blocks.py   # attention, positional encoding, feedforward, layer norm
├── optim.py             # custom Adam, AdamW, RMSprop, AdaGrad, SGD implementations
├── checkpoint.py         # checkpoint saving/loading
└── loss_curve.png        # training/validation loss plot from the last run
```

## Usage

## Before training

- If your dataset was split into chunk files (e.g. via a splitting script), concatenate them back into a single `.txt` file before training — `main.py` expects one plain-text file as `--train-path`, not a folder of chunks:
  ```bash
  cat dataset_chunks/*.txt > TinyStories-train.txt
  ```
- Make sure the `checkpoints/` and `tokenizer/` folders exist in the project directory before running `main.py` — the training script saves checkpoints and the trained tokenizer into these folders but does not create them automatically:
  ```bash
  mkdir -p checkpoints tokenizer
  ```

### Train a model

```bash
python main.py \
    --train-path data/TinyStories-train.txt \
    --vocab-size 8000 \
    --d-emb 384 \
    --heads 6 \
    --n-decoder-layers 8 \
    --block-size 128 \
    --batch-size 64 \
    --iterations 50000 \
    --warmup-steps 3500 \
    --optimizer adamw \
    --lr 0.001 \
    --beta2 0.95
```

Key arguments:

| Argument | Description | Default |
|---|---|---|
| `--train-path` | Path to a plain `.txt` training corpus | `shakespeare.txt` |
| `--vocab-size` | BPE tokenizer vocabulary size | `1000` |
| `--d-emb` | Embedding dimension | `384` |
| `--heads` | Number of attention heads | `6` |
| `--n-decoder-layers` | Number of decoder layers | `4` |
| `--block-size` | Context window (tokens) | `128` |
| `--optimizer` | `adam`, `adamw`, `sgd`, `rmsprop`, or `adagrad` | `adam` |
| `--iterations` | Total training steps | `2000` |
| `--warmup-steps` | LR warmup steps | `400` |

A tokenizer is trained automatically from the training corpus unless `--tokenizer-path` points to an existing one. Tokenized data is cached as a `.bin` file next to the source `.txt` for reuse.

### Generate text

```bash
python generate.py \
    --load-path checkpoints/checkpoint.pt \
    --prompt "Once upon a time, there was a" \
    --n-tokens 500 \
    --topk 30 \
    --temperature 0.7
```

## Notes

- The data pipeline currently expects a single plain-text file with UTF-8 encoding. TinyStories' raw files use `<|endoftext|>` as a story separator; a cleaning step is used beforehand to normalize this into blank-line-separated stories.
- Context length is currently limited to `block_size` tokens — generations longer than the training window will lose access to earlier context, which can affect long-range plot coherence.
- This project is primarily educational: the goal is understanding transformer internals and the LM training pipeline end-to-end, not state-of-the-art output quality. At small scale (~20M parameters), expect locally coherent grammar and sentence structure, with some drift in longer-range narrative consistency.

## Acknowledgments

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset by Ronen Eldan and Yuanzhi Li (Microsoft Research)
- Architecture and training approach inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and the original ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper
