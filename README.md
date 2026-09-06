# nano-narrator

Basically a transformer like the ShakespeareGPT repo except it has a tokenizer and takes substantially more time to train.

I personally trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset to generate short, simple English stories.

## Project structure

```
microgpt2/
├── main.py             # training entry point
├── generate.py          # text generation from a saved checkpoint
├── data.py              # tokenizer training, chunked tokenization, batching, optimizer config
├── models.py            # DecoderOnlyModel definition and training loop
├── building_blocks.py   # attention, positional encoding, feedforward, layer norm
├── optim.py             # Adam, AdamW, RMSprop, AdaGrad, SGD implementations
└── checkpoint.py         # checkpoint saving/loading
```

## A short example
This is an example of a story produced by the model I trained, given the prompt "Once upon a time, there lived a"

```
Once upon a time, there lived a little girl called Lucy. She was three years old and very happy. One day, Lucy decided to go for a walk in the garden. She walked up the hill in a tall grass.
As she walked, she noticed something strange. There was a big black, brown bear sitting on a tree. Lucy was so scared she did not understand why the bear was so loud. "Help!" she shouted.
The bear didn't come down or it growled. "Do you want to catch me?" he asked.
Lucy was scared, but the bear smiled. "Please don't bite me," he said.
"Of course I can," said the bear. "I'll be brave and share your food with you."
Lucy was so happy she jumped up and down. But then she heard a loud noise.
"It's okay," said the bear. "I'm just trying to catch you. I'm very afraid of the woods."
Lucy hugged the bear and said, "I'm sure you should be careful." The bear smiled and said, "I will always protect you away from the forest. It's like you're always brave to explore."
```

## Usage

## Before training

- If your dataset was split into chunk files (e.g. via a splitting script), concatenate them back into a single `.txt` file before training — `main.py` expects one plain-text file as `--train-path`, not a folder of chunks:
  ```bash
  cat dataset_chunks/*.txt > data/clean_data.txt
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

- The given numbers above are also the numbers I used while training the particular model which produced the above example.
- The data pipeline currently expects a single plain-text file with UTF-8 encoding. TinyStories' raw files use `<|endoftext|>` as a story separator; a cleaning step is used beforehand to normalize this into blank-line-separated stories.
- Context length is currently limited to `block_size` tokens — generations longer than the training window will lose access to earlier context, which can affect long-range plot coherence.
- This project is primarily educational: the goal is understanding transformer internals and the LM training pipeline end-to-end, not state-of-the-art output quality. At small scale (~20M parameters), expect locally coherent grammar and sentence structure, with some drift in longer-range narrative consistency.

## Acknowledgments

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset by Ronen Eldan and Yuanzhi Li (Microsoft Research)
- Architecture and training approach inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and the original ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper
