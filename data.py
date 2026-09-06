import torch

import os
import numpy as np
 
from tokenizers import (
    ByteLevelBPETokenizer,
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)


def make_tokenizer_word_level(file_path, vocab_size):
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tok.train(files=[file_path], trainer=trainer)

    tok.save("word_tokenizer.json")

    return Tokenizer.from_file("word_tokenizer.json")


def make_tokenizer_BPE(file_path, vocab_size):
    tok = ByteLevelBPETokenizer()
    tok.train(files=file_path, vocab_size=vocab_size, min_frequency=50)
    tok.save_model("tokenizer")

    return ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
 
def make_data(train_ratio, path, vocab_size, tokenizer_path=None, chunk_size_mb=16):
    tok = (
        make_tokenizer_BPE(path, vocab_size)
        if tokenizer_path is None
        else ByteLevelBPETokenizer(
            f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt"
        )
    )
 
    assert vocab_size <= 65535, "uint16 storage assumes vocab_size fits in 16 bits"
 
    bin_path = os.path.splitext(path)[0] + "_tokens.bin"
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
 
    print("Tokenizing in chunks and writing to disk...")
    total_tokens = 0
    with open(path, "r", encoding="utf-8") as f_in, open(bin_path, "wb") as f_out:
        leftover = ""
        while True:
            chunk = f_in.read(chunk_size_bytes)
            if not chunk:
                break
                
            chunk = leftover + chunk
            split_point = chunk.rfind(" ")
            if split_point == -1:
                to_encode, leftover = chunk, ""
            else:
                to_encode, leftover = chunk[:split_point], chunk[split_point:]
 
            ids = tok.encode(to_encode).ids
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f_out)
            total_tokens += len(ids)
 
            print(f"  ...{total_tokens:,} tokens written so far")
 
        if leftover.strip():
            ids = tok.encode(leftover).ids
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f_out)
            total_tokens += len(ids)
 
    print("Corpus Length:", total_tokens)

    full_data = np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(total_tokens,))
 
    n = int(train_ratio * total_tokens)
    train_data = full_data[:n]
    val_data = full_data[n:]
 
    itos = {i: tok.id_to_token(i) for i in range(tok.get_vocab_size())}
 
    return {
        "vocab_size": vocab_size,
        "train_data": train_data,
        "val_data": val_data,
        "tokenizer": tok,
        "itos": itos,
    }
 
def batch(split, batch_size, train_data, val_data, block_size, device="cpu"):
    split_dict = {
        "train": train_data,
        "val": val_data,
    }
    data = split_dict[split]
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
 
def build_config(args):
    optim_defaults = {
        "adam": {"lr": 0.1, "beta1": 0.999, "beta2": 0.999, "eps": 1e-8},
        "adamw": {"lr": 0.001, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.1},
        "rmsprop": {"lr": 0.01, "gamma": 0.9, "eps": 1e-5},
        "adagrad": {"lr": 0.01, "eps": 1e-5},
        "sgd": {"lr": 0.001, "momentum": 0.0, "dampening": 0.0},
    }
    defaults = optim_defaults[args.optimizer]
    for key, default_val in defaults.items():
        user_val = getattr(args, key, None)
        defaults[key] = user_val if user_val is not None else default_val
    return optim_defaults


def print_state(state):
    for k, v in state.items():
        if k == "lr":
            continue      # lr is printed seperately while training
        print(f"              {k:10s} : {v}")
