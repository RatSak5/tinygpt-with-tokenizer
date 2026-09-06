import argparse

import matplotlib.pyplot as plt

from data import *
from models import DecoderOnlyModel
from optim import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny Char-level LM on Shakespeare."
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--train-path", type=str, default="data/clean_data.txt")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd", "rmsprop", "adagrad", "adamw"],
        type=str,
        default="adam",
    )

    parser.add_argument("--d-emb", type=int, default=384)
    parser.add_argument("--d-ff", type=int, default=0)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--n-decoder-layers", type=int, default=4)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--dampening", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--save-path", type=str, default="checkpoints/checkpoint.pt")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, or leave unset to auto-detect",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    data = make_data(
        train_ratio=args.train_ratio,
        path=args.train_path,
        vocab_size=args.vocab_size,
        tokenizer_path=args.tokenizer_path,
    )
    d_ff = 4 * args.d_emb if args.d_ff <= 0 else args.d_ff
    model_state = {
        "vocab_size": data["vocab_size"],
        "d_emb": args.d_emb,
        "heads": args.heads,
        "n_decoder_layers": args.n_decoder_layers,
        "d_ff": d_ff,
    }

    state = {
        "iterations": args.iterations,
        "warmup steps": args.warmup_steps,
        "train_ratio": args.train_ratio,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "optimizer": args.optimizer,
    }

    optim_vars = build_config(args)
    optims = {
        "adam": Adam,
        "sgd": SGD,
        "rmsprop": RMSprop,
        "adagrad": AdaGrad,
        "adamw": AdamW,
    }

    optimizer = args.optimizer
    model = DecoderOnlyModel(**model_state, device=args.device)
    print(f"Using device: {model.device}")
    optim = optims[optimizer](parameters=model.params, **optim_vars[optimizer])
    total_params = sum(t.nelement() for t in model.params)
    state["Total params"] = total_params

    print("Training State:")
    print_state(state)
    print()
    print("Model State:")
    print_state(model_state)

    train_state = {
        "warmup_steps": args.warmup_steps,
        "iterations": args.iterations,
        "eval_interval": args.eval_interval,
        "train_data": data["train_data"],
        "val_data": data["val_data"],
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "save_path": args.save_path,
        "save_data": data,
    }
    t_loss, v_loss = model.train(optim=optim, **train_state)

    # plot losses
    steps = [i * 10 for i in range(len(t_loss))]
    plt.plot(steps, t_loss, label="val loss")
    plt.plot(steps, v_loss, label="train loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()

    plt.savefig("loss_curves/loss_curve.png")


if __name__ == "__main__":
    main()
