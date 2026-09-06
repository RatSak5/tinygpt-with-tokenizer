import argparse

import torch

from checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny Char-level LM on Shakespeare."
    )
    parser.add_argument("--load-path", type=str, default="checkpoints/checkpoint.pt")
    parser.add_argument("--n-tokens", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)

    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, there lived a",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, or leave unset to auto-detect",
    )

    return parser.parse_args()


def generate(prompt, n_tokens, load_path, topk, temperature, device=None):
    generated_txt = prompt

    model, data = load_checkpoint(load_path, device=device)
    tok = data["tokenizer"]
    block_size = data["block_size"]
    itos = data["itos"]

    X = (
        torch.tensor(tok.encode(prompt).ids)[-block_size:].view(1, -1)
        if len(prompt) >= block_size
        else torch.tensor(tok.encode(prompt).ids).view(1, -1)
    )
    X = X.to(model.device)
    for step in range(n_tokens):
        with torch.no_grad():
            out = model(X)
            relevent = out[:, -1, :][0]

            TOP_K = topk
            TEMPERATURE = temperature

            t = torch.softmax(relevent / TEMPERATURE, dim=0)
            top_probs, top_idx = torch.topk(t, TOP_K)
            top_probs = top_probs / top_probs.sum()
            sampled = torch.multinomial(top_probs, num_samples=1).item()
            idx = top_idx[sampled].item()

        ch = itos[idx]
        if ch[0] == "Ġ":
            ch = " " + ch[1:]
        elif ch[0] == "Ċ":
            ch = "\n" + ch[1:]
        generated_txt += ch
        if X.shape[1] < block_size:
            X = torch.tensor([X.tolist()[0] + [idx]])
        else:
            X = torch.tensor([X.tolist()[0][1:] + [idx]])
        X = X.to(model.device)
    return generated_txt


if __name__ == "__main__":
    args = parse_args()
    generated_text = generate(
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        load_path=args.load_path,
        device=args.device,
        topk=args.topk,
        temperature=args.temperature,
    )
    print("Generated Text: ")
    print(generated_text)
