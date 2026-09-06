import torch


class Linear:
    def __init__(self, fan_in, fan_out, bias=True, device=None):
        self.fan_in = fan_in
        self.weights = torch.randn(fan_in, fan_out, device=device) / fan_in**0.5
        self.biases = torch.zeros(fan_out, device=device) if bias else None

    def __call__(self, inp):
        prod = inp @ self.weights
        if self.biases is not None:
            prod += self.biases
        self.out = prod
        return self.out

    def parameters(self):
        biases = [self.biases] if self.biases is not None else []
        return [self.weights] + biases


class Embedding:
    def __init__(self, vocab_size, d_emb, device=None):
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.device = device

        self.weight = torch.randn((vocab_size, d_emb), device=device) / d_emb**0.5

    def __call__(self, inp):
        self.out = self.weight[inp]
        return self.out

    def parameters(self):
        return [self.weight]

    def reset_params(self):
        self.weight = torch.randn((self.vocab_size, self.d_emb), device=self.device)


class SelfAttention:
    def __init__(self, d_emb, heads, d_k=None, masked=False, device=None):
        self.heads = heads
        self.masked = masked
        self.d_v = d_k if d_k is not None else d_emb // heads

        self.Wq = torch.randn(d_emb, heads * self.d_v, device=device) / (d_emb**0.5)
        self.Wk = torch.randn(d_emb, heads * self.d_v, device=device) / (d_emb**0.5)
        self.Wv = torch.randn(d_emb, heads * self.d_v, device=device) / (d_emb**0.5)
        self.Wo = torch.randn(heads * self.d_v, d_emb, device=device) / (
            (heads * self.d_v) ** 0.5
        )

    def __call__(self, X):
        B, N, _ = X.shape

        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        Q = Q.view(B, N, self.heads, self.d_v).transpose(1, 2)
        K = K.view(B, N, self.heads, self.d_v).transpose(1, 2)
        V = V.view(B, N, self.heads, self.d_v).transpose(1, 2)

        att_out = (Q @ K.transpose(-2, -1)) / (self.d_v**0.5)

        if self.masked:
            inf = torch.full((N, N), float("-inf"), device=X.device)
            inf = torch.tril(inf, diagonal=-1).T
            att_out = torch.tril(att_out) + inf

        att = torch.softmax(att_out, dim=-1)
        out = att @ V

        out = out.transpose(1, 2).contiguous().view(B, N, self.heads * self.d_v)

        self.out = out
        if self.heads > 1:
            self.out = self.out @ self.Wo

        return self.out

    def parameters(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]


class CrossAttention:
    def __init__(self, d_emb, heads, d_k=None, device=None):
        self.heads = heads
        self.d_v = d_k if d_k is not None else d_emb // heads
        self.Wq = torch.randn(d_emb, heads * self.d_v, device=device) / (d_emb**0.5)
        self.Wk = torch.randn(d_emb, heads * self.d_v, device=device) / (d_emb**0.5)
        self.Wv = torch.randn(d_emb, heads * self.d_v, device=device) / (d_emb**0.5)
        self.Wo = torch.randn(heads * self.d_v, d_emb, device=device) / (
            (heads * self.d_v) ** 0.5
        )

    def __call__(self, Query, Key):
        B, Nq, _ = Query.shape
        Nk = Key.shape[1]

        Q = Query @ self.Wq  # (B, Nq, heads*d_v)
        K = Key @ self.Wk  # (B, Nk, heads*d_v)
        V = Key @ self.Wv

        Q = Q.view(B, Nq, self.heads, self.d_v).transpose(1, 2)  # (B, heads, Nq, d_v)
        K = K.view(B, Nk, self.heads, self.d_v).transpose(1, 2)  # (B, heads, Nk, d_v)
        V = V.view(B, Nk, self.heads, self.d_v).transpose(1, 2)

        att_out = (Q @ K.transpose(-2, -1)) / (self.d_v**0.5)  # (B, heads, Nq, Nk)
        att = torch.softmax(att_out, dim=-1)
        out = att @ V  # (B, heads, Nq, d_v)

        out = out.transpose(1, 2).contiguous().view(B, Nq, self.heads * self.d_v)

        self.out = out
        if self.heads > 1:
            self.out = self.out @ self.Wo
        return self.out

    def parameters(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]


class PositionalEncode:
    def __init__(self, n_embd, max_len=1024, device=None):
        # positional embeddings
        pe = torch.zeros(max_len, n_embd, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(
            1
        )
        C = -torch.log(torch.tensor(10000.0, device=device))
        div_term = torch.exp(
            torch.arange(0, n_embd, 2, device=device).float() * (C / n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe

    def __call__(self, X):
        self.out = X + torch.stack([self.pe[: X.size(1), :]] * X.size(0))
        return self.out

    def parameters(self):
        return []


class FFN:
    def __init__(self, d_emb, d_ff, device=None):
        self.W1 = torch.randn(d_emb, d_ff, device=device) * 0.01
        self.b1 = torch.randn(1, d_ff, device=device) * 0.001
        self.W2 = torch.randn(d_ff, d_emb, device=device) * 0.01
        self.b2 = torch.randn(1, d_emb, device=device) * 0.001

    def __call__(self, X):
        hidden_layer = X @ self.W1 + self.b1
        self.out = torch.clamp(hidden_layer, min=0) @ self.W2 + self.b2
        return self.out

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
