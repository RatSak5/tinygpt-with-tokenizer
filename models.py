import torch
import torch.nn.functional as F

from building_blocks import *
from data import batch


class EncoderLayer:
    def __init__(self, d_emb, d_ff, multi_head, device=None):
        self.self_attention = SelfAttention(d_emb, multi_head, device=device)
        self.feed_forward = FFN(d_emb, d_ff, device=device)
        self.layer_norm_gamma1 = torch.ones(1, d_emb, device=device)
        self.layer_norm_beta1 = torch.zeros(1, d_emb, device=device)
        self.layer_norm_gamma2 = torch.ones(1, d_emb, device=device)
        self.layer_norm_beta2 = torch.zeros(1, d_emb, device=device)
        self.eps = 1e-5

        self.sub_layers = [self.self_attention, self.feed_forward]
        self.params = [
            self.layer_norm_gamma1,
            self.layer_norm_beta1,
            self.layer_norm_gamma2,
            self.layer_norm_beta2,
        ]

    def __call__(self, X_pos_encoded, Xk=None):
        variance1 = ((X_pos_encoded - X_pos_encoded.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True)
        X_norm1 = (self.layer_norm_gamma1 * ((X_pos_encoded - X_pos_encoded.mean(dim=-1).unsqueeze(-1)) / (variance1 + self.eps) ** 0.5) + self.layer_norm_beta1)
        X_s_attended = self.self_attention(X_norm1)
        X_next1 = X_norm1 + X_s_attended

        variance2 = ((X_next1 - X_next1.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True)
        X_norm2 = (self.layer_norm_gamma2 * ((X_next1 - X_next1.mean(dim=-1).unsqueeze(-1)) / (variance2 + self.eps) ** 0.5) + self.layer_norm_beta2)
        X_ffn = self.feed_forward(X_norm2)
        self.out = X_norm2 + X_ffn
        
        return self.out

        return self.out

    def parameters(self):
        params = list(self.params)
        for layer in self.sub_layers:
            params += layer.parameters()
        return params


class DecoderLayer:
    def __init__(self, d_emb, d_ff, multi_head, cross_attention=True, device=None):
        self.self_attention = SelfAttention(
            d_emb, multi_head, masked=True, device=device
        )
        self.cross_attention = CrossAttention(d_emb, multi_head, device=device)
        self.ca = cross_attention
        self.feed_forward = FFN(d_emb, d_ff, device=device)
        self.layer_norm_gamma1 = torch.ones(1, d_emb, device=device)
        self.layer_norm_beta1 = torch.zeros(1, d_emb, device=device)
        self.layer_norm_gamma2 = torch.ones(1, d_emb, device=device)
        self.layer_norm_beta2 = torch.zeros(1, d_emb, device=device)
        self.layer_norm_gamma3 = torch.ones(1, d_emb, device=device)
        self.layer_norm_beta3 = torch.zeros(1, d_emb, device=device)
        self.eps = 1e-5

        self.sub_layers = [self.self_attention, self.feed_forward]
        self.params = [
            self.layer_norm_gamma1,
            self.layer_norm_beta1,
            self.layer_norm_gamma3,
            self.layer_norm_beta3,
        ]

        if cross_attention:
            self.params += [self.layer_norm_gamma2, self.layer_norm_beta2]
            self.sub_layers.append(self.cross_attention)

    def __call__(self, X_pos_encoded, Xk=None):
        variance1 = ((X_pos_encoded - X_pos_encoded.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True)
        X_norm1 = (self.layer_norm_gamma1 * ((X_pos_encoded - X_pos_encoded.mean(dim=-1).unsqueeze(-1)) / (variance1 + self.eps) ** 0.5) + self.layer_norm_beta1)
        X_s_attended = self.self_attention(X_norm1)
        X_next = X_norm1 + X_s_attended

        if self.ca:
            variance2 = ((X_next - X_next.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True)
            X_norm2 = (self.layer_norm_gamma2 * ((X_next - X_next.mean(dim=-1).unsqueeze(-1))/ (variance2 + self.eps) ** 0.5) + self.layer_norm_beta2)
            X_c_attended = self.cross_attention(X_norm2, Xk)
            X_next = X_norm2 + X_c_attended
            
        variance3 = ((X_next - X_next.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True)
        X_norm3 = (self.layer_norm_gamma3 * ((X_next - X_next.mean(dim=-1).unsqueeze(-1)) / (variance3 + self.eps) ** 0.5) + self.layer_norm_beta3)
        X_ffn = self.feed_forward(X_norm3)
        self.out = X_norm3 + X_ffn
        
        return self.out

    def parameters(self):
        params = list(self.params)
        for layer in self.sub_layers:
            params += layer.parameters()
        return params


# kept for reference / testing
class Transformer:
    def __init__(self, encoders, decoders):
        self.encoders = encoders  # list containing encoder objects
        self.decoders = decoders  # list containing decoder objects
        self.params = [
            p for layer in (self.encoders + self.decoders) for p in layer.parameters()
        ]

    def __call__(self, input_encoding, output_encoding):
        x = input_encoding
        for encoder in self.encoders:
            x = encoder(x)
        # x will be the encoding by now
        y = output_encoding
        for decoder in self.decoders:
            y = decoder(x, y)
        return y

    def parameters(self):
        return self.params


class DecoderOnlyModel:
    def __init__(self, vocab_size, d_emb, heads, n_decoder_layers, d_ff=0, device=None):
        self.d_emb = d_emb
        self.heads = heads
        self.n_decoder_layers = n_decoder_layers
        self.vocab_size = vocab_size
        self.d_ff = 4 * d_emb if d_ff <= 0 else d_ff

        # auto-detect GPU if no device was specified
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.Embed = Embedding(vocab_size, d_emb, device=self.device)
        self.Pos_enc = PositionalEncode(d_emb, device=self.device)

        self.Decoders = [
            DecoderLayer(
                d_emb, self.d_ff, heads, cross_attention=False, device=self.device
            )
            for _ in range(n_decoder_layers)
        ]

        self.lin = Linear(d_emb, vocab_size, device=self.device)
        self.lin.weights.data *= 0.01
        self.layers = [self.Embed, self.Pos_enc] + self.Decoders + [self.lin]
        self.params = [p for layer in self.layers for p in layer.parameters()]

        for p in self.params:
            p.requires_grad = True

    def __call__(self, X):
        X = X.to(self.device)
        for layer in self.layers:
            X = layer(X)
        self.out = X
        return self.out

    def parameters(self):
        return self.params

    def train(
        self,
        optim,
        warmup_steps,
        iterations,
        eval_interval,
        train_data,
        val_data,
        batch_size,
        block_size,
        save_path,
        save_data,
    ):
        from checkpoint import save_checkpoint

        t_loss, v_loss = [], []
        optim.describe()

        save_data["block_size"] = block_size
        del save_data["train_data"]
        del save_data["val_data"]

        for step in range(iterations):
            X, Y = batch(
                split="train",
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
            )
            out = self(X)
            Y = Y.to(self.device)
            loss = F.cross_entropy(out.transpose(1, 2), Y)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            update_steps = step + 1
            scale = 1.5e-4 / (self.d_emb**-0.5 * warmup_steps**-0.5)
            lr = (
                scale
                * (self.d_emb**-0.5)
                * min(update_steps**-0.5, update_steps * warmup_steps**-1.5)
            )

            optim.lr = lr
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=5.0)
            optim.step()

            if step % eval_interval == 0:
                with torch.no_grad():
                    X_v, Y_v = batch(
                        "val",
                        train_data=train_data,
                        val_data=val_data,
                        batch_size=batch_size,
                        block_size=block_size,
                    )
                    out_v = self(X_v)
                    Y_v = Y_v.to(self.device)
                    loss_v = F.cross_entropy(out_v.transpose(1, 2), Y_v)

                t_loss.append(loss.item())
                v_loss.append(loss_v.item())

                if step % 500 == 0:
                    save_checkpoint(model=self, data=save_data, path=save_path + str(step))
                    print("Saved checkpoint at : ", save_path + str(step))
                if step % 100 == 0:
                    print(f"iteration: {step:5d} | loss: {loss.item():.4f} | val loss: {loss_v.item():.4f}")
                    print(f"----learn: {lr:.10f}")
        save_checkpoint(model=self, data=save_data, path=save_path+"_final")
        return t_loss, v_loss
