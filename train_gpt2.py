from dataclasses import dataclass
import inspect
import math
import tiktoken
import os
import traceback
import torch
import torch.nn as nn
from torch.nn import functional as f
import time


# ==================================================================
# GPT2 MODEL.
# ==================================================================


# Define the config
@dataclass
class GPTConfig:
    block_size: int = 1024    # Max sequence length / context window.
    vocab_size: int = 50257   # Number of tokens.
    n_layer: int = 12         # Number of transformer layers.
    n_head: int = 12          # Number of attention heads.
    n_embd: int = 768         # Embedding dimention.


class CausalSelfAttention(nn.Module):

    # NOTE 1:
    #   n = 100
    #    x = torch.randn(768) * (n**-0.5)
    #    for i in range(n):
    #        x += torch.randn(768) * (n**-0.5)  # In the network i am not sure we are doing this.
    #
    #    x.std() --> it is now 1. If we didn't do the n**-0.5 it would be much higher.

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Key, query, value projections for all heads in a batch (that's why n_embd*3).
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # Final output projection after concat in multi-head attn.
        self.c_proj.NANOGPT_SCALE_INIT = 1  # To ensure we correctly scale down the weights (NOTE 1)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Mask so future tokens don't see past tokens.
        # Register tell torch this is not a parameter.
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)
            )
        )

    def forward(self, x: torch.tensor):
        
        # B=batch size, T=sequence length, C=embedding dim.
        B, T, C = x.size()

        # Instead of:
        # query, key, value = self.query(x), self.key(x), self.value(x)
        # we do it all at the same time and then reshape.
        # nh = number of heads, hd = dimension per head.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # (B, T, C*3) -> [(B, T, C), (B, T, C), (B, T, C)]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # # Regular attention operation.
        # attn = (q @ k.transpose(-2, -1)) / (k.size(-1)**(1/2))  # (B, nh, T, hd) x (B, nh, hd, T) -> (B, nh, T, T)
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # Mask the future tokens with -inf.
        # attn = f.softmax(attn, dim=-1)
        # y = attn @ v  # (B, nh, T, T) x (B, nh, T, hd) -> (B, nh, T, hd).
        
        # Flash attention.
        y = f.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Transform back to (B, T, C) (concat in multi-head attn).
        y = self.c_proj(y)  # Final projection after concat in multi-head attention.
        return y

class MLP(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)  # Layer norm before attn.
        self.attn = CausalSelfAttention(config)  # Attention operation.
        self.ln_2 = nn.LayerNorm(config.n_embd)  # Layer norm before mlp.
        self.mlp = MLP(config)                   # MLP.

    def forward(self, x: torch.tensor):
        
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings.
                wpe = nn.Embedding(config.block_size, config.n_embd),  # Positional embeddings.
                h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),  # List of transformer blocks.
                ln_f = nn.LayerNorm(config.n_embd),  # Layer norms.
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Final projection to map embedding to vocab size.

        # Weight sharing scheme (huge reduction in the number of weights).
        self.transformer.wte.weight = self.lm_head.weight

        # Apply takes the modules of the model and iterates over them.
        # We apply the function _init_weights to the modules.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights of the model"""
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.tensor, targets: torch.tensor = None):
        # The idx's are the token indices. With a batch size of B and a sequence length of T.
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequences larger than, {self.config}. Yours is {T}."

        # Get the position embeddings. We pluck them out of the rows of the wpe matrix.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)

        # Get the token embeddings. We pluck the token embeddings out using the idxs.
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embed)

        # Add the token and position embeddings.
        x = tok_emb + pos_emb  # (B, T, n_embed)

        # Forward the transformer blocks.
        for block in self.transformer.h:
            x = block(x)

        # Forward last layer norm.
        x = self.transformer.ln_f(x)  # (B, T, n_embd)

        # Forward the last liner layer.
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """
        Load pretrained GPT-2 model weights from HuggingFace and add them to our GPT model.
        """
        from transformers import GPT2LMHeadModel
        assert model_type in {"gpt2"}

        print(f"Loading pretrained weights for: {model_type}.")

        # Set the config params.
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # Create the config and our custom model.
        config = GPTConfig(**config_args)
        model = GPT(config=config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]  # Remove buffer.

        # Load the hf model.
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = model_hf.state_dict()
        sd_hf_keys = [k for k in sd_hf.keys() if not (k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"))]

        try:
            assert len(sd_hf_keys) == len(sd_keys)  # Make sure the keys are of the same length.
        except Exception as ex:
            print(f"sd_keys: {len(sd_keys)} != sd_hf_keys: {len(sd_hf_keys)}")
            traceback.print_exc()

        # Some params need to be transposed.
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight"
        ]

        # Set the hf model weights to our weights.
        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                # Some params need to be transposed.
                assert sd[k].shape == sd_hf[k].shape[::-1]
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay: float, learning_rate: float, device: str):
        """
        Configure the optimizer parameters.
        """
        # Get only the params that require grads.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}  # TODO: Not sure why I would need both lines.

        # Split params into params that need to be decayed and ones that don't.
        # To decay it has to be at least 2 dims.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num_decay_tensors: {len(decay_params)} | num_decay_params: {num_decay_params:_}")
        print(f"num_nodecay_tensors: {len(nodecay_params)} | num_nodecay_params: {num_nodecay_params:_}")

        # Create the optimizers.
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = "cuda" in device and fused_available
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

# ==================================================================
# DATALOADER.
# ==================================================================


class DataLoaderLite:

    def __init__(self, B, T):
        
        # Set the params.
        self.B = B
        self.T = T

        # Load in the text file and tokenize.
        with open(os.path.join(os.path.dirname(__file__), "input.txt")) as text_file:
            text = text_file.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text=text)
        self.tokens = torch.tensor(tokens)

        print(f"Total tokens: {len(tokens)}")
        print(f"Batches per epoch: {len(tokens) // (B * T)}")

        self.current_pos = 0

    def next_batch(self):

        # Shapes.
        B, T = self.B, self.T

        # Get next batch.
        next_pos = self.current_pos + B * T
        buf = self.tokens[self.current_pos:next_pos + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Compute new pos.
        self.current_pos = next_pos
        if next_pos + 1 > len(self.tokens):
            self.current_pos = 0

        return x, y


def get_lr(it: int, min_lr: float, max_lr: float, warmup_steps: int, max_steps: int):

    # Linear warm up.
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # If it > max_steps, return min learning rate.
    if it > max_steps:
        return min_lr

    # Cosine decay otherwise.
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coef * (max_lr - min_lr)

if __name__ == "__main__":

    # Auto detect device.
    print("----- DEVICE CONFIG")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Configure batch-sizings.
    total_batch_size = 1024 * 4  # gpt-2 124M was 524_288
    B = 2  # Micro batch size.
    T = 1024
    assert total_batch_size % (B * T) == 0
    grad_accum_steps = total_batch_size // (B * T)

    print("----- BATCH SIZE CONFIGS")
    print(f"Total desired batch size: {total_batch_size} tokens")
    print(f"Micro batch size: {B * T} tokens")
    print(f"grad_accum_steps: {grad_accum_steps}")

    # Set the data loader.
    train_loader = DataLoaderLite(B=2, T=1024)

    # Set matmul precision TF32.
    torch.set_float32_matmul_precision("high")

    # Init model.
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)

    # Set optimizer.
    print("----- OPTIMIZER CONFIG")
    optimizer = model.configure_optimizer(weight_decay=1, learning_rate=3e-4, device=device)

    # Set lr schedule.
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50

    print("----- STEPPING")

    t = time.time()
    lrs = []
    for step in range(max_steps):

        t0 = time.time()
        optimizer.zero_grad()  # Set gradients to 0.

        # Perform gradient accumulation.
        # This is to simulate a larger batch-size than the gpus we have.
        loss_accum = 0.0
        for grad_accum_step in range(grad_accum_steps):

            # Get data from dataloader.
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # Accumulate the grads for this batch.
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)  # Forward through the model.
            
            loss /= grad_accum_steps  # Make sure the normalizer for cross-entropy is correct.
            loss_accum += loss.detach()
            loss.backward()  # Accum grads.

        # Perform the grad update.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(it=step, max_lr=max_lr, min_lr=min_lr, max_steps=max_steps, warmup_steps=warmup_steps)
        lrs.append(lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()  # Optimized step.

        torch.cuda.synchronize()  # Make sure all gpu kernels have completed.
        t1 = time.time()
        dt = (t1 - t0)*1000
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)

        print(f"step: {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    print(f"total_time: {time.time() - t:.4f}s")
    print("loss", loss)

    # Quite before the hf test.
    quit()
    num_return_sequences = 5
    max_length = 30
    model = GPT.from_pretrained(model_type="gpt2")
    model.eval()
    model.to(device)

    # Get the tokens.
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)  # (T,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (B, T)
    x = tokens.to(device)

    # Set the seeds.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Sample from the model.
    for _ in range(max_length):

        with torch.no_grad():

            # Get the logits for each sequence.
            logits = model(x)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size), only want the last token.

            # Get the probabilities for each token in each batch. (B, vocab_size)
            probs = f.softmax(logits, dim=-1)

            # Pick the top k probs. (B, 50).
            topk_probs, topk_idxs = torch.topk(probs, k=50, dim=-1)

            # Use a multi-nomial dist to sample the probs. (B, 1).
            ix = torch.multinomial(topk_probs, num_samples=1)

            # Get the indices in each row we sampled. (B, 1).
            sampled_token_idxs = torch.gather(input=topk_idxs, dim=-1, index=ix)

            # Add the tokens to the sequence. (B, T+1).
            x = torch.cat([x, sampled_token_idxs], dim=1)

# Print the samples.
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens=tokens)
    print(">", decoded)
