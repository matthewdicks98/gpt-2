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


# Transformer Model COnfig.
@dataclass
class GPTConfig:
    block_size: int = 1024    # Max sequence length / context window.
    vocab_size: int = 50257   # Number of tokens.
    n_layer: int = 12         # Number of transformer layers.
    n_head: int = 12          # Number of attention heads.
    n_embd: int = 768         # Embedding dimention.


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        """
        Takes in the transformer model config and builds the causal self-attention block.

        :param config: Transformer model config.
        """
        super().__init__()
        self.config = config

        # Initialize the projection matrices.
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Init the configs.
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Make sure we correctly scale the weights.
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.tensor):
        """
        Takes in a tensor of shape (batch, seq_len, embedding_dim). This is a batch of
        token embeddings on which we will apply the self-attention.

        Using self-attention this contextualizes the token embeddings based on what tokens the
        attention block makes it pay attention too.

        :param x: batch of token embeddings. Shape=(batch, seq_len, embedding_dim).
        :return: Contextualized token embeddings. Shape=(batch, seq_len, embedding_dim).
        """
        
        # Extract the shapes.
        batch_size, seq_len, embed_dim = x.size()

        # Compute the qkv values from the input all at once and reshape.
        qkv = self.c_attn(x)
        query, key, value = qkv.split(self.n_embd, dim=2)  # (B, T, C*3) -> [(B, T, C), (B, T, C), (B, T, C)]
        
        # Reshape the full embedding into multiple heads.
        # (batch, num_heads, seq_len, smaller embedding dim)
        query = query.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2)

        # Use Flash attention.
        y = f.scaled_dot_product_attention(query, key, value, is_causal=True)

         # Transform back to (B, T, C) (concat in multi-head attn).
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final projection after concat in multi-head attention.
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    
    def __init__(self, config: GPTConfig):
        """
        Takes in the transformer model config and builds the MLP block.

        :param config: Transformer model config.
        """
        super().__init__()
        self.config = config

        # Create the MLP layer.
        # TODO: Why are we upsampling by exactly 4 and does this help?
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.tensor):
        """
        Takes in the attention activations, upsamples, non-linear, downsamples.

        :param: Input tensor, usually the attention activations. Shape=(batch, seq_len, embedding_dim).
        :return: The resulting tensor. Shape=(batch, seq_len, embedding_dim).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        """
        Takes in the transformer model config and builds a single attention block.

        :param config: Transformer model config.
        """
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)  # Layer norm before attn.
        self.attn = CausalSelfAttention(config)  # Attention operation.
        self.ln_2 = nn.LayerNorm(config.n_embd)  # Layer norm before mlp.
        self.mlp = MLP(config)                   # MLP.

    def forward(self, x: torch.tensor):
        """
        Takes in an input tensor and applies the transformer block.

        :param: Input tensor. Shape=(batch, seq_len, embedding_dim).
        :return: The resulting tensor. Shape=(batch, seq_len, embedding_dim).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        """
        Takes in the configuration and builds the GPT model.
        """
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings.
                wte = nn.Embedding(config.vocab_size, config.n_embd),

                # Positional embeddings.
                wpe = nn.Embedding(config.block_size, config.n_embd),

                h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),

                ln_f = nn.LayerNorm(config.n_embd),
            )
        )

        # Final projection to map embedding to vocab size.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme (huge reduction in the number of weights).
        self.transformer.wte.weight = self.lm_head.weight

        # Apply takes the modules of the model and iterates over them.
        # We apply the function _init_weights to the modules.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialise the parameter weights.
        """
        if isinstance(module, nn.Linear):

            # Scale the standard deviations. 
            # TODO: Not sure why we need this but good to check.
            std=0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                # Set the bias to zero.
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            # Init the embedding weights using a normal dist.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.tensor, targets: torch.tensor = None):
        """
        Takes in a batch of token indices and optionally a batch of target indices.

        :param idx: Batch of token indices. Shape=(batch, seq_len).
        :param targets: Batch of target indices. Shape=(batch, seq_len).
        :return: Logits for each token in the vocab. Shape=(batch, seq_len, vocab_size).
        """

        # The idx's are the token indices. With a batch size of B and a sequence length of T.
        batch_size, seq_len = idx.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequences larger than, {self.config}. Yours is {seq_len}."

        # Get the position embeddings. We pluck them out of the rows of the wpe matrix.
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)

        # Get the token embeddings. We pluck the token embeddings out using the idxs.
        tok_emb = self.transformer.wte(idx)

        # Add the token and position embeddings.
        x = tok_emb + pos_emb

        # Forward the transformer blocks.
        for block in self.transformer.h:
            x = block(x)

        # Forward last layer norm.
        x = self.transformer.ln_f(x)

        # Forward the last liner layer to upsample to the vocab.
        logits = self.lm_head(x)

        # Compute the loss.
        loss = None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizer(self, weight_decay: float, learning_rate: float, device: str):
        """
        Configure the optimizer parameters.

        :param weight_decay: Weight decay for the optimizer.
        :param learning_rate: Learning rate for the optimizer.
        :param device: Device to run the optimizer on.
        :return: The optimizer.
        """
        # Get only the params that require grads.
        # TODO: Not sure why I would need both lines.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Split params into params that need to be decayed and ones that don't.
        # To decay it has to be at least 2 dims.
        # TODO: Not sure why dim >= 2 need to be decayed.
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
        # TODO: Understand this better.
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = "cuda" in device and fused_available
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.95), 
            eps=1e-8, 
            fused=use_fused
        )
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

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


def get_lr(it: int, min_lr: float, max_lr: float, warmup_steps: int, max_steps: int):
    """
    Gets the learning rate for the optimizer using the cosine decay schedule.

    :param it: Current step.
    :param min_lr: Minimum learning rate.
    :param max_lr: Maximum learning rate.
    :param warmup_steps: Number of warmup steps.
    :param max_steps: Maximum number of steps.
    :return: The learning rate.
    """

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


# ==================================================================
# DATALOADER.
# ==================================================================


class DataLoaderLite:

    def __init__(self, batch_size: int, seq_len: int):
        """
        Takes in the batch size and sequence length and loads the data.

        :param batch_size: Batch size.
        :param seq_len: Sequence length.
        """
        
        # Set the params.
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Load in the text file.
        with open(os.path.join(os.path.dirname(__file__), "input.txt")) as text_file:
            text = text_file.read()

        # Init the tokenizer.
        enc = tiktoken.get_encoding("gpt2")

        # Tokenize the text and covert to tensor.
        tokens = enc.encode(text=text)
        self.tokens = torch.tensor(tokens)

        # Set the current position.
        self.current_pos = 0

        # Print the total tokens and batches per epoch.
        print(f"Total tokens: {len(tokens)}")
        print(f"Batches per epoch: {len(tokens) // (batch_size * seq_len)}")

    def next_batch(self):
        """
        Gets the next batch of data.

        :return: The next batch of data. Shape=(batch_size, seq_len).
        """

        # Shapes.
        batch_size, seq_len = self.batch_size, self.seq_len    

        # Get next batch.
        next_pos = self.current_pos + batch_size * seq_len
        buf = self.tokens[self.current_pos:next_pos + 1]
        x = buf[:-1].view(batch_size, seq_len)
        y = buf[1:].view(batch_size, seq_len)

        # Compute new pos.
        self.current_pos = next_pos
        if next_pos + 1 > len(self.tokens):
            self.current_pos = 0

        return x, y


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
    # gpt-2 124M was 524_288.
    total_batch_size = 1024 * 4
    batch_size = 2  # Micro batch size.
    seq_len = 1024
    assert total_batch_size % (batch_size * seq_len) == 0
    grad_accum_steps = total_batch_size // (batch_size * seq_len)

    print("----- BATCH SIZE CONFIGS")
    print(f"Total desired batch size: {total_batch_size} tokens")
    print(f"Micro batch size: {batch_size * seq_len} tokens")
    print(f"grad_accum_steps: {grad_accum_steps}")

    # Set the data loader.
    train_loader = DataLoaderLite(batch_size=batch_size, seq_len=seq_len)

    # Set matmul precision TF32.
    torch.set_float32_matmul_precision("high")

    # Init model.
    model = GPT(GPTConfig(vocab_size=50_304))
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
        tokens_per_sec = (train_loader.batch_size * train_loader.seq_len * grad_accum_steps) / (t1 - t0)

        print(f"step: {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    print(f"total_time: {time.time() - t:.4f}s")
    print("loss", loss)
