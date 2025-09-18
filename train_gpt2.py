from dataclasses import dataclass
import inspect
import math
import tiktoken
import os
import traceback
import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import time
import numpy as np
from data_loader_lite import DataLoaderLite
from hellaswag import iterate_examples, render_example
    

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


# ================================================================================
# EVALS
# ================================================================================

ENCODER = tiktoken.get_encoding("gpt2")
EOT = ENCODER._special_tokens["<|endoftext|>"]

def generate(model: GPT, prompt: str, num_generations: int, max_tokens: int, device: str):
    """
    Sample from the given model.
    """

    # --- Step _: Tokenize the prompt. ---

    prompt_tokens = ENCODER.encode_ordinary(text=prompt)
    prompt_tokens_np = np.array(prompt_tokens)

    assert prompt_tokens_np.shape[0] <= GPTConfig.block_size, f"Max seq len is {GPTConfig.block_size}"
    prompt_tokens_t = torch.from_numpy(prompt_tokens_np).to(dtype=torch.long, device=device)
    prompt_tokens_t = prompt_tokens_t.repeat(repeats=(num_generations, 1))

    # --- Step _: Forward through the model to get logits. ---
    
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    model.eval()
    with torch.no_grad():

        while prompt_tokens_t.shape[-1] < max_tokens:

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(prompt_tokens_t)  

            # Sample from the model using top k sampling.
            logits = logits[:, -1, :]
            probs = f.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
            topk_samples = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
            samples = torch.gather(input=topk_indices, dim=-1, index=topk_samples)

            # Concat the new tokens to the old tokens.
            prompt_tokens_t = torch.concat((prompt_tokens_t, samples), dim=-1)

    # Decode the tokens back to text so we can print.
    generations = []
    for i in range(prompt_tokens_t.shape[0]):
        decoded_seq = ENCODER.decode(tokens=prompt_tokens_t[i, :].tolist())
        generations.append(decoded_seq)

    return generations


def get_most_likely_row(model: GPT, tokens: torch.tensor, mask: torch.tensor):
    """
    This helps in evaluating HellaSwag.

    For the given tokens and the mask pick the row that is most likely.
    """
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(tokens)  

        # Shift the logits and tokens so we can compute the loss.
        # Makes sure we use the tokens as the labels.
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_tokens = tokens[:, 1:].contiguous()
        shifted_mask = mask[:, 1:].contiguous()

        # Compute the loss at each token.
        flat_logits = shifted_logits.view(-1, logits.shape[-1])
        flat_tokens = shifted_tokens.view(-1)
        loss = f.cross_entropy(flat_logits, flat_tokens, reduction="none")

        # For each option compute the average loss.
        loss = loss.view(-1, shifted_mask.shape[-1])
        loss = loss * shifted_mask
        loss_avg = loss.sum(dim=1) / shifted_mask.sum(dim=1)

        # Pick the option with the lowest loss.
        best_option = loss_avg.argmin().item()
        return best_option


def get_most_likely_row_karp(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = f.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


if __name__ == "__main__":

    # If we have multiple GPUs let's run DDP.
    # DDP will set:
    #   RANK - Global rank of the process in all processes (for multi-node).
    #   LOCAL_RANK - Rank of the process in the node.
    #   WORLD_SIZE - Total number of processes.
    run_with_ddp = os.environ.get("RANK", -1) != -1
    if run_with_ddp is True:
        assert torch.cuda.is_available(), "Need cuda for ddp."
        init_process_group(backend="nccl")
        ddp_rank = os.environ["RANK"]
        ddp_local_rank = os.environ["LOCAL_RANK"]
        ddp_word_size = os.environ["WORLD_SIZE"]
        device = f"cuda:{ddp_local_rank}"
    else:
        # Set the variables for a non-ddp run.
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_word_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master_process = ddp_local_rank == 0
    if master_process is True:
        print("----- DEVICE CONFIG")
        print(f"Using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Configure batch-sizings.
    # gpt-2 124M was 524_288.
    total_batch_size = 524_288  # 1024 * 4
    batch_size = 524_288  # 2  # Micro batch size.
    seq_len = 1024
    assert total_batch_size % (batch_size * seq_len * ddp_word_size) == 0
    grad_accum_steps = total_batch_size // (batch_size * seq_len * ddp_word_size)

    if master_process is True:
        print("----- BATCH SIZE CONFIGS")
        print(f"Total desired batch size: {total_batch_size} tokens")
        print(f"Micro batch size: {batch_size * seq_len} tokens")
        print(f"grad_accum_steps: {grad_accum_steps}")

    # Set the data loader.
    train_loader = DataLoaderLite(
        batch_size=batch_size, 
        seq_len=seq_len, 
        process_num=ddp_rank,
        num_processes=ddp_word_size,
        split="train"
    )
    val_loader = DataLoaderLite(
        batch_size=batch_size, 
        seq_len=seq_len, 
        process_num=ddp_rank,
        num_processes=ddp_word_size,
        split="val"
    )

    # Set matmul precision TF32.
    torch.set_float32_matmul_precision("high")

    # Init model.
    model = GPT(GPTConfig(vocab_size=50_304))
    model.to(device)

    # Wrap the model in the DDP container if needed.
    if run_with_ddp is True:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Set the model for the optimizer to work on.
    raw_model = model.module if run_with_ddp is True else model

    # Set optimizer.
    optimizer = raw_model.configure_optimizer(weight_decay=1, learning_rate=3e-4, device=device)

    # Set lr schedule.
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715  # 10
    max_steps = 19073  # 100 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

    # Set the eval params.
    eval_freq = 30
    val_loss_steps = 20

    # Create the logs and clear.
    log_dir = os.path.join(os.environ.get("PROJECT_DIR", os.path.expanduser("~")), "gpt2", "logs")
    os.makedirs(log_dir, exist_ok=True)

    t = time.time()
    lrs = []
    for step in range(max_steps):

        t0 = time.time()

        # Variables to use for logging.
        val_loss = None
        hella_acc = None
        generations = None
        time_to_eval = step % eval_freq == 0 or step == max_steps - 1

        if time_to_eval:

            # --- Step _: Compute the validation loss. ---

            model.eval()
            val_loader.reset()
            with torch.no_grad():
                
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device=device), y.to(device=device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)  

                    # Scale and accumulate loss.
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

                if run_with_ddp is True:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            val_loss = val_loss_accum.item()

            # --- Step _: Generate some samples. ---

            generations = generate(
                model=model, 
                prompt="Hello, I'm a language model,", 
                max_tokens=50, 
                num_generations=5,
                device=device
            )

            # --- Step _: Eval on HellaSwag. ---
            
            n_examples = 0
            n_correct = 0
            for hix, example in enumerate(iterate_examples(split="val")):
                
                if hix % ddp_word_size == ddp_local_rank:

                    # Split HellaSwag over processes.
                    # For this example predict which option is best.
                    data, tokens, mask, label = render_example(example=example)
                    tokens, mask = tokens.to(device=device), mask.to(device=device)
                    my_pred = get_most_likely_row(model=model, tokens=tokens, mask=mask)

                    n_examples += 1
                    n_correct += int(my_pred == label)
            
            if run_with_ddp is True:

                # Aggregate over processes.
                n_examples_t = torch.tensor(n_examples, device=device)
                n_correct_t = torch.tensor(n_correct, device=device)
                dist.all_reduce(n_examples_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(n_correct_t, op=dist.ReduceOp.SUM)
                n_examples = n_examples_t.item()
                n_correct = n_correct_t.item()

            # Compute the accuracy.
            hella_acc = n_correct / n_examples

        # --- Step _: TODO: Checkpoint the model. ---

        # --- Step _: Optimize the network. ---

        # Init the training.
        model.train()
        optimizer.zero_grad()
        
        # This is to simulate a larger batch-size than the gpus we have.
        loss_accum = 0.0
        for grad_accum_step in range(grad_accum_steps):

            # Get data from dataloader.
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if run_with_ddp is True:
                # If True then we need to sync grad across processes. 
                # If False then we just accumulate per process grads.
                # Avoids expensive all reduce on every grad accum step.
                model.require_backward_grad_sync = grad_accum_step == grad_accum_steps - 1

            # Forward through the model and calculate the loss.
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)  
            
            # Make sure the normalizer for cross-entropy is correct.
            loss /= grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            
        # Compute the average loss for all processes.
        if run_with_ddp is True:
            loss_accum = dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)        
        
        # Perform the grad update.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(it=step, max_lr=max_lr, min_lr=min_lr, max_steps=max_steps, warmup_steps=warmup_steps)
        lrs.append(lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        # --- Step _: Log details. ---

        # Make sure all gpu kernels have completed.
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000
        tokens_processed = train_loader.batch_size * train_loader.seq_len * grad_accum_steps * ddp_word_size
        tokens_per_sec = tokens_processed / (t1 - t0)

        if master_process is True:
            print(
                f"step: {step} | "
                f"loss: {loss_accum.item():.6f} | "
                f"lr: {lr:.4e} | "
                f"norm: {norm:.4f} | "
                f"dt: {dt:.2f}ms | "
                f"tok/sec: {tokens_per_sec:.2f}"
            )

            mode = "w" if step == 0 else "a"

            # Log the losses to a file.
            with open(os.path.join(log_dir, "losses.log"), mode) as f_losses:
                f_losses.write(f"{loss_accum.item()},{val_loss}\n")
            
            # Log the HellaSwag evals.
            with open(os.path.join(log_dir, "hellaswag.log"), mode) as f_hella:
                f_hella.write(f"{hella_acc}\n")

            # Log the generations.
            if isinstance(generations, list) and len(generations) > 0:
                generations_str = "\n".join(generations)
                with open(os.path.join(log_dir, "generations.log"), mode, encoding="utf-8") as f_gen:
                    f_gen.write("=" * 40)
                    f_gen.write("\n")
                    f_gen.write(f"Step - {step}")
                    f_gen.write("\n")
                    f_gen.write(generations_str)
                    f_gen.write("\n")
                    f_gen.write("=" * 40)
                    f_gen.write("\n")
    
        # ----------------------------

    if run_with_ddp is True:
        destroy_process_group()

    if master_process is True:
        print(f"total_time: {time.time() - t:.4f}s")
        print("loss", loss)
