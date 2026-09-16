"""Tiny GPT-style decoder-only transformer (~10M params), nanoGPT-style.

Uses a char-level vocab (built from the training data itself) instead of a
BPE tokenizer, since a full BPE vocab (50k+ tokens) would make the embedding
table alone bigger than the whole rest of the model at this parameter budget.

START HERE if you want to understand the model architecture itself.
Reading order: GPTConfig -> CausalSelfAttention -> MLP -> Block -> GPT.
Each class is a layer of the same standard GPT-2-style design, just small.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """All the numbers that decide the model's size and shape.

    vocab_size: how many distinct tokens (here: characters) exist.
    block_size: max context length in tokens the model can look back at.
    n_layer: how many transformer blocks are stacked.
    n_head: how many attention heads per block (n_embd must divide evenly).
    n_embd: the width of every hidden vector flowing through the model.
    dropout: fraction of activations randomly zeroed during training, to
        reduce overfitting (turned off automatically at eval time).
    """

    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Lets each token look back at earlier tokens (never forward) and decide
    which of them to "pay attention to" when building its own representation.
    This is the mechanism that lets the model use context, not just the last
    character, to predict the next one.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # One linear layer produces Query, Key, and Value all at once (faster
        # than three separate layers); we split it into q/k/v below.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        # A lower-triangular matrix of 1s: mask[i][j] = 1 means token i is
        # allowed to see token j. Since it's lower-triangular, token i can
        # only see tokens at positions <= i -- i.e. the past, not the future.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # x shape: (Batch, Time/sequence position, Channels/n_embd)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # Reshape so each attention head gets its own slice of the embedding,
        # and heads become a separate batch-like dimension we can compute in
        # parallel: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # "How much should token i attend to token j?" -- dot product of
        # query i with key j, for every pair, scaled to keep gradients stable.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Block out future positions by setting their score to -infinity, so
        # after softmax they become exactly 0 probability.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Each token's new representation = weighted average of all values
        # it's allowed to see, weighted by the attention scores above.
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    """Per-token feed-forward network: after attention mixes information
    *between* tokens, this lets each token process that information on its
    own. Expands to 4x width then back down, which is the standard GPT ratio.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer layer: attention (mix across tokens), then MLP
    (process each token). The `x = x + ...` lines are residual connections --
    the layer learns a correction to add to its input rather than replacing
    it outright, which makes deep stacks of these much easier to train.
    LayerNorm before each sub-layer keeps activations at a stable scale.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full model: turn token ids into vectors, run them through the
    stack of Blocks, then project back to a score per vocab token (the
    "logits") at every position.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # tok_emb: a learned vector per character. pos_emb: a learned vector
        # per position (0, 1, 2, ...) so the model can tell token order apart
        # -- without this, "cat" and "tac" would look identical to it.
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: reuse the token embedding matrix as the output layer
        # too. Standard GPT-2 trick -- saves a whole vocab_size x n_embd
        # matrix of parameters, which matters a lot at this small a scale.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Start every weight from small random values instead of the
        library defaults. Not critical at this size, but standard practice
        so training starts from a predictable, well-behaved state."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        """Total trainable parameter count -- how we check we're actually
        near the ~10M target."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx: a batch of token-id sequences, shape (Batch, Time).
        If targets is given (the "correct next token" at each position),
        also compute the training loss, so this one function serves both
        inference and training.
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.dropout(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # shape (B, T, vocab_size): a score per possible next char, at every position

        loss = None
        if targets is not None:
            # Standard next-token cross-entropy: at every position, how
            # surprised was the model by the actual next character?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """Autoregressive sampling: repeatedly predict the next character,
        append it, and feed the whole thing back in. This is how you get
        text *out* of the model after training, one character at a time.

        temperature: >1 makes output more random, <1 makes it more
            confident/repetitive, 0 would always pick the single best guess.
        top_k: if set, only sample from the k most likely next characters,
            to avoid occasionally picking a very unlikely/garbage one.
        """
        for _ in range(max_new_tokens):
            # Only the last block_size tokens fit in the model's context.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only care about the prediction for the *next* position
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # randomly sample one char, weighted by probability
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    # Quick sanity check you can run directly: `python model.py`
    # just to confirm the parameter count lands near the ~10M target.
    cfg = GPTConfig(vocab_size=100)
    model = GPT(cfg)
    print(f"params: {model.num_params():,}")
