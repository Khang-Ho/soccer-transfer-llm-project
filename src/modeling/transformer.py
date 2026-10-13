"""Small causal Transformer initialized from scratch, with no external weights."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 2048
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.1
    pad_id: int = 0

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "d_model", "n_heads", "n_layers", "feedforward_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1 or not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("Invalid dropout or padding ID")


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, causal_mask, padding_mask):
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=causal_mask, key_padding_mask=padding_mask, need_weights=False,
        )
        hidden = hidden + self.residual_dropout(attended)
        return hidden + self.residual_dropout(self.ff(self.ff_norm(hidden)))


class RecallTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        # Each block is separately constructed, avoiding identical cloned initialization.
        self.blocks = nn.ModuleList([CausalBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a 2D torch.long tensor")
        batch, length = input_ids.shape
        if batch == 0 or length == 0 or length > self.config.context_length:
            raise ValueError("Empty input or context limit exceeded; no truncation is performed")
        padding_mask = input_ids.eq(self.config.pad_id)
        # Right padding only; avoids rows with no visible keys in causal attention.
        if padding_mask[:, 0].any() or (padding_mask[:, :-1] & ~padding_mask[:, 1:]).any():
            raise ValueError("Sequences must be nonempty and right-padded")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
        for block in self.blocks:
            hidden = block(hidden, causal_mask, padding_mask)
        logits = self.output(self.final_norm(hidden))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.long:
                raise ValueError("labels must match input_ids shape and dtype")
            # Labels are already shifted by the data collator; do not shift again.
            targets = labels.masked_fill(padding_mask | labels.eq(self.config.pad_id), -100)
            if not targets.ne(-100).any():
                raise ValueError("Batch has no supervised target tokens")
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(self, prompt_ids, *, eos_id=2, max_new_tokens=128, forbidden_ids=(0, 1, 3)):
        """Greedy single-prompt generation; returns completion IDs and stop reason."""
        if not prompt_ids or self.config.pad_id in prompt_ids:
            raise ValueError("Prompt must be nonempty and contain no padding")
        if len(prompt_ids) > self.config.context_length or max_new_tokens < 1:
            raise ValueError("Invalid prompt length or generation budget")
        if eos_id in forbidden_ids:
            raise ValueError("EOS must remain generatable")
        was_training = self.training
        self.eval()
        sequence = list(prompt_ids)
        completion = []
        reason = "max_new_tokens"
        try:
            for _ in range(max_new_tokens):
                if len(sequence) > self.config.context_length:
                    reason = "context_limit"
                    break
                inputs = torch.tensor([sequence], dtype=torch.long, device=self.token_embedding.weight.device)
                scores = self(inputs)["logits"][0, -1].clone()
                scores[list(forbidden_ids)] = -torch.inf
                token = int(scores.argmax())
                completion.append(token)
                sequence.append(token)
                if token == eos_id:
                    reason = "eos"
                    break
        finally:
            self.train(was_training)
        return {"token_ids": completion, "stop_reason": reason}
