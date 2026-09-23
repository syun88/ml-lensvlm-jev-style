"""Jev-style V2: question-conditioned visual-token routing.

V1 reduced each page and question to one mean/max vector. V2 keeps a compact
sequence for both modalities and makes the page decision condition on token-level
question/page interactions.

The cache format is deliberately compact:
- contextual question tokens from the frozen LensVLM/Qwen text backbone
- spatially pooled visual page tokens from the frozen LensVLM vision tower
- both mapped through the SAME fixed random projection

The trainable router is small and non-autoregressive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecisionOutputV2:
    logits: torch.Tensor
    choice_probs: torch.Tensor
    noul_probs: torch.Tensor
    top1_index: torch.Tensor
    confidence_margin: torch.Tensor
    normalized_entropy: torch.Tensor
    temperature: torch.Tensor


def _inverse_softplus(x: float) -> float:
    import math
    return math.log(math.expm1(x))


class JevStyleV2Head(nn.Module):
    """Token-level question-conditioned page scorer.

    Inputs:
      question_tokens: [B, T, Din]
      question_mask:   [B, T]
      page_tokens:     [B, N, K, Din]
      page_mask:       [B, N]

    Architecture:
      1) shared low-dimensional projection for both modalities
      2) lightweight question self-attention
      3) question -> page cross-attention for every candidate page in parallel
      4) token-level fusion and masked mean/max aggregation
      5) one shared scalar scorer per page

    One scalar score is exposed as both Choice (softmax across pages) and Noul
    (independent sigmoid per page).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_p = dropout

        # Shared input mapping because V2 cache puts question and vision tokens
        # in the same fixed projected space.
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        self.q_norm1 = nn.LayerNorm(hidden_dim)
        self.q_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.q_norm2 = nn.LayerNorm(hidden_dim)
        self.q_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.cross_q_norm = nn.LayerNorm(hidden_dim)
        self.cross_p_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Per question-token evidence features:
        # q, attended-page, product, absolute difference.
        self.token_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # mean || max over question-token evidence features.
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.log_temperature = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_temperature), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 1e-4

    def forward(
        self,
        question_tokens: torch.Tensor,
        question_mask: torch.Tensor,
        page_tokens: torch.Tensor,
        page_mask: torch.Tensor,
    ) -> DecisionOutputV2:
        question_mask = question_mask.bool()
        page_mask = page_mask.bool()

        bsz, n_pages, page_k, _ = page_tokens.shape
        _, q_len, _ = question_tokens.shape

        q = self.input_proj(question_tokens)
        p = self.input_proj(page_tokens)

        # Small contextual refinement after the fixed random projection.
        qn = self.q_norm1(q)
        qa, _ = self.q_attn(
            qn,
            qn,
            qn,
            key_padding_mask=~question_mask,
            need_weights=False,
        )
        q = q + qa
        q = q + self.q_ff(self.q_norm2(q))
        q = q.masked_fill(~question_mask.unsqueeze(-1), 0.0)

        # Compare every page to the question in one batched cross-attention.
        q_bn = q[:, None, :, :].expand(bsz, n_pages, q_len, self.hidden_dim)
        q_bn = q_bn.reshape(bsz * n_pages, q_len, self.hidden_dim)
        p_bn = p.reshape(bsz * n_pages, page_k, self.hidden_dim)

        q_mask_bn = (
            question_mask[:, None, :]
            .expand(bsz, n_pages, q_len)
            .reshape(bsz * n_pages, q_len)
        )

        attended, _ = self.cross_attn(
            self.cross_q_norm(q_bn),
            self.cross_p_norm(p_bn),
            self.cross_p_norm(p_bn),
            need_weights=False,
        )

        fused = torch.cat(
            [
                q_bn,
                attended,
                q_bn * attended,
                torch.abs(q_bn - attended),
            ],
            dim=-1,
        )
        token_evidence = self.token_fuse(fused)

        valid_q = q_mask_bn.unsqueeze(-1)
        safe = token_evidence.masked_fill(~valid_q, 0.0)
        denom = valid_q.sum(dim=1).clamp_min(1)
        mean = safe.sum(dim=1) / denom

        neg_inf = torch.finfo(token_evidence.dtype).min
        maxv = token_evidence.masked_fill(~valid_q, neg_inf).amax(dim=1)

        page_repr = torch.cat([mean, maxv], dim=-1)
        raw_logits = self.scorer(page_repr).reshape(bsz, n_pages)

        scaled_logits = raw_logits / self.temperature
        masked_logits = scaled_logits.masked_fill(
            ~page_mask,
            torch.finfo(scaled_logits.dtype).min,
        )

        choice_probs = torch.softmax(masked_logits, dim=-1)
        noul_probs = torch.sigmoid(scaled_logits).masked_fill(~page_mask, 0.0)
        top1 = choice_probs.argmax(dim=-1)

        if n_pages > 1:
            top2 = torch.topk(choice_probs, k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = choice_probs[:, 0]

        eps = 1e-12
        entropy = -(choice_probs.clamp_min(eps).log() * choice_probs).sum(dim=-1)
        valid_counts = page_mask.sum(dim=-1).clamp_min(1)
        max_entropy = valid_counts.float().log().clamp_min(eps)
        normalized_entropy = torch.where(
            valid_counts > 1,
            entropy / max_entropy,
            torch.zeros_like(entropy),
        )

        return DecisionOutputV2(
            logits=masked_logits,
            choice_probs=choice_probs,
            noul_probs=noul_probs,
            top1_index=top1,
            confidence_margin=margin,
            normalized_entropy=normalized_entropy,
            temperature=self.temperature,
        )

    def loss(
        self,
        output: DecisionOutputV2,
        positive_mask: torch.Tensor,
        page_mask: torch.Tensor,
        choice_weight: float = 1.0,
        noul_weight: float = 1.0,
        brier_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        page_mask = page_mask.bool()
        positive_mask = positive_mask.bool() & page_mask

        positives_per_sample = positive_mask.sum(dim=-1)
        valid_choice = positives_per_sample > 0

        log_choice = F.log_softmax(output.logits, dim=-1)
        target = positive_mask.float()
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
        choice_per_sample = -(target * log_choice).sum(dim=-1)
        if valid_choice.any():
            choice_loss = choice_per_sample[valid_choice].mean()
        else:
            choice_loss = output.logits.sum() * 0.0

        binary_targets = positive_mask.float()
        valid = page_mask
        noul_loss = F.binary_cross_entropy_with_logits(
            output.logits[valid],
            binary_targets[valid],
        )
        brier_loss = (
            (output.noul_probs[valid] - binary_targets[valid]) ** 2
        ).mean()

        total = (
            choice_weight * choice_loss
            + noul_weight * noul_loss
            + brier_weight * brier_loss
        )
        return {
            "loss": total,
            "choice_loss": choice_loss.detach(),
            "noul_loss": noul_loss.detach(),
            "brier_loss": brier_loss.detach(),
        }

    def config_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_p,
        }


def pad_v2_batch(
    samples: Iterable[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = list(samples)
    if not samples:
        raise ValueError("Cannot collate empty batch")

    max_q = max(int(s["question_tokens"].shape[0]) for s in samples)
    max_pages = max(int(s["page_tokens"].shape[0]) for s in samples)
    page_k = int(samples[0]["page_tokens"].shape[1])
    dim = int(samples[0]["page_tokens"].shape[2])

    q = torch.zeros(len(samples), max_q, dim, dtype=torch.float32)
    q_mask = torch.zeros(len(samples), max_q, dtype=torch.bool)
    pages = torch.zeros(
        len(samples), max_pages, page_k, dim, dtype=torch.float32
    )
    page_mask = torch.zeros(len(samples), max_pages, dtype=torch.bool)
    positive = torch.zeros(len(samples), max_pages, dtype=torch.bool)

    for i, sample in enumerate(samples):
        qt = sample["question_tokens"].float()
        pt = sample["page_tokens"].float()
        t = qt.shape[0]
        n = pt.shape[0]

        q[i, :t] = qt
        q_mask[i, :t] = True
        pages[i, :n] = pt
        page_mask[i, :n] = True

        for page_num in sample.get("gt_pages", []):
            idx = int(page_num) - 1
            if 0 <= idx < n:
                positive[i, idx] = True

    return q, q_mask, pages, page_mask, positive
