"""Local Jev-style System-1 decision model for LensVLM.

This module intentionally does not call the TypeSafe/Jev API.  It implements a
small, trainable decision layer inspired by the public System-1/typed-decision
idea:

* no autoregressive chain-of-thought generation for routing
* variable number of candidates (document pages)
* Choice-like mutually-exclusive probabilities via softmax
* Noul-like independent relevance probabilities via sigmoid
* a shared scorer across all pages
* calibrated-ish probabilities through a learnable temperature and Brier loss

The LensVLM/Qwen3.5 backbone is used only as a frozen feature extractor:
- image features come from Qwen3.5 get_image_features(), before the 9B language
  model performs autoregressive reasoning
- question features use the frozen token embedding table, not a full 9B text
  forward pass

That makes this module suitable for a fast local System-1 router while keeping
the original LensVLM as the slower System-2 reader/reasoner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecisionOutput:
    """Outputs from the local System-1 decision head."""

    logits: torch.Tensor
    choice_probs: torch.Tensor
    noul_probs: torch.Tensor
    top1_index: torch.Tensor
    confidence_margin: torch.Tensor
    normalized_entropy: torch.Tensor
    temperature: torch.Tensor


def _inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=-1)


def _stats_pool(tokens: torch.Tensor) -> torch.Tensor:
    """Cheap order-agnostic pooling: [mean || max].

    tokens: [..., T, D]
    returns: [..., 2D]
    """
    mean = tokens.mean(dim=-2)
    maxv = tokens.amax(dim=-2)
    return torch.cat([mean, maxv], dim=-1)


class JevStyleDecisionHead(nn.Module):
    """Variable-candidate page router.

    The head consumes one question feature and N page features, where N may
    differ between samples.  It uses a small set-attention block so candidates
    can be judged relative to one another, then applies one shared scorer to
    every page.

    The same scalar score is exposed in two ways:
    - Choice: softmax over pages (which page should be expanded first?)
    - Noul: sigmoid per page (does this page contain useful evidence?)
    """

    def __init__(
        self,
        question_dim: int,
        page_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.question_dim = question_dim
        self.page_dim = page_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_p = dropout

        self.question_proj = nn.Sequential(
            nn.LayerNorm(question_dim),
            nn.Linear(question_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.page_proj = nn.Sequential(
            nn.LayerNorm(page_dim),
            nn.Linear(page_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.set_norm1 = nn.LayerNorm(hidden_dim)
        self.set_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.set_norm2 = nn.LayerNorm(hidden_dim)
        self.set_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # q, p, q*p, |q-p| -> shared score.
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
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
        question_features: torch.Tensor,
        page_features: torch.Tensor,
        page_mask: torch.Tensor | None = None,
    ) -> DecisionOutput:
        """Score pages in parallel.

        Args:
            question_features: [B, Dq]
            page_features: [B, N, Dp]
            page_mask: [B, N], True for real pages and False for padding.
        """
        if page_mask is None:
            page_mask = torch.ones(
                page_features.shape[:2],
                dtype=torch.bool,
                device=page_features.device,
            )
        else:
            page_mask = page_mask.bool()

        q = self.question_proj(question_features)  # [B, H]
        p = self.page_proj(page_features)  # [B, N, H]

        # Candidate interaction. Padding pages are ignored as keys/values.
        p_norm = self.set_norm1(p)
        attn_out, _ = self.set_attn(
            p_norm,
            p_norm,
            p_norm,
            key_padding_mask=~page_mask,
            need_weights=False,
        )
        p = p + attn_out
        p = p + self.set_ff(self.set_norm2(p))
        p = p.masked_fill(~page_mask.unsqueeze(-1), 0.0)

        q_expanded = q.unsqueeze(1).expand_as(p)
        fused = torch.cat(
            [
                q_expanded,
                p,
                q_expanded * p,
                torch.abs(q_expanded - p),
            ],
            dim=-1,
        )

        raw_logits = self.scorer(fused).squeeze(-1)
        scaled_logits = raw_logits / self.temperature
        masked_logits = scaled_logits.masked_fill(
            ~page_mask, torch.finfo(scaled_logits.dtype).min
        )

        choice_probs = torch.softmax(masked_logits, dim=-1)
        noul_probs = torch.sigmoid(scaled_logits).masked_fill(~page_mask, 0.0)

        top1 = choice_probs.argmax(dim=-1)
        if choice_probs.shape[-1] > 1:
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

        return DecisionOutput(
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
        output: DecisionOutput,
        positive_mask: torch.Tensor,
        page_mask: torch.Tensor,
        choice_weight: float = 1.0,
        noul_weight: float = 1.0,
        brier_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Hybrid Choice + Noul + calibration loss.

        positive_mask may contain more than one evidence page.  Choice loss uses
        a uniform distribution over all positive pages, while Noul/Brier treat
        page relevance as a multi-label problem.
        """
        page_mask = page_mask.bool()
        positive_mask = positive_mask.bool() & page_mask

        # Choice loss: uniform target mass across all GT evidence pages.
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

        # Noul-like independent evidence decisions.
        flat_valid = page_mask
        binary_targets = positive_mask.float()
        noul_logits = output.logits
        noul_loss = F.binary_cross_entropy_with_logits(
            noul_logits[flat_valid],
            binary_targets[flat_valid],
        )

        brier_loss = (
            (output.noul_probs[flat_valid] - binary_targets[flat_valid]) ** 2
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
            "question_dim": self.question_dim,
            "page_dim": self.page_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_p,
        }


def _feature_model(backbone):
    """Return the object exposing get_image_features()."""
    if hasattr(backbone, "get_image_features"):
        return backbone
    if hasattr(backbone, "model") and hasattr(backbone.model, "get_image_features"):
        return backbone.model
    raise AttributeError("Backbone does not expose get_image_features()")


def _input_embedding_layer(backbone):
    if hasattr(backbone, "get_input_embeddings"):
        layer = backbone.get_input_embeddings()
        if layer is not None:
            return layer
    if hasattr(backbone, "model") and hasattr(backbone.model, "get_input_embeddings"):
        return backbone.model.get_input_embeddings()
    raise AttributeError("Backbone does not expose input embeddings")


@torch.inference_mode()
def extract_question_feature(
    backbone,
    processor,
    question: str,
    device: str | torch.device,
    max_tokens: int = 256,
) -> torch.Tensor:
    """Extract a cheap frozen question feature without running the 9B decoder.

    We reuse LensVLM's own token embedding table and stats-pool the question
    tokens.  This keeps System-1 local and cheap while retaining the backbone's
    lexical embedding space.
    """
    encoded = processor.tokenizer(
        question,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device).bool()

    embedding_layer = _input_embedding_layer(backbone)
    token_embeds = embedding_layer(input_ids)  # [1, T, D]

    # Mask only for completeness; tokenizer normally returns no padding here.
    valid = attention_mask.unsqueeze(-1)
    safe = token_embeds.masked_fill(~valid, 0.0)
    denom = valid.sum(dim=-2).clamp_min(1)
    mean = safe.sum(dim=-2) / denom

    neg_inf = torch.finfo(token_embeds.dtype).min
    maxv = token_embeds.masked_fill(~valid, neg_inf).amax(dim=-2)
    feature = torch.cat([mean, maxv], dim=-1)

    return feature.squeeze(0).float().cpu()


@torch.inference_mode()
def extract_page_features(
    backbone,
    processor,
    images: Sequence,
    device: str | torch.device,
) -> torch.Tensor:
    """Extract one frozen feature per compressed page using the vision tower.

    Qwen3.5's get_image_features() returns a tuple/list of merged visual-token
    tensors, one tensor per input image.  We stats-pool each page independently.
    The 9B language model is not run.
    """
    if not images:
        raise ValueError("images must contain at least one page")

    image_inputs = processor.image_processor(
        images=list(images),
        return_tensors="pt",
    )

    pixel_values = image_inputs["pixel_values"].to(device)
    image_grid_thw = image_inputs["image_grid_thw"].to(device)

    model = _feature_model(backbone)
    vision_output = model.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        return_dict=True,
    )

    pooled = vision_output.pooler_output

    if isinstance(pooled, torch.Tensor):
        # Current Qwen3.5 returns split tensors, but keep a fallback for future
        # Transformers versions.
        if pooled.ndim == 3 and pooled.shape[0] == len(images):
            per_image = list(pooled)
        elif pooled.ndim == 2:
            visual = getattr(model, "visual", None)
            merge = getattr(visual, "spatial_merge_size", 1)
            split_sizes = (image_grid_thw.prod(-1) // (merge**2)).tolist()
            per_image = list(torch.split(pooled, split_sizes))
        else:
            raise RuntimeError(
                f"Unexpected pooler_output shape: {tuple(pooled.shape)}"
            )
    else:
        per_image = list(pooled)

    if len(per_image) != len(images):
        raise RuntimeError(
            f"Expected {len(images)} image feature groups, got {len(per_image)}"
        )

    page_features = torch.stack(
        [_stats_pool(tokens.float()) for tokens in per_image],
        dim=0,
    )

    return page_features.cpu()


def pad_feature_batch(
    samples: Iterable[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad cached variable-page samples into tensors for training.

    Each input sample must contain:
      question_feature: [Dq]
      page_features: [N, Dp]
      gt_pages: 1-indexed evidence page numbers
    """
    samples = list(samples)
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    questions = torch.stack([s["question_feature"].float() for s in samples])
    max_pages = max(s["page_features"].shape[0] for s in samples)
    page_dim = samples[0]["page_features"].shape[-1]

    pages = torch.zeros(len(samples), max_pages, page_dim, dtype=torch.float32)
    mask = torch.zeros(len(samples), max_pages, dtype=torch.bool)
    positive = torch.zeros(len(samples), max_pages, dtype=torch.bool)

    for i, sample in enumerate(samples):
        feats = sample["page_features"].float()
        n = feats.shape[0]
        pages[i, :n] = feats
        mask[i, :n] = True
        for page_num in sample.get("gt_pages", []):
            idx = int(page_num) - 1
            if 0 <= idx < n:
                positive[i, idx] = True

    return questions, pages, mask, positive
