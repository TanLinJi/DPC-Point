"""Text semantic distribution construction for DPC-Point."""

from __future__ import annotations

import torch

from utils.prompt_utils import build_prompt_texts, encode_texts, is_weighted_prompt_fusion


@torch.no_grad()
def distribution_from_embeddings(embeddings: torch.Tensor, min_var: float) -> dict:
    mean = embeddings.mean(dim=0, keepdim=True)
    if embeddings.size(0) <= 1:
        var = torch.ones_like(mean) * float(min_var)
    else:
        var = embeddings.var(dim=0, unbiased=True, keepdim=True).clamp_min(float(min_var))
    return {"count": int(embeddings.size(0)), "mean": mean.detach(), "var": var.detach()}


@torch.no_grad()
def weighted_distribution_from_embeddings(embeddings: torch.Tensor, weights: torch.Tensor, min_var: float) -> dict:
    weights = weights.to(device=embeddings.device, dtype=embeddings.dtype).view(-1, 1)
    weights = weights / weights.sum().clamp_min(1e-12)
    mean = (embeddings * weights).sum(dim=0, keepdim=True)
    if embeddings.size(0) <= 1:
        var = torch.ones_like(mean) * float(min_var)
    else:
        var = ((embeddings - mean).pow(2) * weights).sum(dim=0, keepdim=True).clamp_min(float(min_var))
    return {"count": int(embeddings.size(0)), "mean": mean.detach(), "var": var.detach()}


@torch.no_grad()
def build_text_distribution(text_encoder, tokenize, classnames: list[str], template, device: torch.device, min_var: float = 1e-4) -> dict[int, dict]:
    text_dist = {}
    for class_index, classname in enumerate(classnames):
        if is_weighted_prompt_fusion(template):
            static_texts = build_prompt_texts(classname, template["static_template"])
            dynamic_texts = build_prompt_texts(classname, template["dynamic_template"])
            static_embeddings = encode_texts(text_encoder, tokenize, static_texts, device)
            dynamic_embeddings = encode_texts(text_encoder, tokenize, dynamic_texts, device)
            embeddings = torch.cat([static_embeddings, dynamic_embeddings], dim=0)
            static_weight = float(template.get("static_weight", 0.75)) / max(static_embeddings.size(0), 1)
            dynamic_weight = float(template.get("dynamic_weight", 0.25)) / max(dynamic_embeddings.size(0), 1)
            weights = torch.cat([
                torch.full((static_embeddings.size(0),), static_weight, device=embeddings.device, dtype=embeddings.dtype),
                torch.full((dynamic_embeddings.size(0),), dynamic_weight, device=embeddings.device, dtype=embeddings.dtype),
            ])
            text_dist[int(class_index)] = weighted_distribution_from_embeddings(embeddings, weights, min_var)
        else:
            texts = build_prompt_texts(classname, template)
            embeddings = encode_texts(text_encoder, tokenize, texts, device)
            text_dist[int(class_index)] = distribution_from_embeddings(embeddings, min_var)
    return text_dist
