"""Prompt construction and text-prototype helpers for DPC-Point."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def clean_class_name(classname: str) -> str:
    return str(classname).replace("_", " ")


def lookup_class_prompts(classname: str, prompt_dict: dict) -> list[str]:
    raw_name = str(classname)
    clean_name = clean_class_name(raw_name)
    candidate_keys = [
        raw_name,
        clean_name,
        raw_name.lower(),
        clean_name.lower(),
        raw_name.replace("_", " ").lower(),
    ]
    for key in candidate_keys:
        if key in prompt_dict:
            return prompt_dict[key]
    raise KeyError(f"No prompts found for class '{classname}'.")


def build_prompt_texts(classname: str, template) -> list[str]:
    clean_name = clean_class_name(classname)
    if isinstance(template, list):
        return [text.format(clean_name) for text in template]
    if isinstance(template, tuple):
        return [text.format(clean_name) for text in template]
    if isinstance(template, dict):
        if is_weighted_prompt_fusion(template):
            raise TypeError("Weighted prompt fusion must be encoded via weighted fusion helpers.")
        return lookup_class_prompts(classname, template)
    raise TypeError(f"Unsupported prompt template type: {type(template)}")


def is_weighted_prompt_fusion(template) -> bool:
    return isinstance(template, dict) and template.get("__dpc_point_prompt_type__") == "weighted_fusion"


def make_weighted_prompt_fusion(static_template, dynamic_template, static_weight: float = 0.75, dynamic_weight: float = 0.25) -> dict:
    return {
        "__dpc_point_prompt_type__": "weighted_fusion",
        "static_template": static_template,
        "dynamic_template": dynamic_template,
        "static_weight": float(static_weight),
        "dynamic_weight": float(dynamic_weight),
    }


def tokenize_texts(tokenize, texts: list[str], device: torch.device) -> torch.Tensor:
    return tokenize(texts).to(device)


@torch.no_grad()
def encode_texts(text_encoder, tokenize, texts: list[str], device: torch.device) -> torch.Tensor:
    tokens = tokenize_texts(tokenize, texts, device)
    if hasattr(text_encoder, "encode_text"):
        features = text_encoder.encode_text(tokens)
    else:
        features = text_encoder(tokens)
    return F.normalize(features.float(), dim=-1)


@torch.no_grad()
def encode_text_prototype(text_encoder, tokenize, texts: list[str], device: torch.device) -> torch.Tensor:
    features = encode_texts(text_encoder, tokenize, texts, device)
    return F.normalize(features.mean(dim=0), dim=0)


@torch.no_grad()
def encode_weighted_prompt_fusion(text_encoder, tokenize, classname: str, template: dict, device: torch.device) -> torch.Tensor:
    static_texts = build_prompt_texts(classname, template["static_template"])
    dynamic_texts = build_prompt_texts(classname, template["dynamic_template"])
    static_embedding = encode_text_prototype(text_encoder, tokenize, static_texts, device)
    dynamic_embedding = encode_text_prototype(text_encoder, tokenize, dynamic_texts, device)
    prototype = (
        float(template.get("static_weight", 0.75)) * static_embedding
        + float(template.get("dynamic_weight", 0.25)) * dynamic_embedding
    )
    return F.normalize(prototype, dim=0)


@torch.no_grad()
def build_text_classifier(text_encoder, tokenize, classnames: list[str], template, device: torch.device) -> torch.Tensor:
    weights = []
    for classname in classnames:
        if is_weighted_prompt_fusion(template):
            prototype = encode_weighted_prompt_fusion(text_encoder, tokenize, classname, template, device)
        else:
            texts = build_prompt_texts(classname, template)
            prototype = encode_text_prototype(text_encoder, tokenize, texts, device)
        weights.append(prototype)
    return torch.stack(weights, dim=1).to(device)
