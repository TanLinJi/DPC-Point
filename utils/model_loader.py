"""Backbone loading and point-logit helpers for DPC-Point."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import clip
import open_clip
from models import openshape, ulip, uni3d


BACKBONE_ALIASES = {
    "ulip": "ulip",
    "openshape": "openshape",
    "open_shape": "openshape",
    "uni3d": "uni3d",
}


def canonical_backbone(name: str) -> str:
    key = str(name).strip().lower()
    if key not in BACKBONE_ALIASES:
        raise ValueError(f"Unsupported backbone: {name}")
    return BACKBONE_ALIASES[key]


def project_path(project_root: Path, value: str | os.PathLike) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def torch_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    if requested == "float16":
        return torch.float16
    if requested == "float32":
        return torch.float32
    return torch.float16 if device.type == "cuda" else torch.float32


def strip_prefix(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    if not next(iter(state_dict)).startswith(prefix):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def checkpoint_state_dict(checkpoint, preferred_keys: Iterable[str] = ("state_dict", "module", "model")) -> dict:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in preferred_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def prepare_model(model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    model.eval()
    if dtype == torch.float16:
        model = model.half()
    return model.to(device)


def load_ulip_backbone(args, project_root: Path, device: torch.device, dtype: torch.dtype):
    model_args = SimpleNamespace(
        cache_type=getattr(args, "cache_type", "global"),
        n_cluster=args.n_cluster,
        pc_feat_dim=768,
        embed_dim=512,
        group_size=32,
        num_group=512,
        encoder_dim=256,
        pc_depth=12,
        num_head=6,
        drop_path_rate=0.1,
    )

    text_encoder = ulip.create_clip_text_encoder(model_args)
    slip_ckpt = load_checkpoint(project_path(project_root, args.ulip_text_ckpt))
    slip_sd = strip_prefix(checkpoint_state_dict(slip_ckpt), "module.")
    text_sd = {
        key: value for key, value in slip_sd.items()
        if key.startswith(("positional_embedding", "text_projection", "logit_scale", "transformer", "token_embedding", "ln_final"))
    }
    text_encoder.load_state_dict(text_sd, strict=True)
    text_encoder = prepare_model(text_encoder, device, dtype)

    point_encoder = ulip.create_ulip(model_args)
    point_ckpt = load_checkpoint(project_path(project_root, args.ulip_point_ckpt))
    point_sd = strip_prefix(checkpoint_state_dict(point_ckpt), "module.")
    point_sd = {key: value for key, value in point_sd.items() if key.startswith(("pc_projection", "point_encoder"))}
    point_encoder.load_state_dict(point_sd, strict=True)
    point_encoder = prepare_model(point_encoder, device, dtype)

    return text_encoder, point_encoder, clip.tokenize


def load_openshape_backbone(args, project_root: Path, device: torch.device, dtype: torch.dtype):
    text_ckpt = str(project_path(project_root, args.openshape_text_ckpt))
    text_encoder, _, _ = open_clip.create_model_and_transforms(args.openshape_clip_model, pretrained=text_ckpt, device="cpu")
    text_encoder = prepare_model(text_encoder, device, dtype)

    cfg = OmegaConf.load(project_path(project_root, args.openshape_config))
    cfg = OmegaConf.merge(cfg, {"cache_type": getattr(args, "cache_type", "global"), "n_cluster": args.n_cluster})
    OmegaConf.resolve(cfg)

    point_encoder = openshape.create_openshape(cfg)
    point_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(point_encoder)

    checkpoint = load_checkpoint(project_path(project_root, args.openshape_point_ckpt))
    point_sd = strip_prefix(checkpoint_state_dict(checkpoint), "module.")
    point_encoder.load_state_dict(point_sd, strict=True)
    point_encoder = prepare_model(point_encoder, device, dtype)

    return text_encoder, point_encoder, open_clip.tokenize


def choose_uni3d_point_ckpt(args, project_root: Path, dataset_name: str) -> Path:
    if args.uni3d_point_ckpt:
        return project_path(project_root, args.uni3d_point_ckpt)
    if dataset_name in {"modelnet", "modelnet_c"}:
        return project_path(project_root, args.uni3d_modelnet_ckpt)
    if dataset_name in {"scanobjectnn", "scanobjectnn_c"}:
        return project_path(project_root, args.uni3d_scanobjectnn_ckpt)
    return project_path(project_root, args.uni3d_general_ckpt)


def move_open_clip_text(model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    model.eval()
    if hasattr(model, "text"):
        if dtype == torch.float16:
            model.text = model.text.half()
        model.text = model.text.to(device)
        return model
    return prepare_model(model, device, dtype)


def load_uni3d_backbone(args, project_root: Path, device: torch.device, dtype: torch.dtype, dataset_name: str):
    text_ckpt = str(project_path(project_root, args.uni3d_text_ckpt))
    text_encoder, _, _ = open_clip.create_model_and_transforms(args.uni3d_clip_model, pretrained=text_ckpt, device="cpu")
    text_encoder = move_open_clip_text(text_encoder, device, dtype)

    model_args = SimpleNamespace(
        cache_type=getattr(args, "cache_type", "global"),
        n_cluster=args.n_cluster,
        pc_model=args.uni3d_pc_model,
        pretrained_pc=args.uni3d_pretrained_pc,
        drop_path_rate=args.drop_path_rate,
        pc_feat_dim=args.uni3d_pc_feat_dim,
        group_size=args.uni3d_group_size,
        num_group=args.uni3d_num_group,
        pc_encoder_dim=args.uni3d_pc_encoder_dim,
        embed_dim=args.uni3d_embed_dim,
        patch_dropout=args.patch_dropout,
    )
    point_encoder = uni3d.create_uni3d(model_args)
    checkpoint = load_checkpoint(choose_uni3d_point_ckpt(args, project_root, dataset_name))
    point_sd = strip_prefix(checkpoint_state_dict(checkpoint, preferred_keys=("module", "state_dict", "model")), "module.")
    point_encoder.load_state_dict(point_sd, strict=True)
    point_encoder = prepare_model(point_encoder, device, dtype)

    return text_encoder, point_encoder, open_clip.tokenize


def load_backbone(args, project_root: Path, device: torch.device, dtype: torch.dtype, dataset_name: str):
    backbone = canonical_backbone(args.backbone)
    if backbone == "ulip":
        return load_ulip_backbone(args, project_root, device, dtype)
    if backbone == "openshape":
        return load_openshape_backbone(args, project_root, device, dtype)
    if backbone == "uni3d":
        return load_uni3d_backbone(args, project_root, device, dtype, dataset_name)
    raise ValueError(f"Unsupported backbone: {args.backbone}")


@torch.no_grad()
def point_features(backbone: str, point_encoder, xyz: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    backbone = canonical_backbone(backbone)
    feature = torch.cat([xyz, rgb], dim=-1)
    if backbone == "ulip":
        pc_features = point_encoder(feature[:, :, :3])
    elif backbone == "openshape":
        pc_features = point_encoder(feature[:, :, :3], feature)
    elif backbone == "uni3d":
        pc_features = point_encoder.encode_pc(feature)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return F.normalize(pc_features.float(), dim=-1)


@torch.no_grad()
def point_logits(backbone: str, point_encoder, xyz: torch.Tensor, rgb: torch.Tensor, text_weights: torch.Tensor) -> torch.Tensor:
    pc_features = point_features(backbone, point_encoder, xyz, rgb)
    return 100.0 * pc_features @ text_weights
