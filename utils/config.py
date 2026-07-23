"""Release configuration helpers for DPC-Point."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional for non-LLM runs.
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CacheConfig:
    entropy_cap: int = 3
    gpa_cap: int = 3
    local_cap: int = 3
    neg_cap: int = 6
    local_centers: int = 3


@dataclass(frozen=True)
class DistributionConfig:
    dist_eps: float = 1e-4
    dist_min_var: float = 1e-4
    text_dist_eps: float = 1e-4
    text_dist_min_var: float = 1e-4
    text_score_weight: float = 0.15
    score_norm_mode: str = "running_zscore"
    score_norm_min_count: int = 8
    score_norm_eps: float = 1e-6
    score_norm_clip: float = 0.0


@dataclass(frozen=True)
class PromptConfig:
    prompt_source: str = "handcrafted_with_llm"
    prompt_static_weight: float = 0.75
    prompt_dynamic_weight: float = 0.25
    dynamic_prompt_count: int = 10
    prompt_cache_dir: str = "text_templates/llm_supplement"
    prompt_cache_file: str = ""
    llm_prompt_mode: str = "multiview_2d3d"
    force_regenerate_prompts: bool = False
    llm_max_retries: int = 3


DEFAULT_CACHE_CONFIG = CacheConfig()
DEFAULT_DISTRIBUTION_CONFIG = DistributionConfig()
DEFAULT_PROMPT_CONFIG = PromptConfig()
DEFAULT_FINAL_SCORE_WEIGHTS = [{"name": "best", "alpha_g": 4.4, "alpha_l": 3.9, "alpha_n": 0.19}]
FINAL_SCORE_FORMULA = "y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n"


def load_project_env(project_root: Path = PROJECT_ROOT) -> None:
    """Load .env from the project root when python-dotenv is available."""
    if load_dotenv is not None:
        load_dotenv(project_root / ".env")


def llm_env_config(project_root: Path = PROJECT_ROOT) -> dict:
    """Return OpenAI-compatible LLM settings from .env without the old LLM_ prefix."""
    load_project_env(project_root)
    return {
        "api_key": os.environ.get("API_KEY", ""),
        "base_url": os.environ.get("BASE_URL", ""),
        "model": os.environ.get("MODEL", ""),
        "provider": os.environ.get("PROVIDER", ""),
        "temperature": float(os.environ.get("TEMPERATURE", "0.7") or 0.7),
    }


def _float_token(value: float) -> str:
    value = float(value)
    if value.is_integer():
        token = f"{value:.1f}"
    else:
        token = f"{value:g}"
    return token.replace("-", "m").replace(".", "p")


def final_score_weight_name(alpha_g: float, alpha_l: float, alpha_n: float) -> str:
    return f"ag{_float_token(alpha_g)}_al{_float_token(alpha_l)}_an{_float_token(alpha_n)}"


def parse_final_score_weights(raw: str | Iterable[dict] | None) -> list[dict]:
    if raw is None or raw == "":
        return [dict(item) for item in DEFAULT_FINAL_SCORE_WEIGHTS]
    if isinstance(raw, (list, tuple)):
        return [dict(item) for item in raw]

    weights = []
    used_names = set()
    for index, item in enumerate(str(raw).replace("\n", ";").split(";"), start=1):
        item = item.strip()
        if not item:
            continue
        item = item.replace("，", ",").replace("(", "").replace(")", "")
        explicit_name = None
        if ":" in item:
            maybe_name, item = item.split(":", 1)
            maybe_name = maybe_name.strip()
            if maybe_name:
                explicit_name = maybe_name
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid final-score weight item: {item!r}; expected alpha_g,alpha_l,alpha_n.")
        alpha_g, alpha_l, alpha_n = [float(part) for part in parts]
        name = explicit_name or final_score_weight_name(alpha_g, alpha_l, alpha_n)
        if name in used_names:
            name = f"{name}_{index}"
        used_names.add(name)
        weights.append({"name": name, "alpha_g": alpha_g, "alpha_l": alpha_l, "alpha_n": alpha_n})

    if not weights:
        raise ValueError("No valid final-score weights were parsed.")
    return weights


def positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def non_negative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def release_config_payload(args, tasks) -> dict:
    task_payload = [task.to_config() if hasattr(task, "to_config") else dict(task) for task in tasks]
    return {
        "method": "DPC-Point",
        "backbone": getattr(args, "backbone"),
        "runtime": {
            "device": getattr(args, "device", ""),
            "dtype": getattr(args, "dtype", ""),
            "seed": int(getattr(args, "seed", 1)),
            "npoints": int(getattr(args, "npoints", 1024)),
            "num_workers": int(getattr(args, "num_workers", 2)),
            "print_freq": int(getattr(args, "print_freq", 500)),
        },
        "output": {
            "directory": getattr(args, "output_dir", "results/dpc_point"),
            "experiment_name": getattr(args, "resolved_exp_name", getattr(args, "exp_name", "")),
            "files": ["run.log", "summary.csv", "config.json"],
        },
        "backbone_config": {
            "cache_type": getattr(args, "cache_type", "hierarchical"),
            "ulip_text_ckpt": getattr(args, "ulip_text_ckpt", ""),
            "ulip_point_ckpt": getattr(args, "ulip_point_ckpt", ""),
            "openshape_config": getattr(args, "openshape_config", ""),
            "openshape_clip_model": getattr(args, "openshape_clip_model", ""),
            "openshape_text_ckpt": getattr(args, "openshape_text_ckpt", ""),
            "openshape_point_ckpt": getattr(args, "openshape_point_ckpt", ""),
            "uni3d_clip_model": getattr(args, "uni3d_clip_model", ""),
            "uni3d_text_ckpt": getattr(args, "uni3d_text_ckpt", ""),
            "uni3d_point_ckpt": getattr(args, "uni3d_point_ckpt", ""),
            "uni3d_modelnet_ckpt": getattr(args, "uni3d_modelnet_ckpt", ""),
            "uni3d_scanobjectnn_ckpt": getattr(args, "uni3d_scanobjectnn_ckpt", ""),
            "uni3d_general_ckpt": getattr(args, "uni3d_general_ckpt", ""),
            "uni3d_pc_model": getattr(args, "uni3d_pc_model", ""),
            "uni3d_pc_feat_dim": int(getattr(args, "uni3d_pc_feat_dim", 0) or 0),
            "uni3d_pc_encoder_dim": int(getattr(args, "uni3d_pc_encoder_dim", 0) or 0),
            "uni3d_embed_dim": int(getattr(args, "uni3d_embed_dim", 0) or 0),
            "uni3d_group_size": int(getattr(args, "uni3d_group_size", 0) or 0),
            "uni3d_num_group": int(getattr(args, "uni3d_num_group", 0) or 0),
        },
        "dataset": {
            "name": getattr(args, "dataset"),
            "severity_set": getattr(args, "severity_set", "s2"),
            "sonn_variant": getattr(args, "sonn_variant", "hardest"),
            "modelnet_c_root": getattr(args, "modelnet_c_root", "data/modelnet_c"),
            "scanobjectnn_c_root": getattr(args, "scanobjectnn_c_root", "data/sonn_c"),
            "corruptions": getattr(args, "corruptions", "all"),
        },
        "prompt": {
            "source": getattr(args, "prompt_source", DEFAULT_PROMPT_CONFIG.prompt_source),
            "static_weight": float(getattr(args, "prompt_static_weight", DEFAULT_PROMPT_CONFIG.prompt_static_weight)),
            "dynamic_weight": float(getattr(args, "prompt_dynamic_weight", DEFAULT_PROMPT_CONFIG.prompt_dynamic_weight)),
            "dynamic_prompt_count": int(getattr(args, "dynamic_prompt_count", DEFAULT_PROMPT_CONFIG.dynamic_prompt_count)),
            "cache_dir": getattr(args, "prompt_cache_dir", DEFAULT_PROMPT_CONFIG.prompt_cache_dir),
            "cache_file": getattr(args, "prompt_cache_file", DEFAULT_PROMPT_CONFIG.prompt_cache_file),
        },
        "distribution": {
            "dist_eps": float(getattr(args, "dist_eps", DEFAULT_DISTRIBUTION_CONFIG.dist_eps)),
            "dist_min_var": float(getattr(args, "dist_min_var", DEFAULT_DISTRIBUTION_CONFIG.dist_min_var)),
            "text_dist_eps": float(getattr(args, "text_dist_eps", DEFAULT_DISTRIBUTION_CONFIG.text_dist_eps)),
            "text_dist_min_var": float(getattr(args, "text_dist_min_var", DEFAULT_DISTRIBUTION_CONFIG.text_dist_min_var)),
            "text_score_weight": float(getattr(args, "text_score_weight", DEFAULT_DISTRIBUTION_CONFIG.text_score_weight)),
            "score_norm_mode": getattr(args, "score_norm_mode", DEFAULT_DISTRIBUTION_CONFIG.score_norm_mode),
            "score_norm_min_count": int(getattr(args, "score_norm_min_count", DEFAULT_DISTRIBUTION_CONFIG.score_norm_min_count)),
            "score_norm_eps": float(getattr(args, "score_norm_eps", DEFAULT_DISTRIBUTION_CONFIG.score_norm_eps)),
            "score_norm_clip": float(getattr(args, "score_norm_clip", DEFAULT_DISTRIBUTION_CONFIG.score_norm_clip)),
        },
        "cache": {
            "entropy_cap": int(getattr(args, "entropy_cap", DEFAULT_CACHE_CONFIG.entropy_cap)),
            "gpa_cap": int(getattr(args, "gpa_cap", DEFAULT_CACHE_CONFIG.gpa_cap)),
            "local_cap": int(getattr(args, "local_cap", DEFAULT_CACHE_CONFIG.local_cap)),
            "neg_cap": int(getattr(args, "neg_cap", DEFAULT_CACHE_CONFIG.neg_cap)),
            "local_centers": int(getattr(args, "local_centers", DEFAULT_CACHE_CONFIG.local_centers)),
        },
        "final_score": {
            "formula": FINAL_SCORE_FORMULA,
            "weights": parse_final_score_weights(getattr(args, "final_score_weights", DEFAULT_FINAL_SCORE_WEIGHTS)),
        },
        "tasks": task_payload,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def dataclass_dict(value) -> dict:
    return asdict(value)
