"""Prompt template compatibility exports for dataset classes.

Dataset classes only need the original handcrafted templates during release
evaluation. LLM-generated descriptions are managed by ``utils.llm_prompts`` and
stored under ``text_templates/llm_supplement``.
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDCRAFTED_DIR = PROJECT_ROOT / "text_templates" / "handcrafted"


def _load_template_list(filename: str) -> list[str]:
    path = HANDCRAFTED_DIR / filename
    with path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    templates = payload.get("templates") if isinstance(payload, dict) else payload
    if not isinstance(templates, list) or not all(isinstance(item, str) for item in templates):
        raise ValueError(f"Invalid handcrafted template file: {path}")
    return list(templates)


text_prompts = _load_template_list("original_handcrafted.json")
text_prompts_pc2_view = _load_template_list("pointcloud_depth_view.json")
