#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    item for item in sys.path
    if item and Path(item).resolve() != SCRIPT_DIR
]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from runners.dpc_point.cache_model import run_dpc_point  # noqa: E402
from runners.dpc_point.datasets import build_dataset, build_loader, build_task_specs, canonical_dataset  # noqa: E402
from runners.dpc_point.text_distribution import build_text_distribution  # noqa: E402
from utils.config import (  # noqa: E402
    DEFAULT_CACHE_CONFIG,
    DEFAULT_DISTRIBUTION_CONFIG,
    DEFAULT_FINAL_SCORE_WEIGHTS,
    DEFAULT_PROMPT_CONFIG,
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,
    parse_final_score_weights,
    release_config_payload,
    write_json,
)
from utils.llm_prompts import get_prompt_template  # noqa: E402
from utils.metrics import set_random_seed, tee_output  # noqa: E402
from utils.model_loader import canonical_backbone, load_backbone, project_path, torch_dtype  # noqa: E402
from utils.prompt_utils import build_text_classifier  # noqa: E402


SUMMARY_FIELDS = [
    "backbone",
    "dataset",
    "display_dataset",
    "variant",
    "corruption",
    "severity",
    "cor_type",
    "weight_name",
    "alpha_g",
    "alpha_l",
    "alpha_n",
    "num_samples",
    "accuracy",
]


def add_backbone_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ulip-text-ckpt", default="weights/ulip/slip_base_100ep.pt")
    parser.add_argument("--ulip-point-ckpt", default="weights/ulip/pointbert_ulip1.pt")
    parser.add_argument("--openshape-config", default="models/openshape/config.yaml")
    parser.add_argument("--openshape-clip-model", default="ViT-bigG-14")
    parser.add_argument("--openshape-text-ckpt", default="weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin")
    parser.add_argument("--openshape-point-ckpt", default="weights/openshape/openshape-pointbert-vitg14-rgb/model.pt")
    parser.add_argument("--uni3d-clip-model", default="EVA02-E-14-plus")
    parser.add_argument("--uni3d-text-ckpt", default="weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin")
    parser.add_argument("--uni3d-point-ckpt", default="")
    parser.add_argument("--uni3d-modelnet-ckpt", default="weights/uni3d/modelnet40/model.pt")
    parser.add_argument("--uni3d-scanobjectnn-ckpt", default="weights/uni3d/scanobjnn/model.pt")
    parser.add_argument("--uni3d-general-ckpt", default="weights/uni3d/model.pt")
    parser.add_argument("--uni3d-pc-model", default="eva_giant_patch14_560")
    parser.add_argument("--uni3d-pretrained-pc", default="")
    parser.add_argument("--uni3d-pc-feat-dim", type=int, default=1408)
    parser.add_argument("--uni3d-pc-encoder-dim", type=int, default=512)
    parser.add_argument("--uni3d-embed-dim", type=int, default=1024)
    parser.add_argument("--uni3d-group-size", type=int, default=64)
    parser.add_argument("--uni3d-num-group", type=int, default=512)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--patch-dropout", type=float, default=0.0)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run formal DPC-Point inference.")
    parser.add_argument("--backbone", required=True, choices=["ulip", "openshape", "uni3d"])
    parser.add_argument("--dataset", required=True, choices=["modelnet", "modelnet_c", "scanobjectnn", "scanobjectnn_c"])
    parser.add_argument("--severity-set", default="s2", help="clean, s2, all35, or comma-separated severity ids.")
    parser.add_argument("--corruptions", default="all", help="all or comma-separated corruption names.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--print-freq", type=int, default=500)
    parser.add_argument("--modelnet-c-root", default="data/modelnet_c")
    parser.add_argument("--scanobjectnn-c-root", default="data/sonn_c")
    parser.add_argument("--sonn-variant", default="hardest", choices=["hardest", "obj_bg", "obj_only"])
    parser.add_argument("--output-dir", default="results/dpc_point")
    parser.add_argument("--exp-name", default="")

    parser.add_argument("--prompt-source", default=DEFAULT_PROMPT_CONFIG.prompt_source, choices=["handcrafted", "llm_descriptions", "handcrafted_with_llm"])
    parser.add_argument("--prompt-static-weight", type=float, default=DEFAULT_PROMPT_CONFIG.prompt_static_weight)
    parser.add_argument("--prompt-dynamic-weight", type=float, default=DEFAULT_PROMPT_CONFIG.prompt_dynamic_weight)
    parser.add_argument("--dynamic-prompt-count", type=int, default=DEFAULT_PROMPT_CONFIG.dynamic_prompt_count)
    parser.add_argument("--prompt-cache-dir", default=DEFAULT_PROMPT_CONFIG.prompt_cache_dir)
    parser.add_argument("--prompt-cache-file", default=DEFAULT_PROMPT_CONFIG.prompt_cache_file)
    parser.add_argument("--llm-prompt-mode", default=DEFAULT_PROMPT_CONFIG.llm_prompt_mode)
    parser.add_argument("--force-regenerate-prompts", action="store_true", default=DEFAULT_PROMPT_CONFIG.force_regenerate_prompts)
    parser.add_argument("--llm-max-retries", type=int, default=DEFAULT_PROMPT_CONFIG.llm_max_retries)

    parser.add_argument("--entropy-cap", type=int, default=DEFAULT_CACHE_CONFIG.entropy_cap)
    parser.add_argument("--gpa-cap", type=int, default=DEFAULT_CACHE_CONFIG.gpa_cap)
    parser.add_argument("--local-cap", type=int, default=DEFAULT_CACHE_CONFIG.local_cap)
    parser.add_argument("--neg-cap", type=int, default=DEFAULT_CACHE_CONFIG.neg_cap)
    parser.add_argument("--local-centers", "--n-cluster", dest="local_centers", type=int, default=DEFAULT_CACHE_CONFIG.local_centers)
    parser.add_argument("--positive-beta", type=float, default=3.0)
    parser.add_argument("--negative-beta", type=float, default=1.0)
    parser.add_argument("--negative-entropy-lower", type=float, default=0.2)
    parser.add_argument("--negative-entropy-upper", type=float, default=0.5)
    parser.add_argument("--negative-mask-lower", type=float, default=0.03)
    parser.add_argument("--negative-mask-upper", type=float, default=1.0)

    parser.add_argument("--dist-eps", type=float, default=DEFAULT_DISTRIBUTION_CONFIG.dist_eps)
    parser.add_argument("--dist-min-var", type=float, default=DEFAULT_DISTRIBUTION_CONFIG.dist_min_var)
    parser.add_argument("--text-dist-eps", type=float, default=DEFAULT_DISTRIBUTION_CONFIG.text_dist_eps)
    parser.add_argument("--text-dist-min-var", type=float, default=DEFAULT_DISTRIBUTION_CONFIG.text_dist_min_var)
    parser.add_argument("--text-score-weight", type=float, default=DEFAULT_DISTRIBUTION_CONFIG.text_score_weight)
    parser.add_argument("--score-norm-mode", default=DEFAULT_DISTRIBUTION_CONFIG.score_norm_mode, choices=["none", "running_zscore"])
    parser.add_argument("--score-norm-min-count", type=int, default=DEFAULT_DISTRIBUTION_CONFIG.score_norm_min_count)
    parser.add_argument("--score-norm-eps", type=float, default=DEFAULT_DISTRIBUTION_CONFIG.score_norm_eps)
    parser.add_argument("--score-norm-clip", type=float, default=DEFAULT_DISTRIBUTION_CONFIG.score_norm_clip)
    parser.add_argument("--final-score-weights", default="best:4.4,3.9,0.19")

    add_backbone_args(parser)
    args = parser.parse_args(argv)
    args.dataset = canonical_dataset(args.dataset)
    args.backbone = canonical_backbone(args.backbone)
    args.cache_type = "hierarchical"
    args.n_cluster = args.local_centers
    args.final_score_weights = parse_final_score_weights(args.final_score_weights)
    return args


def default_exp_name(args) -> str:
    dataset = canonical_dataset(args.dataset)
    backbone = canonical_backbone(args.backbone)
    if dataset == "scanobjectnn_c":
        return f"{backbone}_{dataset}_{args.sonn_variant}_{args.severity_set}"
    if dataset == "scanobjectnn":
        return f"{backbone}_{dataset}_{args.sonn_variant}"
    if dataset == "modelnet_c":
        return f"{backbone}_{dataset}_{args.severity_set}"
    return f"{backbone}_{dataset}"


def output_paths(args, project_root: Path = PROJECT_ROOT):
    exp_name = getattr(args, "resolved_exp_name", "") or args.exp_name or default_exp_name(args)
    run_dir = project_path(project_root, args.output_dir) / exp_name
    return run_dir, run_dir / "run.log", run_dir / "summary.csv", run_dir / "config.json"


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["accuracy"] = f"{float(out['accuracy']):.4f}"
            writer.writerow(out)


def print_run_header(args, tasks, run_dir: Path) -> None:
    print("============================================================")
    print("DPC-Point formal inference")
    print(f"run_dir: {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"backbone: {args.backbone}")
    print(f"dataset: {args.dataset}")
    print(f"severity_set: {args.severity_set}")
    print(f"sonn_variant: {args.sonn_variant}")
    print(f"device: {args.device}")
    print(f"dtype: {args.dtype}")
    print(f"npoints: {args.npoints}")
    print(f"tasks: {len(tasks)}")
    for task in tasks:
        print(f"  - {task.display_dataset}: {task.cor_type} {task.variant}".rstrip())
    print("============================================================")


def result_rows_for_task(args, task, result: dict, num_samples: int) -> list[dict]:
    rows = []
    for item in result["weight_results"]:
        rows.append({
            "backbone": args.backbone,
            "dataset": task.dataset_key,
            "display_dataset": task.display_dataset,
            "variant": task.variant,
            "corruption": task.corruption,
            "severity": task.severity,
            "cor_type": task.cor_type,
            "weight_name": item["name"],
            "alpha_g": f"{float(item['alpha_g']):g}",
            "alpha_l": f"{float(item['alpha_l']):g}",
            "alpha_n": f"{float(item['alpha_n']):g}",
            "num_samples": int(num_samples),
            "accuracy": float(item["acc"]),
        })
    return rows


def run(args) -> list[dict]:
    if CONFIG_PROJECT_ROOT != PROJECT_ROOT:
        raise RuntimeError("Project root mismatch in configuration helpers.")
    set_random_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch_dtype(device, args.dtype)

    tasks = build_task_specs(
        args.dataset,
        severity_set=args.severity_set,
        modelnet_c_root=args.modelnet_c_root,
        scanobjectnn_c_root=args.scanobjectnn_c_root,
        sonn_variant=args.sonn_variant,
        corruptions=args.corruptions,
    )
    missing = [str(task.file_path) for task in tasks if not project_path(PROJECT_ROOT, task.file_path).exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))

    args.resolved_exp_name = args.exp_name or default_exp_name(args)
    run_dir, _log_path, summary_path, config_path = output_paths(args, PROJECT_ROOT)
    write_json(config_path, release_config_payload(args, tasks))
    print_run_header(args, tasks, run_dir)

    first_dataset = build_dataset(args, PROJECT_ROOT, tasks[0])
    text_encoder, point_encoder, tokenize = load_backbone(args, PROJECT_ROOT, device, dtype, args.dataset)
    prompt_template = get_prompt_template(args, first_dataset.classnames, args.dataset, PROJECT_ROOT)
    print(f"Building text prototypes: {len(first_dataset.classnames)} classes")
    text_weights = build_text_classifier(text_encoder, tokenize, first_dataset.classnames, prompt_template, device)
    print("Building text semantic distribution")
    text_dist = build_text_distribution(
        text_encoder,
        tokenize,
        first_dataset.classnames,
        prompt_template,
        device,
        min_var=args.text_dist_min_var,
    )

    rows = []
    for task in tasks:
        dataset = build_dataset(args, PROJECT_ROOT, task)
        if list(dataset.classnames) != list(first_dataset.classnames):
            raise RuntimeError("Class names changed across tasks; this runner expects one class space per experiment.")
        loader = build_loader(args, dataset)
        args.current_cor_type = task.cor_type
        print("------------------------------------------------------------")
        print(f"Running task: {task.display_dataset} {task.cor_type}")
        result = run_dpc_point(args, loader, point_encoder, text_weights, text_dist, device, dtype)
        print(f"Result: dataset={task.display_dataset}, corruption={task.cor_type}, OA={result['primary_acc']:.2f}")
        rows.extend(result_rows_for_task(args, task, result, len(dataset)))

    write_summary(summary_path, rows)
    print("============================================================")
    print(f"summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"config: {config_path.relative_to(PROJECT_ROOT)}")
    print("============================================================")
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.resolved_exp_name = args.exp_name or default_exp_name(args)
    run_dir, log_path, summary_path, config_path = output_paths(args, PROJECT_ROOT)
    run_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(log_path):
        rows = run(args)
        print(f"log: {log_path.relative_to(PROJECT_ROOT)}")
        print(f"summary rows: {len(rows)}")


if __name__ == "__main__":
    main()
