#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from datasets.modelnet_c import ModelNet_C  # noqa: E402
from datasets.sonn_c import SONN_C  # noqa: E402
from utils.metrics import set_random_seed, summarize_rows, tee_output, write_summary  # noqa: E402
from utils.model_loader import canonical_backbone, load_backbone, point_logits, project_path, torch_dtype  # noqa: E402
from utils.prompt_utils import build_text_classifier  # noqa: E402


DATASET_ALIASES = {
    "modelnet": "modelnet",
    "modelnet_c": "modelnet_c",
    "modelnet-c": "modelnet_c",
    "scanobjectnn": "scanobjectnn",
    "scanobjectnn_c": "scanobjectnn_c",
    "scanobjectnn-c": "scanobjectnn_c",
}


@dataclass(frozen=True)
class EvalTask:
    dataset: str
    corruption: str = "clean"
    severity: str = "clean"
    variant: str = ""

    @property
    def cor_type(self) -> str:
        if self.corruption == "clean":
            return "clean"
        return f"{self.corruption}_{self.severity}"


def canonical_dataset(name: str) -> str:
    key = str(name).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_ALIASES[key]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_corruption_stem(stem: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<name>.+)_(?P<severity>\d+)$", stem)
    if not match:
        return None
    return match.group("name"), match.group("severity")


def list_corruption_tasks(dataset: str, root: Path, variant: str, corruptions: str, severities: str) -> list[EvalTask]:
    search_dir = root / variant if dataset == "scanobjectnn_c" else root
    if not search_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {search_dir}")

    selected_corruptions = None if corruptions == "all" else set(split_csv(corruptions))
    selected_severities = None if severities == "all" else set(split_csv(severities))

    tasks = []
    for file_path in sorted(search_dir.glob("*.h5")):
        parsed = parse_corruption_stem(file_path.stem)
        if parsed is None:
            continue
        corruption, severity = parsed
        if selected_corruptions is not None and corruption not in selected_corruptions and file_path.stem not in selected_corruptions:
            continue
        if selected_severities is not None and severity not in selected_severities:
            continue
        tasks.append(EvalTask(dataset=dataset, corruption=corruption, severity=severity, variant=variant if dataset == "scanobjectnn_c" else ""))

    if not tasks:
        raise FileNotFoundError(
            f"No corruption files matched dataset={dataset}, corruptions={corruptions}, severities={severities} in {search_dir}"
        )
    return tasks


def build_eval_tasks(args, project_root: Path) -> list[EvalTask]:
    dataset = canonical_dataset(args.dataset)
    if dataset == "modelnet":
        return [EvalTask(dataset=dataset)]
    if dataset == "scanobjectnn":
        return [EvalTask(dataset=dataset, variant=args.sonn_variant)]
    if dataset == "modelnet_c":
        return list_corruption_tasks(dataset, project_path(project_root, args.modelnet_c_root), "", args.corruptions, args.severities)
    if dataset == "scanobjectnn_c":
        return list_corruption_tasks(dataset, project_path(project_root, args.scanobjectnn_c_root), args.sonn_variant, args.corruptions, args.severities)
    raise ValueError(f"Unsupported dataset: {dataset}")


def dataset_config(args, project_root: Path, task: EvalTask) -> SimpleNamespace:
    return SimpleNamespace(
        lm3d=canonical_backbone(args.backbone),
        npoints=args.npoints,
        modelnet_c_root=str(project_path(project_root, args.modelnet_c_root)),
        sonn_c_root=str(project_path(project_root, args.scanobjectnn_c_root)),
        sonn_variant=task.variant or args.sonn_variant,
        cor_type=task.cor_type,
    )


def build_dataset(args, project_root: Path, task: EvalTask):
    cfg = dataset_config(args, project_root, task)
    if task.dataset in {"modelnet", "modelnet_c"}:
        return ModelNet_C(cfg)
    if task.dataset in {"scanobjectnn", "scanobjectnn_c"}:
        return SONN_C(cfg)
    raise ValueError(f"Unsupported dataset: {task.dataset}")


def build_loader(args, dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run DPC-Point zero-shot backbone evaluation.")
    parser.add_argument("--backbone", required=True, choices=["ulip", "openshape", "uni3d"])
    parser.add_argument("--dataset", required=True, choices=["modelnet", "modelnet_c", "scanobjectnn", "scanobjectnn_c"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=200)
    parser.add_argument("--modelnet-c-root", default="data/modelnet_c")
    parser.add_argument("--scanobjectnn-c-root", default="data/sonn_c")
    parser.add_argument("--sonn-variant", default="hardest", choices=["hardest", "obj_bg", "obj_only"])
    parser.add_argument("--corruptions", default="all", help="Corruption names separated by commas, or all.")
    parser.add_argument("--severities", default="2", help="Severity ids separated by commas, or all.")
    parser.add_argument("--output-dir", default="results/zero_shot")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--n-cluster", type=int, default=3)
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
    return parser.parse_args()


def default_exp_name(args) -> str:
    backbone = canonical_backbone(args.backbone)
    dataset = canonical_dataset(args.dataset)
    if dataset in {"modelnet_c", "scanobjectnn_c"}:
        severity = "all" if args.severities == "all" else "s" + "_".join(args.severities.split(","))
        corruption = "all" if args.corruptions == "all" else args.corruptions.replace(",", "_")
        return f"{backbone}_{dataset}_{args.sonn_variant if dataset == 'scanobjectnn_c' else 'standard'}_{corruption}_{severity}"
    if dataset == "scanobjectnn":
        return f"{backbone}_{dataset}_{args.sonn_variant}"
    return f"{backbone}_{dataset}"


@torch.no_grad()
def evaluate_task(args, task: EvalTask, dataset, loader, text_encoder, point_encoder, text_weights, device: torch.device, dtype: torch.dtype) -> dict:
    backbone = canonical_backbone(args.backbone)
    correct = 0
    total = 0
    for batch_index, (xyz, target, _classname, rgb) in enumerate(loader, start=1):
        xyz = xyz.to(device=device, dtype=dtype, non_blocking=True)
        rgb = rgb.to(device=device, dtype=dtype, non_blocking=True)
        target = target.to(device=device, non_blocking=True).view(-1).long()
        logits = point_logits(backbone, point_encoder, xyz, rgb, text_weights)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.numel()
        if args.print_freq > 0 and (batch_index % args.print_freq == 0 or batch_index == len(loader)):
            acc = 100.0 * correct / max(total, 1)
            print(f"[{task.dataset} {task.cor_type}] batch {batch_index}/{len(loader)} OA={acc:.2f}")
    accuracy = 100.0 * correct / max(total, 1)
    return {
        "backbone": backbone,
        "dataset": task.dataset,
        "corruption": task.corruption,
        "severity": task.severity,
        "variant": task.variant,
        "num_samples": total,
        "accuracy": accuracy,
    }


def run(args) -> list[dict]:
    set_random_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch_dtype(device, args.dtype)
    dataset_name = canonical_dataset(args.dataset)
    backbone = canonical_backbone(args.backbone)
    print("DPC-Point zero-shot backbone evaluation")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"backbone: {backbone}")
    print(f"dataset: {dataset_name}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"npoints: {args.npoints}")
    print(f"batch_size: {args.batch_size}")

    tasks = build_eval_tasks(args, PROJECT_ROOT)
    print(f"tasks: {len(tasks)}")
    for task in tasks:
        print(f"  - {task.dataset}: {task.cor_type} {task.variant}".rstrip())

    text_encoder, point_encoder, tokenize = load_backbone(args, PROJECT_ROOT, device, dtype, dataset_name)
    rows = []
    cached_text = None
    cached_classnames = None
    for task in tasks:
        dataset = build_dataset(args, PROJECT_ROOT, task)
        loader = build_loader(args, dataset)
        if cached_text is None or cached_classnames != dataset.classnames:
            print(f"Building text prototypes: {len(dataset.classnames)} classes")
            cached_text = build_text_classifier(text_encoder, tokenize, dataset.classnames, dataset.template, device)
            cached_classnames = list(dataset.classnames)
        print(f"Evaluating: dataset={task.dataset}, corruption={task.cor_type}, variant={task.variant}")
        row = evaluate_task(args, task, dataset, loader, text_encoder, point_encoder, cached_text, device, dtype)
        print(f"Result: OA={row['accuracy']:.2f}, samples={row['num_samples']}")
        rows.append(row)
    average = summarize_rows(rows)
    print(f"Overall average OA: {average:.2f}")
    return rows


def main():
    args = parse_args()
    exp_name = args.exp_name or default_exp_name(args)
    output_dir = PROJECT_ROOT / args.output_dir / exp_name
    log_path = output_dir / "run.log"
    summary_path = output_dir / "summary.csv"
    with tee_output(log_path):
        rows = run(args)
        write_summary(summary_path, rows)
        print(f"summary: {summary_path.relative_to(PROJECT_ROOT)}")
        print(f"log: {log_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
