"""Dataset task definitions for formal DPC-Point inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from datasets.modelnet_c import ModelNet_C
from datasets.sonn_c import SONN_C
from utils.model_loader import canonical_backbone, project_path


DATASET_ALIASES = {
    "modelnet": "modelnet",
    "modelnet_clean": "modelnet",
    "modelnet_c": "modelnet_c",
    "modelnet-c": "modelnet_c",
    "scanobjectnn": "scanobjectnn",
    "scanobjectnn_clean": "scanobjectnn",
    "scanobjectnn_c": "scanobjectnn_c",
    "scanobjectnn-c": "scanobjectnn_c",
}

CORRUPTIONS = [
    "add_global",
    "add_local",
    "dropout_global",
    "dropout_local",
    "jitter",
    "rotate",
    "scale",
]

SEVERITY_SETS = {
    "clean": ["clean"],
    "s2": ["2"],
    "all35": ["0", "1", "2", "3", "4"],
}


@dataclass(frozen=True)
class TaskSpec:
    dataset_key: str
    display_dataset: str
    corruption: str
    severity: str
    cor_type: str
    file_path: Path
    variant: str = ""

    def to_config(self) -> dict:
        return {
            "dataset": self.dataset_key,
            "display_dataset": self.display_dataset,
            "corruption": self.corruption,
            "severity": self.severity,
            "cor_type": self.cor_type,
            "file_path": str(self.file_path),
            "variant": self.variant,
        }


def canonical_dataset(name: str) -> str:
    key = str(name).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_ALIASES[key]


def _severity_values(severity_set: str) -> list[str]:
    key = str(severity_set).strip().lower()
    if key in SEVERITY_SETS:
        return SEVERITY_SETS[key]
    values = [item.strip() for item in key.split(",") if item.strip()]
    if values and all(item.isdigit() for item in values):
        return values
    raise ValueError(f"Unsupported severity_set: {severity_set}")


def build_task_specs(
    dataset: str,
    severity_set: str = "s2",
    modelnet_c_root: str | Path = "data/modelnet_c",
    scanobjectnn_c_root: str | Path = "data/sonn_c",
    sonn_variant: str = "hardest",
    corruptions: list[str] | tuple[str, ...] | str = "all",
) -> list[TaskSpec]:
    dataset_key = canonical_dataset(dataset)
    modelnet_root = Path(modelnet_c_root)
    sonn_root = Path(scanobjectnn_c_root)

    if dataset_key == "modelnet":
        return [TaskSpec(dataset_key, "ModelNet", "clean", "clean", "clean", modelnet_root / "clean.h5")]
    if dataset_key == "scanobjectnn":
        return [TaskSpec(dataset_key, "ScanObjectNN", "clean", "clean", "clean", sonn_root / sonn_variant / "clean.h5", sonn_variant)]

    selected_corruptions = CORRUPTIONS if corruptions == "all" else [item.strip() for item in str(corruptions).split(",") if item.strip()]
    unknown = [item for item in selected_corruptions if item not in CORRUPTIONS]
    if unknown:
        raise ValueError(f"Unsupported corruptions: {unknown}")

    tasks = []
    severity_values = _severity_values(severity_set)
    for corruption in selected_corruptions:
        for severity in severity_values:
            cor_type = f"{corruption}_{severity}"
            if dataset_key == "modelnet_c":
                tasks.append(TaskSpec(dataset_key, "ModelNet-C", corruption, severity, cor_type, modelnet_root / f"{cor_type}.h5"))
            elif dataset_key == "scanobjectnn_c":
                tasks.append(TaskSpec(dataset_key, "ScanObjectNN-C", corruption, severity, cor_type, sonn_root / sonn_variant / f"{cor_type}.h5", sonn_variant))
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
    return tasks


def dataset_namespace(args, project_root: Path, task: TaskSpec) -> SimpleNamespace:
    return SimpleNamespace(
        lm3d=canonical_backbone(args.backbone),
        npoints=args.npoints,
        modelnet_c_root=str(project_path(project_root, getattr(args, "modelnet_c_root", "data/modelnet_c"))),
        sonn_c_root=str(project_path(project_root, getattr(args, "scanobjectnn_c_root", "data/sonn_c"))),
        sonn_variant=task.variant or getattr(args, "sonn_variant", "hardest"),
        cor_type=task.cor_type,
    )


def build_dataset(args, project_root: Path, task: TaskSpec):
    cfg = dataset_namespace(args, project_root, task)
    if task.dataset_key in {"modelnet", "modelnet_c"}:
        return ModelNet_C(cfg)
    if task.dataset_key in {"scanobjectnn", "scanobjectnn_c"}:
        return SONN_C(cfg)
    raise ValueError(f"Unsupported dataset: {task.dataset_key}")


def build_loader(args, dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
