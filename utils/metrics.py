"""Common metrics, seeding, and CSV/log helpers for DPC-Point."""

from __future__ import annotations

import csv
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cls_acc(output: torch.Tensor, target: torch.Tensor, topk: int = 1) -> float:
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0).item())
    return 100.0 * acc / max(target.numel(), 1)


def write_summary(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = ["backbone", "dataset", "corruption", "severity", "variant", "num_samples", "accuracy"]
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if "accuracy" in out:
                out["accuracy"] = f"{float(out['accuracy']):.4f}"
            writer.writerow(out)


def summarize_rows(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    total = sum(int(row["num_samples"]) for row in rows)
    correct = sum(float(row["accuracy"]) * int(row["num_samples"]) / 100.0 for row in rows)
    return 100.0 * correct / max(total, 1)
