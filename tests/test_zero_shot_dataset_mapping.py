from types import SimpleNamespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.zero_shot import build_eval_tasks, canonical_dataset, dataset_config


def make_args(dataset):
    return SimpleNamespace(
        backbone="ulip",
        dataset=dataset,
        npoints=1024,
        modelnet_c_root="data/modelnet_c",
        scanobjectnn_c_root="data/sonn_c",
        sonn_variant="hardest",
        corruptions="all",
        severities="2",
    )


def test_clean_dataset_names_follow_release_protocol():
    assert canonical_dataset("modelnet") == "modelnet"
    assert canonical_dataset("scanobjectnn") == "scanobjectnn"


def test_modelnet_clean_uses_modelnet_c_clean_h5_protocol():
    args = make_args("modelnet")
    task = build_eval_tasks(args, Path("/project"))[0]
    cfg = dataset_config(args, Path("/project"), task)

    assert task.dataset == "modelnet"
    assert task.cor_type == "clean"
    assert cfg.modelnet_c_root == "/project/data/modelnet_c"
    assert cfg.cor_type == "clean"


def test_scanobjectnn_clean_uses_sonn_c_hardest_clean_h5_protocol():
    args = make_args("scanobjectnn")
    task = build_eval_tasks(args, Path("/project"))[0]
    cfg = dataset_config(args, Path("/project"), task)

    assert task.dataset == "scanobjectnn"
    assert task.variant == "hardest"
    assert task.cor_type == "clean"
    assert cfg.sonn_c_root == "/project/data/sonn_c"
    assert cfg.sonn_variant == "hardest"
    assert cfg.cor_type == "clean"

def main():
    test_clean_dataset_names_follow_release_protocol()
    test_modelnet_clean_uses_modelnet_c_clean_h5_protocol()
    test_scanobjectnn_clean_uses_sonn_c_hardest_clean_h5_protocol()
    print("zero-shot dataset mapping tests passed")


if __name__ == "__main__":
    main()
