from pathlib import Path
import sys
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.dpc_point.cache_model import format_progress_line, progress_interval
from runners.dpc_point.datasets import build_task_specs
from runners.dpc_point.infer import default_exp_name, output_paths, parse_args, print_run_header


def test_infer_has_no_dry_run_argument():
    args = parse_args(["--backbone", "ulip", "--dataset", "modelnet"])
    assert not hasattr(args, "dry_run")


def test_default_exp_name_and_output_paths_are_release_style():
    args = SimpleNamespace(
        backbone="uni3d",
        dataset="scanobjectnn_c",
        severity_set="s2",
        sonn_variant="hardest",
        exp_name="",
        output_dir="results/dpc_point",
    )
    exp_name = default_exp_name(args)
    assert exp_name == "uni3d_scanobjectnn_c_hardest_s2"

    run_dir, log_path, summary_path, config_path = output_paths(args, PROJECT_ROOT)
    assert run_dir == PROJECT_ROOT / "results/dpc_point" / exp_name
    assert log_path == run_dir / "run.log"
    assert summary_path == run_dir / "summary.csv"
    assert config_path == run_dir / "config.json"
    assert "logs" not in str(log_path.relative_to(run_dir))


def test_release_header_keeps_hyperparameters_out_of_console():
    args = parse_args(["--backbone", "uni3d", "--dataset", "scanobjectnn_c", "--severity-set", "s2"])
    run_dir, _log_path, _summary_path, _config_path = output_paths(args, PROJECT_ROOT)
    tasks = build_task_specs("scanobjectnn_c", severity_set="s2")

    stream = StringIO()
    with redirect_stdout(stream):
        print_run_header(args, tasks, run_dir)
    output = stream.getvalue()

    assert "backbone: uni3d" in output
    assert "dataset: scanobjectnn_c" in output
    assert "tasks: 7" in output
    assert "final_score" not in output
    assert "alpha_g" not in output
    assert "alpha_l" not in output
    assert "alpha_n" not in output


def test_release_progress_uses_oa_name_and_frequent_interval():
    assert progress_interval(total_batches=2400, print_freq=500) == 120
    assert progress_interval(total_batches=7, print_freq=500) == 1

    line = format_progress_line(
        stage="infer",
        dataset="scanobjectnn_c",
        cor_type="rotate_2",
        batch_index=120,
        total_batches=2400,
        oa=54.321,
    )

    assert "infer" in line
    assert "scanobjectnn_c" in line
    assert "rotate_2" in line
    assert "120/2400" in line
    assert "OA=54.32" in line
    assert "primary_OA" not in line


if __name__ == "__main__":
    test_infer_has_no_dry_run_argument()
    test_default_exp_name_and_output_paths_are_release_style()
    test_release_header_keeps_hyperparameters_out_of_console()
    test_release_progress_uses_oa_name_and_frequent_interval()
    print("dpc-point infer contract tests passed")
