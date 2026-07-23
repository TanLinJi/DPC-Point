from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners.dpc_point.datasets import build_task_specs
from utils.llm_prompts import get_prompt_template
from utils.config import (
    DEFAULT_CACHE_CONFIG,
    DEFAULT_FINAL_SCORE_WEIGHTS,
    final_score_weight_name,
    parse_final_score_weights,
    release_config_payload,
)


def test_final_score_weights_parse_named_and_default_names():
    weights = parse_final_score_weights("best:4.4,3.9,0.19;4.5,4.0,0.18")
    assert weights[0] == {"name": "best", "alpha_g": 4.4, "alpha_l": 3.9, "alpha_n": 0.19}
    assert weights[1]["name"] == "ag4p5_al4p0_an0p18"
    assert weights[1]["alpha_l"] == 4.0


def test_default_release_hyperparameters_are_fixed_setting():
    assert DEFAULT_CACHE_CONFIG.entropy_cap == 3
    assert DEFAULT_CACHE_CONFIG.gpa_cap == 3
    assert DEFAULT_CACHE_CONFIG.local_cap == 3
    assert DEFAULT_CACHE_CONFIG.neg_cap == 6
    assert DEFAULT_CACHE_CONFIG.local_centers == 3
    assert DEFAULT_FINAL_SCORE_WEIGHTS[0]["alpha_g"] == 4.4
    assert DEFAULT_FINAL_SCORE_WEIGHTS[0]["alpha_l"] == 3.9
    assert DEFAULT_FINAL_SCORE_WEIGHTS[0]["alpha_n"] == 0.19
    assert final_score_weight_name(4.4, 3.9, 0.19) == "ag4p4_al3p9_an0p19"


def test_dataset_task_specs_follow_release_paths():
    modelnet = build_task_specs("modelnet")
    assert len(modelnet) == 1
    assert modelnet[0].dataset_key == "modelnet"
    assert modelnet[0].file_path == Path("data/modelnet_c/clean.h5")

    modelnet_c = build_task_specs("modelnet_c", severity_set="s2")
    assert len(modelnet_c) == 7
    assert modelnet_c[0].file_path == Path("data/modelnet_c/add_global_2.h5")

    scanobjectnn = build_task_specs("scanobjectnn")
    assert scanobjectnn[0].file_path == Path("data/sonn_c/hardest/clean.h5")

    scanobjectnn_c = build_task_specs("scanobjectnn_c", severity_set="all35")
    assert len(scanobjectnn_c) == 35
    assert scanobjectnn_c[-1].file_path == Path("data/sonn_c/hardest/scale_4.h5")


def test_release_config_payload_records_experiment_hyperparameters():
    args = SimpleNamespace(
        backbone="uni3d",
        dataset="scanobjectnn_c",
        severity_set="s2",
        prompt_source="handcrafted_with_llm",
        prompt_static_weight=0.75,
        prompt_dynamic_weight=0.25,
        text_score_weight=0.15,
        entropy_cap=3,
        gpa_cap=3,
        local_cap=3,
        neg_cap=6,
        local_centers=3,
        final_score_weights=parse_final_score_weights("best:4.4,3.9,0.19"),
        device="cuda:0",
        dtype="auto",
        seed=1,
        npoints=1024,
        num_workers=2,
        print_freq=500,
        output_dir="results/dpc_point",
        exp_name="",
        ulip_text_ckpt="weights/ulip/slip_base_100ep.pt",
        ulip_point_ckpt="weights/ulip/pointbert_ulip1.pt",
    )
    payload = release_config_payload(args, build_task_specs("scanobjectnn_c", severity_set="s2"))
    assert payload["method"] == "DPC-Point"
    assert payload["backbone"] == "uni3d"
    assert payload["dataset"]["name"] == "scanobjectnn_c"
    assert payload["cache"]["neg_cap"] == 6
    assert payload["runtime"]["npoints"] == 1024
    assert payload["output"]["directory"] == "results/dpc_point"
    assert payload["backbone_config"]["ulip_point_ckpt"] == "weights/ulip/pointbert_ulip1.pt"
    assert payload["final_score"]["formula"] == "y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n"
    assert payload["tasks"][0]["cor_type"] == "add_global_2"


def test_handcrafted_templates_live_under_text_templates():
    handcrafted_path = PROJECT_ROOT / "text_templates" / "handcrafted" / "original_handcrafted.json"
    removed_experiment_path = PROJECT_ROOT / "text_templates" / "handcrafted" / ("manual" + "_3d.json")
    assert handcrafted_path.exists()
    assert not removed_experiment_path.exists()

    handcrafted = get_prompt_template(SimpleNamespace(prompt_source="handcrafted"), ["chair"], "modelnet_c", PROJECT_ROOT)
    assert "a point cloud model of {}." in handcrafted
    assert "a photo of a {}." in handcrafted
    assert len(handcrafted) == 64


def test_release_prompt_sources_do_not_keep_experiment_names():
    legacy_sources = [
        "manual" + "_full",
        "manual" + "_3d",
        "manual" + "full" + "_llm" + "_dynamic" + "_init",
        "llm" + "_dynamic" + "_init",
    ]
    for source in legacy_sources:
        try:
            get_prompt_template(SimpleNamespace(prompt_source=source), ["chair"], "modelnet_c", PROJECT_ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Experiment prompt source should not be supported in release code: {source}")


if __name__ == "__main__":
    test_final_score_weights_parse_named_and_default_names()
    test_default_release_hyperparameters_are_fixed_setting()
    test_dataset_task_specs_follow_release_paths()
    test_release_config_payload_records_experiment_hyperparameters()
    test_handcrafted_templates_live_under_text_templates()
    test_release_prompt_sources_do_not_keep_experiment_names()
    print("dpc-point release config tests passed")
