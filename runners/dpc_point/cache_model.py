"""DPC-Point cache maintenance and prediction calibration."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.config import DEFAULT_CACHE_CONFIG, DEFAULT_DISTRIBUTION_CONFIG, DEFAULT_FINAL_SCORE_WEIGHTS, parse_final_score_weights
from utils.metrics import cls_acc
from utils.model_loader import canonical_backbone


@dataclass(frozen=True)
class CacheRuntimeConfig:
    entropy_cap: int = DEFAULT_CACHE_CONFIG.entropy_cap
    gpa_cap: int = DEFAULT_CACHE_CONFIG.gpa_cap
    local_cap: int = DEFAULT_CACHE_CONFIG.local_cap
    neg_cap: int = DEFAULT_CACHE_CONFIG.neg_cap
    local_centers: int = DEFAULT_CACHE_CONFIG.local_centers
    positive_beta: float = 3.0
    negative_beta: float = 1.0
    negative_entropy_lower: float = 0.2
    negative_entropy_upper: float = 0.5
    negative_mask_lower: float = 0.03
    negative_mask_upper: float = 1.0
    dist_eps: float = DEFAULT_DISTRIBUTION_CONFIG.dist_eps
    dist_min_var: float = DEFAULT_DISTRIBUTION_CONFIG.dist_min_var
    text_dist_eps: float = DEFAULT_DISTRIBUTION_CONFIG.text_dist_eps
    text_dist_min_var: float = DEFAULT_DISTRIBUTION_CONFIG.text_dist_min_var
    text_score_weight: float = DEFAULT_DISTRIBUTION_CONFIG.text_score_weight
    score_norm_mode: str = DEFAULT_DISTRIBUTION_CONFIG.score_norm_mode
    score_norm_min_count: int = DEFAULT_DISTRIBUTION_CONFIG.score_norm_min_count
    score_norm_eps: float = DEFAULT_DISTRIBUTION_CONFIG.score_norm_eps
    score_norm_clip: float = DEFAULT_DISTRIBUTION_CONFIG.score_norm_clip


def runtime_config_from_args(args) -> CacheRuntimeConfig:
    return CacheRuntimeConfig(
        entropy_cap=int(getattr(args, "entropy_cap", DEFAULT_CACHE_CONFIG.entropy_cap)),
        gpa_cap=int(getattr(args, "gpa_cap", DEFAULT_CACHE_CONFIG.gpa_cap)),
        local_cap=int(getattr(args, "local_cap", DEFAULT_CACHE_CONFIG.local_cap)),
        neg_cap=int(getattr(args, "neg_cap", DEFAULT_CACHE_CONFIG.neg_cap)),
        local_centers=int(getattr(args, "local_centers", DEFAULT_CACHE_CONFIG.local_centers)),
        positive_beta=float(getattr(args, "positive_beta", 3.0)),
        negative_beta=float(getattr(args, "negative_beta", 1.0)),
        negative_entropy_lower=float(getattr(args, "negative_entropy_lower", 0.2)),
        negative_entropy_upper=float(getattr(args, "negative_entropy_upper", 0.5)),
        negative_mask_lower=float(getattr(args, "negative_mask_lower", 0.03)),
        negative_mask_upper=float(getattr(args, "negative_mask_upper", 1.0)),
        dist_eps=float(getattr(args, "dist_eps", DEFAULT_DISTRIBUTION_CONFIG.dist_eps)),
        dist_min_var=float(getattr(args, "dist_min_var", DEFAULT_DISTRIBUTION_CONFIG.dist_min_var)),
        text_dist_eps=float(getattr(args, "text_dist_eps", DEFAULT_DISTRIBUTION_CONFIG.text_dist_eps)),
        text_dist_min_var=float(getattr(args, "text_dist_min_var", DEFAULT_DISTRIBUTION_CONFIG.text_dist_min_var)),
        text_score_weight=float(getattr(args, "text_score_weight", DEFAULT_DISTRIBUTION_CONFIG.text_score_weight)),
        score_norm_mode=getattr(args, "score_norm_mode", DEFAULT_DISTRIBUTION_CONFIG.score_norm_mode),
        score_norm_min_count=int(getattr(args, "score_norm_min_count", DEFAULT_DISTRIBUTION_CONFIG.score_norm_min_count)),
        score_norm_eps=float(getattr(args, "score_norm_eps", DEFAULT_DISTRIBUTION_CONFIG.score_norm_eps)),
        score_norm_clip=float(getattr(args, "score_norm_clip", DEFAULT_DISTRIBUTION_CONFIG.score_norm_clip)),
    )


def progress_interval(total_batches: int, print_freq: int) -> int:
    if int(print_freq) <= 0:
        return 0
    total_batches = max(int(total_batches), 1)
    return max(1, min(int(print_freq), total_batches // 20 or 1))


def should_print_progress(batch_index: int, total_batches: int, print_freq: int) -> bool:
    interval = progress_interval(total_batches, print_freq)
    if interval <= 0:
        return False
    return batch_index == 1 or batch_index == total_batches or batch_index % interval == 0


def format_progress_line(
    *,
    stage: str,
    dataset: str,
    cor_type: str,
    batch_index: int,
    total_batches: int,
    oa: float | None = None,
) -> str:
    prefix = f"[{stage}] dataset={dataset}, corruption={cor_type}, batch={batch_index}/{total_batches}"
    if oa is None:
        return prefix
    return f"{prefix}, OA={float(oa):.2f}"


def _loss_value(loss) -> float:
    if torch.is_tensor(loss):
        return float(loss.detach().float().cpu().view(-1)[0].item())
    return float(loss)


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def normalized_entropy(loss: torch.Tensor, num_classes: int) -> float:
    return float(_loss_value(loss) / math.log2(max(int(num_classes), 2)))


def _sort_cache_by_entropy(cache: dict, pred: int) -> None:
    cache[pred] = sorted(cache[pred], key=lambda item: _loss_value(item[1]))


def _update_entropy_cache(cache: dict, pred: int, item: list, capacity: int, stats: dict, phase: str) -> bool:
    pred = int(pred)
    if pred in cache:
        if len(cache[pred]) < capacity:
            cache[pred].append(item)
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_entropy_add"] += 1
            return True
        if _loss_value(item[1]) < _loss_value(cache[pred][-1][1]):
            cache[pred][-1] = item
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_entropy_replace"] += 1
            return True
        stats[f"{phase}_entropy_reject"] += 1
        return False
    cache[pred] = [item]
    stats[f"{phase}_entropy_add"] += 1
    return True


def _update_negative_cache(cache: dict, pred: int, item: list, capacity: int, stats: dict, phase: str) -> bool:
    pred = int(pred)
    if pred in cache:
        if len(cache[pred]) < capacity:
            cache[pred].append(item)
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_negative_add"] += 1
            return True
        if _loss_value(item[1]) < _loss_value(cache[pred][-1][1]):
            cache[pred][-1] = item
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_negative_replace"] += 1
            return True
        stats[f"{phase}_negative_reject"] += 1
        return False
    cache[pred] = [item]
    stats[f"{phase}_negative_add"] += 1
    return True


def _update_local_cache(local_cache: dict, pred: int, local_item: list, capacity: int, stats: dict, phase: str) -> bool:
    pred = int(pred)
    if capacity <= 0:
        stats[f"{phase}_local_reject_capacity_zero"] += 1
        return False
    if pred in local_cache:
        if len(local_cache[pred]) < capacity:
            local_cache[pred].append(local_item)
            _sort_cache_by_entropy(local_cache, pred)
            stats[f"{phase}_local_add"] += 1
            return True
        if _loss_value(local_item[1]) < _loss_value(local_cache[pred][-1][1]):
            local_cache[pred][-1] = local_item
            _sort_cache_by_entropy(local_cache, pred)
            stats[f"{phase}_local_replace"] += 1
            return True
        stats[f"{phase}_local_reject"] += 1
        return False
    local_cache[pred] = [local_item]
    stats[f"{phase}_local_add"] += 1
    return True


def _feature_key(feat: torch.Tensor) -> tuple:
    x = feat.detach()
    storage = x.untyped_storage() if hasattr(x, "untyped_storage") else x.storage()
    return (int(x.data_ptr()), int(storage.data_ptr()), tuple(x.shape), str(x.device))


def _update_visual_distribution(visual_dist: dict, pred: int, feat: torch.Tensor, cfg: CacheRuntimeConfig, stats: dict, phase: str) -> bool:
    pred = int(pred)
    key = _feature_key(feat)
    if pred not in visual_dist:
        visual_dist[pred] = {"count": 0, "mean": None, "m2": None, "seen": set()}
    entry = visual_dist[pred]
    if key in entry["seen"]:
        return False

    x = feat.detach().float()
    entry["seen"].add(key)
    if int(entry["count"]) == 0:
        entry["count"] = 1
        entry["mean"] = x.clone()
        entry["m2"] = torch.zeros_like(x)
    else:
        count_old = int(entry["count"])
        count_new = count_old + 1
        delta = x - entry["mean"]
        mean_new = entry["mean"] + delta / float(count_new)
        entry["m2"] = entry["m2"] + delta * (x - mean_new)
        entry["mean"] = mean_new
        entry["count"] = count_new
    stats[f"{phase}_visual_distribution_update"] += 1
    return True


def _visual_distribution_entry(visual_dist: dict, pred: int, cfg: CacheRuntimeConfig):
    pred = int(pred)
    if pred not in visual_dist:
        return None
    entry = visual_dist[pred]
    count = int(entry["count"])
    if count < 2:
        return None
    var = (entry["m2"] / float(max(count - 1, 1))).clamp_min(cfg.dist_min_var)
    return {"count": count, "mean": entry["mean"], "var": var}


def _text_distribution_entry(text_dist: dict | None, pred: int, ref_feat: torch.Tensor):
    if text_dist is None:
        return None
    pred = int(pred)
    if pred not in text_dist:
        return None
    entry = text_dist[pred]
    return {
        "count": int(entry["count"]),
        "mean": entry["mean"].to(device=ref_feat.device, dtype=ref_feat.dtype),
        "var": entry["var"].to(device=ref_feat.device, dtype=ref_feat.dtype),
    }


def _distribution_score(entry, feat: torch.Tensor, eps: float):
    if entry is None or int(entry["count"]) < 2:
        return None
    x = feat.detach().float().to(device=entry["mean"].device, dtype=entry["mean"].dtype)
    raw = torch.mean(((x - entry["mean"]) ** 2) / (entry["var"] + float(eps)))
    return float((-raw).detach().cpu().item())


def _make_score_norm_state() -> dict:
    return {
        "visual": {"count": 0, "mean": 0.0, "m2": 0.0},
        "text": {"count": 0, "mean": 0.0, "m2": 0.0},
    }


def _running_std(entry: dict):
    count = int(entry["count"])
    if count < 2:
        return None
    return (float(entry["m2"]) / float(count - 1)) ** 0.5


def _score_norm_ready(score_norm_state: dict, modalities: tuple[str, ...], cfg: CacheRuntimeConfig) -> bool:
    if cfg.score_norm_mode != "running_zscore":
        return False
    for modality in modalities:
        entry = score_norm_state[modality]
        std = _running_std(entry)
        if int(entry["count"]) < cfg.score_norm_min_count or std is None or std < cfg.score_norm_eps:
            return False
    return True


def _score_for_joint(score_norm_state: dict, modality: str, raw_score: float, cfg: CacheRuntimeConfig) -> float:
    if cfg.score_norm_mode != "running_zscore":
        return float(raw_score)
    entry = score_norm_state[modality]
    std = _running_std(entry)
    if int(entry["count"]) < cfg.score_norm_min_count or std is None or std < cfg.score_norm_eps:
        return float(raw_score)
    score = (float(raw_score) - float(entry["mean"])) / (std + cfg.score_norm_eps)
    if cfg.score_norm_clip > 0:
        score = max(min(score, cfg.score_norm_clip), -cfg.score_norm_clip)
    return float(score)


def _update_running_score(entry: dict, value: float) -> None:
    value = float(value)
    count_old = int(entry["count"])
    count_new = count_old + 1
    if count_old == 0:
        entry["count"] = 1
        entry["mean"] = value
        entry["m2"] = 0.0
        return
    delta = value - float(entry["mean"])
    mean_new = float(entry["mean"]) + delta / float(count_new)
    entry["count"] = count_new
    entry["mean"] = mean_new
    entry["m2"] = float(entry["m2"]) + delta * (value - mean_new)


def _update_score_norm_state(score_norm_state: dict, score: dict | None, cfg: CacheRuntimeConfig) -> int:
    if cfg.score_norm_mode != "running_zscore" or score is None:
        return 0
    updates = 0
    for modality in ("visual", "text"):
        value = score.get(modality)
        if value is not None:
            _update_running_score(score_norm_state[modality], value)
            updates += 1
    return updates


def _joint_distribution_score(visual_dist: dict, text_dist: dict | None, pred: int, feat: torch.Tensor, score_norm_state: dict, cfg: CacheRuntimeConfig):
    visual_entry = _visual_distribution_entry(visual_dist, pred, cfg)
    text_entry = _text_distribution_entry(text_dist, pred, feat)
    visual_score = _distribution_score(visual_entry, feat, cfg.dist_eps)
    text_score = _distribution_score(text_entry, feat, cfg.text_dist_eps)
    if visual_score is None:
        return None
    if text_score is None:
        visual_for_joint = _score_for_joint(score_norm_state, "visual", visual_score, cfg) if _score_norm_ready(score_norm_state, ("visual",), cfg) else float(visual_score)
        return {"joint": visual_for_joint, "visual": float(visual_score), "text": None}
    if _score_norm_ready(score_norm_state, ("visual", "text"), cfg):
        visual_for_joint = _score_for_joint(score_norm_state, "visual", visual_score, cfg)
        text_for_joint = _score_for_joint(score_norm_state, "text", text_score, cfg)
    else:
        visual_for_joint = float(visual_score)
        text_for_joint = float(text_score)
    return {
        "joint": float(visual_for_joint + cfg.text_score_weight * text_for_joint),
        "visual": float(visual_score),
        "text": float(text_score),
    }


def _update_gpa_cache(
    gpa_cache: dict,
    local_cache: dict,
    visual_dist: dict,
    text_dist: dict | None,
    score_norm_state: dict,
    pred: int,
    global_item: list,
    local_item: list,
    cfg: CacheRuntimeConfig,
    stats: dict,
    phase: str,
) -> bool:
    pred = int(pred)
    if pred not in gpa_cache:
        gpa_cache[pred] = []
        local_cache[pred] = []

    if len(gpa_cache[pred]) < cfg.gpa_cap:
        gpa_cache[pred].append(global_item)
        _sort_cache_by_entropy(gpa_cache, pred)
        _update_local_cache(local_cache, pred, local_item, cfg.local_cap, stats, phase)
        stats[f"{phase}_gpa_add_not_full"] += 1
        _update_visual_distribution(visual_dist, pred, global_item[0], cfg, stats, phase)
        return True

    worst_item = gpa_cache[pred][-1]
    curr_entropy = _loss_value(global_item[1])
    worst_entropy = _loss_value(worst_item[1])
    curr_score = _joint_distribution_score(visual_dist, text_dist, pred, global_item[0], score_norm_state, cfg)
    worst_score = _joint_distribution_score(visual_dist, text_dist, pred, worst_item[0], score_norm_state, cfg)

    if curr_score is None or worst_score is None:
        stats[f"{phase}_gpa_reject_no_distribution"] += 1
        return False
    if curr_entropy >= worst_entropy:
        stats[f"{phase}_gpa_reject_entropy"] += 1
        return False

    updates = _update_score_norm_state(score_norm_state, curr_score, cfg)
    updates += _update_score_norm_state(score_norm_state, worst_score, cfg)
    if updates:
        stats[f"{phase}_score_norm_update"] += updates

    if curr_score["joint"] > worst_score["joint"]:
        gpa_cache[pred][-1] = global_item
        _sort_cache_by_entropy(gpa_cache, pred)
        _update_local_cache(local_cache, pred, local_item, cfg.local_cap, stats, phase)
        stats[f"{phase}_gpa_replace"] += 1
        _update_visual_distribution(visual_dist, pred, global_item[0], cfg, stats, phase)
        return True

    stats[f"{phase}_gpa_reject_joint"] += 1
    return False


def hierarchical_outputs(backbone: str, point_encoder, feature: torch.Tensor, clip_weights: torch.Tensor):
    backbone = canonical_backbone(backbone)
    if backbone == "ulip":
        pc_feats, patch_centers = point_encoder(feature[:, :, :3])
    elif backbone == "openshape":
        pc_feats, patch_centers = point_encoder(feature[:, :, :3], feature)
    elif backbone == "uni3d":
        pc_feats, patch_centers = point_encoder.encode_pc(feature)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    pc_feats = F.normalize(pc_feats.float(), dim=-1)
    patch_centers = F.normalize(patch_centers.float(), dim=-1)
    clip_logits = 100.0 * pc_feats @ clip_weights
    loss = softmax_entropy(clip_logits)
    prob_map = clip_logits.softmax(dim=1)
    pred = int(clip_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0])
    return pc_feats, patch_centers, clip_logits, loss, prob_map, pred


def _move_batch(xyz, target, rgb, device: torch.device, dtype: torch.dtype):
    xyz = xyz.to(device=device, dtype=dtype, non_blocking=True)
    rgb = rgb.to(device=device, dtype=dtype, non_blocking=True)
    target = target.to(device=device, non_blocking=True).view(-1).long()
    return torch.cat([xyz, rgb], dim=-1), target


@torch.no_grad()
def build_positive_caches(args, loader, point_encoder, clip_weights: torch.Tensor, text_dist: dict | None, cfg: CacheRuntimeConfig, device: torch.device, dtype: torch.dtype):
    entropy_cache = {}
    gpa_cache = {}
    local_cache = {}
    visual_dist = {}
    score_norm_state = _make_score_norm_state()
    stats = defaultdict(int)
    correct = 0
    total = 0
    total_batches = len(loader)

    print("Building DPC-Point positive caches")
    for batch_index, (xyz, target, _classname, rgb) in enumerate(loader, start=1):
        feature, target = _move_batch(xyz, target, rgb, device, dtype)
        pc_feats, patch_centers, clip_logits, loss, _prob_map, pred = hierarchical_outputs(args.backbone, point_encoder, feature, clip_weights)
        global_item = [pc_feats, loss]
        local_item = [patch_centers, loss]
        correct += int((clip_logits.argmax(dim=1) == target).sum().item())
        total += int(target.numel())

        _update_gpa_cache(gpa_cache, local_cache, visual_dist, text_dist, score_norm_state, pred, global_item, local_item, cfg, stats, "build")
        if _update_entropy_cache(entropy_cache, pred, global_item, cfg.entropy_cap, stats, "build"):
            _update_visual_distribution(visual_dist, pred, global_item[0], cfg, stats, "build")

        if should_print_progress(batch_index, total_batches, getattr(args, "print_freq", 0)):
            oa = 100.0 * correct / max(total, 1)
            print(format_progress_line(
                stage="cache",
                dataset=getattr(args, "dataset", ""),
                cor_type=getattr(args, "current_cor_type", "clean"),
                batch_index=batch_index,
                total_batches=total_batches,
                oa=oa,
            ), flush=True)

        num_classes = clip_logits.size(1)
        if (
            sum(len(v) for v in entropy_cache.values()) >= cfg.entropy_cap * num_classes
            and sum(len(v) for v in gpa_cache.values()) >= cfg.gpa_cap * num_classes
            and sum(len(v) for v in local_cache.values()) >= cfg.local_cap * num_classes
        ):
            print("DPC-Point positive caches are full", flush=True)
            break

    print("Cache stage completed", flush=True)
    return entropy_cache, gpa_cache, local_cache, visual_dist, score_norm_state, stats


@torch.no_grad()
def compute_cache_score(pc_feats: torch.Tensor, cache: dict, beta: float, clip_weights: torch.Tensor, neg_mask_thresholds: tuple[float, float] | None = None) -> torch.Tensor:
    cache_keys = []
    cache_values = []
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item[0])
            cache_values.append(item[2] if neg_mask_thresholds is not None else int(class_index))
    if not cache_keys:
        return torch.zeros_like(pc_feats @ clip_weights)

    cache_keys = torch.cat(cache_keys, dim=0).to(device=pc_feats.device, dtype=pc_feats.dtype).permute(1, 0)
    if neg_mask_thresholds is not None:
        values = torch.cat(cache_values, dim=0).to(device=pc_feats.device, dtype=pc_feats.dtype)
        lower, upper = neg_mask_thresholds
        cache_values_tensor = ((values > lower) & (values < upper)).to(dtype=pc_feats.dtype)
    else:
        labels = torch.tensor(cache_values, device=pc_feats.device, dtype=torch.long)
        cache_values_tensor = F.one_hot(labels, num_classes=clip_weights.size(1)).to(dtype=pc_feats.dtype)

    affinity = pc_feats @ cache_keys
    return ((-1.0) * (float(beta) - float(beta) * affinity)).exp() @ cache_values_tensor


@torch.no_grad()
def compute_local_cache_score(patch_centers: torch.Tensor, local_cache: dict, beta: float, clip_weights: torch.Tensor) -> torch.Tensor:
    local_keys = []
    labels = []
    for class_index in sorted(local_cache.keys()):
        for item in local_cache[class_index]:
            centers = item[0]
            local_keys.append(centers)
            labels.extend([int(class_index)] * int(centers.shape[0]))
    if not local_keys:
        return torch.zeros((1, clip_weights.size(1)), device=patch_centers.device, dtype=patch_centers.dtype)
    local_keys = torch.cat(local_keys, dim=0).to(device=patch_centers.device, dtype=patch_centers.dtype).permute(1, 0)
    local_values = F.one_hot(torch.tensor(labels, device=patch_centers.device, dtype=torch.long), num_classes=clip_weights.size(1)).to(dtype=patch_centers.dtype)
    affinity = patch_centers.mean(dim=0, keepdim=True) @ local_keys
    return ((-1.0) * (float(beta) - float(beta) * affinity)).exp() @ local_values


@torch.no_grad()
def run_dpc_point(args, loader, point_encoder, clip_weights: torch.Tensor, text_dist: dict | None, device: torch.device, dtype: torch.dtype) -> dict:
    cfg = runtime_config_from_args(args)
    final_score_weights = parse_final_score_weights(getattr(args, "final_score_weights", DEFAULT_FINAL_SCORE_WEIGHTS))
    primary_weight = final_score_weights[0]

    entropy_cache, _gpa_cache, local_cache, visual_dist, score_norm_state, stats = build_positive_caches(
        args, loader, point_encoder, clip_weights, text_dist, cfg, device, dtype
    )
    negative_cache = {}
    accuracies_by_weight = {weight["name"]: [] for weight in final_score_weights}
    total_batches = len(loader)
    dataset_name = getattr(args, "dataset", "")
    cor_type = getattr(args, "current_cor_type", "clean")

    for batch_index, (xyz, target, _classname, rgb) in enumerate(loader, start=1):
        feature, target = _move_batch(xyz, target, rgb, device, dtype)
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = hierarchical_outputs(args.backbone, point_encoder, feature, clip_weights)

        global_item = [pc_feats, loss]
        local_item = [patch_centers, loss]
        _update_gpa_cache(_gpa_cache, local_cache, visual_dist, text_dist, score_norm_state, pred, global_item, local_item, cfg, stats, "test")
        if _update_entropy_cache(entropy_cache, pred, global_item, cfg.entropy_cap, stats, "test"):
            _update_visual_distribution(visual_dist, pred, global_item[0], cfg, stats, "test")

        prop_entropy = normalized_entropy(loss, clip_weights.size(1))
        if cfg.negative_entropy_lower < prop_entropy < cfg.negative_entropy_upper:
            _update_negative_cache(negative_cache, pred, [pc_feats, loss, prob_map], cfg.neg_cap, stats, "test")

        y_zs = clip_logits.clone()
        y_g = compute_cache_score(pc_feats, entropy_cache, cfg.positive_beta, clip_weights)
        y_l = compute_local_cache_score(patch_centers, local_cache, cfg.positive_beta, clip_weights)
        y_n = compute_cache_score(
            pc_feats,
            negative_cache,
            cfg.negative_beta,
            clip_weights,
            (cfg.negative_mask_lower, cfg.negative_mask_upper),
        )

        for weight in final_score_weights:
            final_logits = y_zs + weight["alpha_g"] * y_g + weight["alpha_l"] * y_l - weight["alpha_n"] * y_n
            accuracies_by_weight[weight["name"]].append(cls_acc(final_logits, target))

        if should_print_progress(batch_index, total_batches, getattr(args, "print_freq", 0)):
            values = accuracies_by_weight[primary_weight["name"]]
            print(format_progress_line(
                stage="infer",
                dataset=dataset_name,
                cor_type=cor_type,
                batch_index=batch_index,
                total_batches=total_batches,
                oa=sum(values) / len(values),
            ), flush=True)

    weight_results = []
    for weight in final_score_weights:
        values = accuracies_by_weight[weight["name"]]
        weight_results.append({
            "name": weight["name"],
            "alpha_g": weight["alpha_g"],
            "alpha_l": weight["alpha_l"],
            "alpha_n": weight["alpha_n"],
            "acc": sum(values) / max(len(values), 1),
        })

    primary = weight_results[0]
    print(f"Final OA: {primary['acc']:.2f}", flush=True)

    return {
        "primary_acc": float(primary["acc"]),
        "primary_weight": primary_weight,
        "weight_results": weight_results,
        "cache_summary": {
            "entropy_cache_total": int(sum(len(v) for v in entropy_cache.values())),
            "gpa_cache_total": int(sum(len(v) for v in _gpa_cache.values())),
            "local_cache_total": int(sum(len(v) for v in local_cache.values())),
            "negative_cache_total": int(sum(len(v) for v in negative_cache.values())),
        },
    }
