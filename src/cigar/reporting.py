"""Console formatting for per-fold metrics and KFold summaries."""

from __future__ import annotations

import numpy as np


PARAMETER_ORDER = [
    "G1",
    "G2",
    "G3",
    "a_init",
    "m_rod",
    "n_rod",
    "p_rod",
    "m_fil",
    "n_fil",
    "p_fil",
]
METRIC_ORDER = ["MSE", "MAE", "MAPE", "R2"]


def _to_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def format_type_mapping(label_encoder) -> str:
    mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    return f"全局Type映射关系: {mapping}"


def format_physical_params(model) -> str:
    if not hasattr(model, "get_physical_params"):
        return ""
    params = model.get_physical_params()
    names = [name for name in PARAMETER_ORDER if name in params]
    names.extend(name for name in params if name not in names)
    return ", ".join(f"{name}: {_to_float(params[name]):.4f}" for name in names)


def format_fold_metrics(model_name: str, target: str, fold_idx: int, n_splits: int, metrics: dict[str, float]) -> str:
    return (
        f"{model_name} - {target} fold {fold_idx}/{n_splits}: "
        f"MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, "
        f"MAPE={metrics['MAPE']:.4f}, R2={metrics['R2']:.4f}"
    )


def format_metric_summary(metric: str, mean: float, std: float) -> str:
    return f"  {metric}: mean={mean:.4f} std={std:.4f}"


def print_kfold_summary(model_name: str, target: str, n_splits: int, rows: list[dict[str, object]]) -> None:
    print(f"{model_name} - {target} KFold summary (n_splits={n_splits}):")
    for metric in METRIC_ORDER:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        print(format_metric_summary(metric, values.mean(), values.std(ddof=0)))


def print_type_embeddings(model, label_encoder) -> None:
    if not hasattr(model, "type_embedding"):
        return
    weights = model.type_embedding.weight.detach().cpu().view(-1).tolist()
    print("Type Embedding Parameters:")
    print("-" * 40)
    for cls, weight in zip(label_encoder.classes_, weights):
        print(f"Type {cls}: k_G = {float(weight):.4f}")
