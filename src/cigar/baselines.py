"""XGBoost and TabPFN baseline evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from .constants import BASELINE_DROP_COLS, RANDOM_STATE, TARGETS
from .data import PaperData, preprocess_df
from .metrics import regression_metrics
from .reporting import format_fold_metrics, print_kfold_summary
from .tabpfn import make_tabpfn_regressor


def build_fold_frames(data: PaperData, train_idx, test_idx) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical_train = data.dataset2_physical.iloc[train_idx]
    physical_test = data.dataset2_physical.iloc[test_idx]
    chemical_train = data.dataset2_chemical.iloc[train_idx]
    chemical_test = data.dataset2_chemical.iloc[test_idx]

    train_df = pd.merge(physical_train, chemical_train, left_index=True, right_index=True, how="inner")
    train_df = pd.concat([train_df, data.dataset1_physics], ignore_index=False)
    test_df = pd.merge(physical_test, chemical_test, left_index=True, right_index=True, how="inner")
    return train_df, test_df


def split_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    train_proc = preprocess_df(train_df, target_col=target, max_puffs=10)
    test_proc = preprocess_df(test_df, target_col=target, max_puffs=10)
    x_train = train_proc.drop(columns=[c for c in TARGETS if c in train_proc.columns]).copy()
    x_train = x_train.drop(columns=BASELINE_DROP_COLS, errors="ignore").astype(np.float32)
    x_test = test_proc.drop(columns=[c for c in TARGETS if c in test_proc.columns]).copy()
    x_test = x_test.drop(columns=BASELINE_DROP_COLS, errors="ignore")
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0).astype(np.float32)
    y_train = train_proc[target].astype(np.float32)
    y_test = test_proc[target].astype(np.float32)
    return x_train, x_test, y_train, y_test


def load_xgb_params(path: str | Path = "monolithic_xgb_params.json") -> dict[str, dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {}
    for target, target_params in raw.items():
        cleaned = dict(target_params)
        cleaned["device"] = device
        cleaned.setdefault("tree_method", "hist")
        cleaned.setdefault("objective", "reg:squarederror")
        cleaned.setdefault("random_state", RANDOM_STATE)
        params[target] = cleaned
    return params


def run_baseline_cv(
    data: PaperData,
    model_name: str,
    targets: list[str] | None = None,
    n_splits: int = 5,
    xgb_params_path: str | Path = "monolithic_xgb_params.json",
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = targets or TARGETS
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    xgb_params = load_xgb_params(xgb_params_path) if model_name == "XGBoost" else {}
    rows: list[dict[str, object]] = []
    log_model_name = "XGBoost (log target)" if model_name == "XGBoost" else model_name

    if verbose and model_name == "XGBoost" and xgb_params:
        first_params = next(iter(xgb_params.values()))
        print(f"XGBoost device: {first_params.get('device', 'cpu')}")

    for target in targets:
        if verbose:
            print(f"\n=== {model_name} target: {target} ===")
        target_rows: list[dict[str, object]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.arange(len(data.dataset2_physical))), start=1):
            train_df, test_df = build_fold_frames(data, train_idx, test_idx)
            x_train, x_test, y_train, y_test = split_xy(train_df, test_df, target)

            if model_name == "TabPFN":
                model = make_tabpfn_regressor(random_state=RANDOM_STATE)
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
            elif model_name == "XGBoost":
                model = XGBRegressor(**xgb_params[target])
                model.fit(x_train, np.log1p(y_train))
                predictions = np.expm1(model.predict(x_test))
            else:
                raise ValueError("model_name must be 'TabPFN' or 'XGBoost'")

            metrics = regression_metrics(y_test, predictions)
            row = {"Target": target, "Model": model_name, "Fold": fold_idx, **metrics}
            rows.append(row)
            target_rows.append(row)
            if verbose:
                print(format_fold_metrics(log_model_name, target, fold_idx, n_splits, metrics))
        if verbose:
            print_kfold_summary(log_model_name, target, n_splits, target_rows)

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["Target", "Model"])[["MAE", "MSE", "MAPE", "R2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    return detail, summary
