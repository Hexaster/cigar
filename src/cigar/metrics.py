"""Regression metrics used in the paper."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def summarize_metric_rows(rows: list[dict[str, object]]) -> tuple[object, object]:
    import pandas as pd

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["Target", "Model"])[["MAE", "MSE", "MAPE", "R2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    return detail, summary
