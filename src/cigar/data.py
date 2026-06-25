"""Dataset loading and per-puff feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .constants import RANDOM_STATE, TARGETS


@dataclass
class PaperData:
    dataset1_physics: pd.DataFrame
    dataset2_physical: pd.DataFrame
    dataset2_chemical: pd.DataFrame
    label_encoder: LabelEncoder


def load_paper_data(data_dir: str | Path = "data/raw") -> PaperData:
    data_dir = Path(data_dir)
    dataset1 = pd.read_excel(data_dir / "data202605.xlsx", index_col=0).T
    dataset2_physical = pd.read_excel(
        data_dir / "data_chemical_factor.xlsx",
        sheet_name=1,
        index_col=0,
    ).T
    dataset2_chemical = pd.read_excel(
        data_dir / "data20260120.xlsx",
        sheet_name="X3",
        index_col=0,
    )

    if not dataset2_physical.index.equals(dataset2_chemical.index):
        raise ValueError("Dataset 2 physical and chemical rows are not aligned.")

    label_encoder = LabelEncoder()
    label_encoder.fit(dataset1["type"])
    type_to_idx = {name: idx for idx, name in enumerate(label_encoder.classes_)}

    dataset1 = dataset1.copy()
    dataset2_physical = dataset2_physical.copy()
    dataset1["type_idx"] = dataset1["type"].map(type_to_idx).fillna(-1).astype(int)
    dataset2_physical["type_idx"] = -1

    return PaperData(
        dataset1_physics=dataset1,
        dataset2_physical=dataset2_physical,
        dataset2_chemical=dataset2_chemical,
        label_encoder=label_encoder,
    )


def puff_columns(k: int) -> dict[str, str]:
    suffix = ",mL" if k == 1 else ""
    return {
        "cone": f"第{k}口燃烧锥进气{suffix}",
        "paper": f"第{k}口卷烟纸进气{suffix}",
        "vent": f"第{k}口通风孔进气{suffix}",
    }


def ensure_puff_columns(df: pd.DataFrame, max_puffs: int) -> pd.DataFrame:
    df_out = df.copy()
    for k in range(1, max_puffs + 1):
        for col in puff_columns(k).values():
            if col not in df_out.columns:
                df_out[col] = 0
        if not any(c.startswith(f"第{k}口烟丝长") for c in df_out.columns):
            df_out[f"第{k}口烟丝长"] = 0
    return df_out


def preprocess_df(raw_df: pd.DataFrame, target_col: str, max_puffs: int = 10) -> pd.DataFrame:
    df = ensure_puff_columns(raw_df, max_puffs=max_puffs)
    df = df.dropna(subset=[target_col]).copy()
    df["area"] = pd.to_numeric(df["卷烟圆周,cm"], errors="coerce") ** 2 / (4 * np.pi)

    new_columns: dict[str, pd.Series] = {}
    puff_count = pd.to_numeric(df["抽吸口数"], errors="coerce")
    area = pd.to_numeric(df["area"], errors="coerce")

    for k in range(1, max_puffs + 1):
        cols = puff_columns(k)
        puff_time = 2 * (puff_count - (k - 1)).clip(lower=0, upper=1)
        safe_puff_time = puff_time.mask(puff_time == 0)

        cone_flow = pd.to_numeric(df[cols["cone"]], errors="coerce")
        paper_flow = pd.to_numeric(df[cols["paper"]], errors="coerce")
        vent_flow = pd.to_numeric(df[cols["vent"]], errors="coerce")

        q_cone_area = cone_flow / area
        new_columns[f"Q_cone/A_{k}"] = q_cone_area
        new_columns[f"(Q_cone/A_{k})^2"] = q_cone_area**2

        v_rod = (cone_flow + paper_flow / 2) / (safe_puff_time * area)
        v_pre = (cone_flow + paper_flow) / (safe_puff_time * area)
        v_post = (cone_flow + paper_flow + vent_flow) / (safe_puff_time * area)
        new_columns[f"V_rod_{k}"] = v_rod.where(np.isfinite(v_rod), 0).fillna(0)
        new_columns[f"v_prefilter_{k}"] = v_pre.where(np.isfinite(v_pre), 0).fillna(0)
        new_columns[f"v_postfilter_{k}"] = v_post.where(np.isfinite(v_post), 0).fillna(0)

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def target_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {target: int(frame[target].notna().sum()) for target in TARGETS}


def dataset_contract(data: PaperData) -> dict[str, object]:
    return {
        "dataset1_rows": len(data.dataset1_physics),
        "dataset1_types": int(data.dataset1_physics["type"].nunique()),
        "dataset1_target_counts": target_counts(data.dataset1_physics),
        "dataset2_physical_rows": len(data.dataset2_physical),
        "dataset2_chemical_rows": len(data.dataset2_chemical),
        "dataset2_chemical_features": data.dataset2_chemical.shape[1],
        "dataset2_aligned": data.dataset2_physical.index.equals(data.dataset2_chemical.index),
        "seed": RANDOM_STATE,
    }
