"""PyTorch dataset wrapper for CIGAR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import PHYSICS_STATIC_COLS


class CigaretteDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        max_puffs: int = 10,
        type_idx_col: str = "type_idx",
        factor_col: str | None = "provided_factor",
    ) -> None:
        self.df = df.copy()
        self.idx = self.df.index.values
        self.df = self.df.reset_index(drop=True)
        self.target = self.df[target_col].values.astype(np.float32)

        self.static_features = (
            self.df[PHYSICS_STATIC_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )

        if type_idx_col in self.df.columns:
            self.types = (
                pd.to_numeric(self.df[type_idx_col], errors="coerce")
                .fillna(-1)
                .values.astype(np.int64)
            )
        else:
            self.types = np.full(len(self.df), -1, dtype=np.int64)

        if factor_col is not None and factor_col in self.df.columns:
            self.provided_factor = (
                pd.to_numeric(self.df[factor_col], errors="coerce")
                .values.astype(np.float32)
            )
        else:
            self.provided_factor = np.full(len(self.df), np.nan, dtype=np.float32)

        self.puff_counts = (
            pd.to_numeric(self.df["抽吸口数"], errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )
        self.area = (
            pd.to_numeric(self.df["area"], errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )

        n_rows = len(self.df)
        self.dynamic_features = np.zeros((n_rows, max_puffs, 6), dtype=np.float32)
        for i in range(1, max_puffs + 1):
            len_col = next(c for c in self.df.columns if c.startswith(f"第{i}口烟丝长"))
            self.dynamic_features[:, i - 1, 0] = self.df[len_col].fillna(0).values
            self.dynamic_features[:, i - 1, 1] = self.df[f"Q_cone/A_{i}"].fillna(0).values
            self.dynamic_features[:, i - 1, 2] = self.df[f"(Q_cone/A_{i})^2"].fillna(0).values
            self.dynamic_features[:, i - 1, 3] = self.df[f"V_rod_{i}"].fillna(0).values
            self.dynamic_features[:, i - 1, 4] = self.df[f"v_prefilter_{i}"].fillna(0).values
            self.dynamic_features[:, i - 1, 5] = self.df[f"v_postfilter_{i}"].fillna(0).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "idx": self.idx[idx],
            "static": torch.tensor(self.static_features[idx], dtype=torch.float32),
            "dynamic": torch.tensor(self.dynamic_features[idx], dtype=torch.float32),
            "type_idx": torch.tensor(self.types[idx], dtype=torch.long),
            "provided_factor": torch.tensor(self.provided_factor[idx], dtype=torch.float32),
            "puff_count": torch.tensor(self.puff_counts[idx], dtype=torch.float32),
            "area": torch.tensor(self.area[idx], dtype=torch.float32),
            "target": torch.tensor(self.target[idx], dtype=torch.float32),
        }
