"""Two-stage CIGAR pipeline: mechanistic stage, then per-fold chemical factor stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from .constants import RANDOM_STATE, TARGETS
from .data import PaperData, preprocess_df
from .dataset import CigaretteDataset
from .metrics import regression_metrics
from .model import CIGARModel
from .reporting import format_fold_metrics, print_kfold_summary, print_type_embeddings
from .seed import fix_seed
from .tabpfn import make_tabpfn_regressor
from .training import collect_predictions, train_model


def train_mechanistic_stage(
    data: PaperData,
    target: str,
    epochs: int = 2000,
    lr: float = 1e-3,
    patience: int = 30,
    batch_size: int = 32,
    verbose: bool = False,
) -> CIGARModel:
    fix_seed(RANDOM_STATE)
    frame = preprocess_df(data.dataset1_physics, target_col=target, max_puffs=10)
    train_df, val_df = train_test_split(
        frame,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=frame["type"],
    )
    train_loader = DataLoader(CigaretteDataset(train_df, target, max_puffs=10), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CigaretteDataset(val_df, target, max_puffs=10), batch_size=batch_size, shuffle=False)
    model = CIGARModel(num_types=len(data.label_encoder.classes_))
    if verbose:
        print(f"Training model for target: {target}")
        print(f"Train size: {len(train_df)}, Test size: {len(val_df)}")
    return train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, patience=patience, verbose=verbose)


def neutral_predictions(model: CIGARModel, frame: pd.DataFrame, target: str, max_puffs: int) -> tuple[np.ndarray, np.ndarray]:
    proc = preprocess_df(frame, target_col=target, max_puffs=max_puffs).copy()
    proc["provided_factor"] = 1.0
    loader = DataLoader(CigaretteDataset(proc, target, max_puffs=max_puffs), batch_size=32, shuffle=False)
    return collect_predictions(model, loader)


def run_cigar_cv(
    data: PaperData,
    targets: list[str] | None = None,
    epochs: int = 2000,
    n_splits: int = 5,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = targets or TARGETS
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, object]] = []

    for target in targets:
        if verbose:
            print(f"\n{'=' * 60}\nTarget: {target}\n{'=' * 60}")
        model = train_mechanistic_stage(data, target, epochs=epochs, verbose=verbose)
        if verbose:
            print()
            print_type_embeddings(model, data.label_encoder)

        target_rows: list[dict[str, object]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.arange(len(data.dataset2_physical))), start=1):
            if verbose:
                print(f"\n--- CIGAR fold {fold_idx}/{n_splits} for {target} ---")
            physical_train = data.dataset2_physical.iloc[train_idx].copy()
            physical_test = data.dataset2_physical.iloc[test_idx].copy()
            chemical_train = data.dataset2_chemical.iloc[train_idx].copy()
            chemical_test = data.dataset2_chemical.iloc[test_idx].copy()

            train_neutral, train_targets = neutral_predictions(model, physical_train, target, max_puffs=15)
            factors = train_targets / np.clip(train_neutral, 1e-8, None)
            factor_model = make_tabpfn_regressor(random_state=RANDOM_STATE)
            factor_model.fit(chemical_train, factors)
            predicted_factors = factor_model.predict(chemical_test)

            test_proc = preprocess_df(physical_test, target_col=target, max_puffs=15).copy()
            test_proc["provided_factor"] = predicted_factors
            test_loader = DataLoader(CigaretteDataset(test_proc, target, max_puffs=15), batch_size=32, shuffle=False)
            predictions, y_true = collect_predictions(model, test_loader)
            metrics = regression_metrics(y_true, predictions)
            row = {"Target": target, "Model": "CIGAR", "Fold": fold_idx, **metrics}
            rows.append(row)
            target_rows.append(row)
            if verbose:
                print(format_fold_metrics("CIGAR", target, fold_idx, n_splits, metrics))
        if verbose:
            print_kfold_summary("CIGAR", target, n_splits, target_rows)

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["Target", "Model"])[["MAE", "MSE", "MAPE", "R2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    return detail, summary
