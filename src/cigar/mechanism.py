"""Mechanism trajectory export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import PaperData, preprocess_df
from .dataset import CigaretteDataset
from .model import CIGARModel, FixedParamCIGARModel


PAPER_CO_JH7_PARAMS = {
    "G1": -1.53e-4,
    "G2": 0.1386,
    "G3": 1.0292,
    "a_init": 0.5024,
    "m_rod": -0.2674,
    "n_rod": 1.4157,
    "p_rod": 1.6315,
    "m_fil": -0.2206,
    "n_fil": 1.5589,
    "p_fil": 1.5559,
}


def trajectory_dataframe(
    data: PaperData,
    model: CIGARModel,
    target: str,
    sample_id: str,
    factor: float,
    max_puffs: int = 15,
) -> pd.DataFrame:
    frame = preprocess_df(data.dataset2_physical, target_col=target, max_puffs=max_puffs)
    if sample_id not in frame.index:
        raise ValueError(f"Sample {sample_id!r} is not present after preprocessing.")
    sample = frame.loc[[sample_id]].copy()
    sample["provided_factor"] = factor
    loader = DataLoader(CigaretteDataset(sample, target, max_puffs=max_puffs), batch_size=1, shuffle=False)
    batch = next(iter(loader))

    model.eval()
    rows = []
    with torch.no_grad():
        total, aux = model(batch, return_aux=True)
        for puff in aux:
            for segment_name in ["rod", "pre", "post"]:
                V, k1, k2, k3, a_out, eta = puff[segment_name]
                rows.append(
                    {
                        "sample": sample_id,
                        "target": target,
                        "k_G": factor,
                        "puff": puff["puff"],
                        "segment": segment_name,
                        "V": float(V.item()),
                        "k1": float(k1.item()),
                        "k2": float(k2.item()),
                        "k3": float(k3.item()),
                        "a_out": float(a_out.item()),
                        "eta": float(eta.item()),
                        "G_j": float(puff["generation"].item()) if segment_name == "rod" else None,
                        "yield_j": float(puff["yield"].item()) if segment_name == "rod" else None,
                        "cumulative": float(puff["cumulative"].item()) if segment_name == "rod" else None,
                        "predicted_total": float(total.item()),
                        "observed_total": float(batch["target"].item()),
                    }
                )
    return pd.DataFrame(rows)


def paper_co_jh7_model(num_types: int = 15) -> FixedParamCIGARModel:
    return FixedParamCIGARModel(PAPER_CO_JH7_PARAMS, num_types=num_types)


def write_mechanism_outputs(table: pd.DataFrame, output_prefix: str | Path) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    tex_path = output_prefix.with_suffix(".tex")
    table.to_csv(csv_path, index=False)
    table.to_latex(tex_path, index=False, float_format="%.4f")
    return csv_path, tex_path
