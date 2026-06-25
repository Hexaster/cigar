#!/usr/bin/env python
"""Run the paper's 5-fold Dataset 2 model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cigar.baselines import run_baseline_cv
from cigar.constants import TARGETS
from cigar.data import dataset_contract, load_paper_data
from cigar.pipeline import run_cigar_cv
from cigar.reporting import format_type_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["CIGAR", "XGBoost", "TabPFN"], choices=["CIGAR", "XGBoost", "TabPFN"])
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--cigar-epochs", type=int, default=2000)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--quiet", action="store_true", help="Suppress notebook-style progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_paper_data(ROOT / "data" / "raw")
    verbose = not args.quiet

    if verbose:
        contract = dataset_contract(data)
        print(format_type_mapping(data.label_encoder))
        print(
            f"Dataset 1 rows: {contract['dataset1_rows']}; "
            f"Dataset 2 aligned rows: {contract['dataset2_physical_rows']}; "
            f"chemical features: {contract['dataset2_chemical_features']}"
        )
        print(f"Models: {', '.join(args.models)}")
        print(f"Targets: {', '.join(args.targets)}")

    detail_frames = []
    summary_frames = []
    if "CIGAR" in args.models:
        detail, summary = run_cigar_cv(data, targets=args.targets, epochs=args.cigar_epochs, verbose=verbose)
        detail_frames.append(detail)
        summary_frames.append(summary)
    for model_name in ["XGBoost", "TabPFN"]:
        if model_name in args.models:
            detail, summary = run_baseline_cv(
                data,
                model_name=model_name,
                targets=args.targets,
                xgb_params_path=ROOT / "monolithic_xgb_params.json",
                verbose=verbose,
            )
            detail_frames.append(detail)
            summary_frames.append(summary)

    detail_all = pd.concat(detail_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    detail_path = output_dir / "paper_model_comparison_detail.csv"
    summary_path = output_dir / "paper_model_comparison_summary.csv"
    detail_all.to_csv(detail_path, index=False)
    summary_all.to_csv(summary_path, index=False)
    print("\n--- Paper Model Comparison Summary ---")
    print(summary_all.to_string(index=False))
    print(f"\nSaved detail -> {detail_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
