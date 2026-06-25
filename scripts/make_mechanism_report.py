#!/usr/bin/env python
"""Regenerate the paper's JH-7 CO mechanism table as CSV and LaTeX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cigar.data import load_paper_data
from cigar.mechanism import paper_co_jh7_model, trajectory_dataframe, write_mechanism_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="JH-7")
    parser.add_argument("--target", default="CO,mg/支")
    parser.add_argument("--factor", type=float, default=0.8333)
    parser.add_argument("--output-prefix", default=str(ROOT / "results" / "mechanism_jh7_co"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_paper_data(ROOT / "data" / "raw")
    model = paper_co_jh7_model(num_types=len(data.label_encoder.classes_))
    table = trajectory_dataframe(data, model, args.target, args.sample, args.factor, max_puffs=15)
    csv_path, tex_path = write_mechanism_outputs(table, args.output_prefix)
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(
        f"Predicted={table['predicted_total'].iloc[0]:.4f}; "
        f"Observed={table['observed_total'].iloc[0]:.4f}"
    )


if __name__ == "__main__":
    main()
