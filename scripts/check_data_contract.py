#!/usr/bin/env python
"""Print dataset sizes, target counts, and Dataset 2 alignment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cigar.data import dataset_contract, load_paper_data


def main() -> None:
    data = load_paper_data(ROOT / "data" / "raw")
    print(json.dumps(dataset_contract(data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
