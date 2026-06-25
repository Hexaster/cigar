# CIGAR

Code and data for the paper:

**CIGAR: A Combustion-Informed Physics-Based Model for Interpretable Prediction of Cigarette Mainstream Smoke Constituents**

This is a repository for the paper, not the project repository itself.

## Layout

```text
data/raw/                         the three Excel workbooks
paper/                            manuscript source, figures, tables
src/cigar/                        the CIGAR model and evaluation code
scripts/check_data_contract.py    print dataset sizes and target counts
scripts/run_paper_experiments.py  CIGAR / XGBoost / TabPFN, 5-fold
scripts/make_mechanism_report.py  rebuild the JH-7 CO mechanism table
results/                          output goes here
```

## Data

- Dataset 1 (physical properties): `data/raw/data202605.xlsx`
- Dataset 2 (physical descriptors): `data/raw/data_chemical_factor.xlsx`, sheet `计算结果`
- Dataset 2 (chemical composition): `data/raw/data20260120.xlsx`, sheet `X3`

Dataset 2 has 253 aligned samples (`JH-1` to `JH-262`, with the missing IDs dropped).

## Setup

Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The full comparison needs TabPFN. The access token is read from the environment, so set it first if your client requires one:

```bash
export TABPFN_API_TOKEN="your-token"
```

## Running the experiments

Check that the datasets load and line up:

```bash
python scripts/check_data_contract.py
```

Rebuild the JH-7 CO mechanism trajectory table (writes `results/mechanism_jh7_co.csv` and `.tex`):

```bash
python scripts/make_mechanism_report.py
```

Run the model comparison across all targets:

```bash
python scripts/run_paper_experiments.py
```

By default it prints progress as it runs: the type mapping, dataset sizes, per-epoch CIGAR metrics and physical parameters, per-fold metrics for each model, and the KFold summary per target. Pass `--quiet` to skip the progress output and just write the result CSVs. To try a single model on a single target:

```bash
python scripts/run_paper_experiments.py --models XGBoost --targets "CO,mg/支"
```

A full CIGAR run trains one mechanistic model per target, then runs 5-fold cross-validation on Dataset 2. Runtime depends on your hardware and on TabPFN access.

## Evaluation setup

- Seed 42 throughout.
- Dataset 1 is split 80:20 (train/validation), stratified by `type`.
- Dataset 2 uses 5-fold `sklearn.KFold(shuffle=True, random_state=42)`.
- XGBoost trains on a log-transformed target with the tuned hyperparameters in `monolithic_xgb_params.json`.
- CIGAR uses the raw physical and tobacco-composition measurements, with no feature scaling.

The `paper/` directory holds the manuscript source, bibliography, and the figures and tables it references.
