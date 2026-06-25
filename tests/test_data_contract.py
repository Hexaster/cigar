from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cigar.data import dataset_contract, load_paper_data


def test_data_contract_matches_current_files():
    data = load_paper_data(ROOT / "data" / "raw")
    contract = dataset_contract(data)

    assert contract["dataset1_rows"] == 885
    assert contract["dataset1_types"] == 15
    assert contract["dataset2_physical_rows"] == 253
    assert contract["dataset2_chemical_rows"] == 253
    assert contract["dataset2_chemical_features"] == 39
    assert contract["dataset2_aligned"] is True
    assert contract["dataset1_target_counts"]["CO,mg/支"] == 885
    assert contract["dataset1_target_counts"]["NH3,ug/支"] == 471
