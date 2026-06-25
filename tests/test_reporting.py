from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cigar.reporting import (
    format_fold_metrics,
    format_metric_summary,
    format_physical_params,
    format_type_mapping,
)


class FakeParam:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.value


class FakeModel:
    def get_physical_params(self):
        return {"G1": FakeParam(-0.000151), "G2": FakeParam(0.132353)}


class FakeEncoder:
    classes_ = ["FC", "FJ"]


def test_format_type_mapping_matches_notebook_prefix():
    assert format_type_mapping(FakeEncoder()) == "全局Type映射关系: {'FC': 0, 'FJ': 1}"


def test_format_physical_params_matches_notebook_style():
    assert format_physical_params(FakeModel()) == "G1: -0.0002, G2: 0.1324"


def test_format_fold_metrics_matches_notebook_style():
    metrics = {"MSE": 1.23456, "MAE": 0.98765, "MAPE": 0.12345, "R2": 0.54321}
    assert (
        format_fold_metrics("TabPFN", "CO,mg/支", 2, 5, metrics)
        == "TabPFN - CO,mg/支 fold 2/5: MSE=1.2346, MAE=0.9877, MAPE=0.1235, R2=0.5432"
    )


def test_format_metric_summary_matches_notebook_style():
    assert format_metric_summary("MAE", 0.8849, 0.0632) == "  MAE: mean=0.8849 std=0.0632"
