"""Shared constants: targets, feature columns, and the global seed."""

from __future__ import annotations

TARGETS = [
    "CO,mg/支",
    "HCN,ug/支",
    "苯酚,ug/支",
    "巴豆醛,ug/支",
    "NH3,ug/支",
    "BaP,ng/支",
    "NNK,ng/支",
    "焦油,mg/支",
    "烟碱,mg/支",
]

PHYSICS_STATIC_COLS = [
    "卷烟圆周,cm",
    "通风孔前滤嘴,cm",
    "通风孔后滤嘴,cm",
    "烟丝段标况封闭吸阻,Pa/cm",
    "滤棒标况吸阻,Pa/cm",
]

# type_idx is attached by load_paper_data for CIGAR's type embedding; the
# monolithic baselines never use it, so drop it with the raw label column.
BASELINE_DROP_COLS = ["type", "type_idx", "标况嘴通风率"]

RANDOM_STATE = 42
