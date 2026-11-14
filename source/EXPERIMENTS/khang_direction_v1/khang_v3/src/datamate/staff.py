"""Ten nhan vien khi so ban hang chi ghi ma (NV001): tra ho ten tu bang danh sach nhan vien.

Tinh khi sandbox khoi dong va khi dung bo nhan dien ten (nhanh, khong can cache).
"""

from __future__ import annotations

import re

import pandas as pd

from .loader import _split_name

_NAME_COLS = ("ten goi", "ho ten", "ten", "ten nhan vien", "ho va ten", "ten nv", "nhan vien")


def _n(c) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(str(c))).split())


def employee_names(tables: dict, codes: pd.Series | None) -> dict[str, str] | None:
    """{ma nhan vien: ho ten} neu cot employee la ma (NV001…) va co bang nhan vien (moi ma mot dong, co cot ten)."""
    if codes is None:
        return None
    vals = set(codes.dropna().astype(str))
    if not vals or sum(bool(re.fullmatch(r"[A-Za-z]{1,5}[-_]?\d{1,6}", v)) for v in vals) < 0.8 * len(vals):
        return None
    for df in tables.values():
        for c in df.columns:
            if str(c).startswith("_") or df[c].dtype != object:
                continue
            col = df[c].dropna().astype(str)
            if len(col) == 0 or not col.is_unique or col.isin(vals).mean() < 0.5 or not vals <= set(col):
                continue
            name = next((n for n in df.columns if n != c and _n(n) in _NAME_COLS and df[n].dtype == object), None)
            if name:
                d = df.dropna(subset=[c, name])
                return dict(zip(d[c].astype(str), d[name].astype(str).str.strip()))
    return None
