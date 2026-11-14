"""So giao hang nhieu dot (rulebook muc "Giao nhieu dot"): bang D_giao va ham dem dot giao.

- Moi dong so giao: mot mat hang cua mot dot (Luot giao 1, 2, 3) cua mot don, ngay giao, so luong giao (don vi co ban).
- So dot giao trong thang X = so (don, dot) co ngay giao trong thang X; moi dot cua moi don tinh mot dot, ke ca dot cua don
  sau do tra hang. Tinh den het hom nay (dot co ngay giao sau hom nay coi nhu chua giao).
Tinh khi sandbox khoi dong tu bang goc (nhanh), khong can cache.
"""

from __future__ import annotations

import re

import pandas as pd

from .loader import _split_name


def _n(c) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(str(c))).split())


def build_delivery(tables: dict, O: pd.DataFrame | None) -> pd.DataFrame | None:
    known = set(O["order_id"].astype(str)) if O is not None and len(O) else set()
    for t, df in tables.items():
        cols = {c: _n(c) for c in df.columns if not str(c).startswith("_")}
        dot = next((c for c, v in cols.items() if v in ("luot giao", "dot giao", "dot", "lan giao", "chuyen", "dot cho")), None)
        if dot is None:
            continue
        # ma don: theo ten, hoac cot co gia tri trung ma don cua O
        oid = next((c for c, v in cols.items() if v in ("so don", "ma don", "ma don hang", "so don hang", "ma bill", "so bill", "ma hoa don", "so hd")), None)
        if oid is None and known:
            oid = next((c for c in cols if df[c].dtype == object and df[c].dropna().astype(str).isin(known).mean() > 0.8), None)
        dts = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        day = next((c for c in dts if any(k in cols[c] for k in ("giao", "cho", "van chuyen"))), dts[0] if dts else None)
        if not (oid and day):
            continue
        D = pd.DataFrame({"order_id": df[oid].astype(str).values, "dot": pd.to_numeric(df[dot], errors="coerce").values,
                          "ngay_giao": pd.to_datetime(df[day], errors="coerce").dt.normalize().values})
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != dot]
        roles = {
            "item": next((c for c, v in cols.items() if df[c].dtype == object and any(k in v for k in ("hang hoa", "ten hang", "mat hang", "san pham"))), None),
            "qty_giao": next((c for c in num if "giao" in cols[c] or cols[c].startswith(("sl", "so luong"))), None),
            "xe": next((c for c, v in cols.items() if "bien so" in v or v == "xe"), None),
            "lai_xe": next((c for c, v in cols.items() if any(k in v for k in ("lai xe", "tai xe", "nguoi cho", "nguoi giao"))), None),
        }
        for role, c in roles.items():
            if c is not None:
                D[role] = df[c].values
        if known:
            m = O.set_index(O["order_id"].astype(str))
            D["order_date"] = D["order_id"].map(m["order_date"])
            D["status"] = D["order_id"].map(m["status"])
        return D.dropna(subset=["ngay_giao"]).reset_index(drop=True)
    return None


def delivery_rounds(D: pd.DataFrame, a, b, today) -> int:
    """So dot giao (don, dot) co ngay giao trong [a, b] va khong sau hom nay; ke ca dot cua don sau do tra hang."""
    from .dates import to_date

    lo, hi = pd.Timestamp(to_date(a)), min(pd.Timestamp(to_date(b)), pd.Timestamp(to_date(today)))
    d = D[(D["ngay_giao"] >= lo) & (D["ngay_giao"] <= hi)]
    return int(d[["order_id", "dot"]].drop_duplicates().shape[0])


def describe(D: pd.DataFrame) -> str:
    return "\n".join([
        "## Sổ giao hàng dựng sẵn (rulebook mục Giao nhiều đợt)",
        f"- D_giao: {len(D)} dòng (mỗi dòng một mặt hàng của một đợt giao), cột {', '.join(D.columns)}; order_date, status lấy từ O.",
        "- delivery_rounds(kỳ) -> số đợt giao (đơn, đợt) có ngày giao trong kỳ, tính đến hết hôm nay, kể cả đợt của đơn sau đó trả hàng.",
        "- Số đợt đã giao của một đơn tới hôm nay = số giá trị dot khác nhau có ngay_giao <= hôm nay; số lượng đã giao = tổng qty_giao (đơn vị cơ bản).",
    ])
