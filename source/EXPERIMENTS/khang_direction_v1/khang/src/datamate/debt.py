"""Cong no theo rulebook §4, dung san tu README (khong dung model).

Nguon: dong quy uoc README '**Cong no**: moi lan khach tra tien cho mot don la mot dong, **X** la so tien tra
lan do; **Y** la so khach phai tra cho ca don' (hoac `bang.cot`). Can them bang chuan O (semantic.py) de co
ngay dat, trang thai, khach, hinh thuc thanh toan, kenh.

Ham dung trong sandbox:
  debt_as_of(ngay)            -> moi don con no > 0 tai het ngay do (order_id, order_date, khach, con_no, tuoi_no)
  debt_by_customer(ngay)      -> no theo khach (bo khach le), sap xep giam dan
  debt_added(tu, den)         -> no them trong ky (phan chua tra ngay trong ngay mua cua don ghi no dat trong ky)
  debt_paid(tu, den)          -> tien tra no trong ky (cac lan tra sau ngay mua, ngay tra trong ky)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .loader import META_COLS, Business, _split_name
from .parsing import norm_key

_RE_PAY = re.compile(r"(?:\*\*([^*]+)\*\*|`([^`]+)`)\s*là số tiền trả lần đó")
_RE_DUE = re.compile(r"(?:\*\*([^*]+)\*\*|`([^`]+)`)\s*là số khách phải trả cho cả đơn")
_EXCLUDED_CHANNEL = ("san tmdt", "san thuong mai", "shopee", "lazada", "tiki", "tiktok", "sendo", "ecommerce", "marketplace")


@dataclass
class DebtModel:
    orders: pd.DataFrame  # order_id, order_date, status, customer..., total_due, debt_eligible
    payments: pd.DataFrame  # order_id, paid_at, paid_date, amount
    mapping: dict[str, str]
    notes: list[str]


def _locate(biz: Business, ref: str) -> tuple[str, str] | None:
    """'bang.cot' hoac 'cot' -> (bang, cot)."""
    if "." in ref:
        t, c = ref.split(".", 1)
        for name in biz.tables:
            if name == t or name.lower() == t.lower():
                if c in biz.tables[name].columns:
                    return name, c
    hits = [(t, ref) for t, df in biz.tables.items() if ref in df.columns]
    return hits[0] if hits else None


def _date_col(biz: Business, table: str, exclude: str) -> str | None:
    df = biz.tables[table]
    cands = [c for c in df.columns if c != exclude and biz.info[table].columns.get(c) and biz.info[table].columns[c].kind in ("datetime", "date")]
    if not cands:
        return None
    cands.sort(key=lambda c: (biz.info[table].columns[c].kind != "datetime", "thu" not in _split_name(c)))
    return cands[0]


def _order_key(biz: Business, table: str, ids: set[str], exclude: str) -> str | None:
    df = biz.tables[table]
    best = None
    for c in df.columns:
        if c in META_COLS or c == exclude:
            continue
        v = df[c].dropna().astype(str)
        if v.empty:
            continue
        ov = v.isin(ids).mean()
        if ov > 0.5 and (best is None or ov > best[0]):
            best = (ov, c)
    return best[1] if best else None


def build_debt(biz: Business, sales) -> DebtModel | None:
    if sales is None or sales.O is None or not len(sales.O):
        return None
    text = biz.readme.text
    mp, md = _RE_PAY.search(text), _RE_DUE.search(text)
    if not mp or not md:
        return None
    pay = _locate(biz, mp.group(1) or mp.group(2))
    due = _locate(biz, md.group(1) or md.group(2))
    if pay is None or due is None:
        return None
    O = sales.O.copy()
    ids = set(O["order_id"].astype(str))
    # tong khach phai tra cua don
    dt_, dc = due
    dkey = _order_key(biz, dt_, ids, dc)
    if dkey is None:
        return None
    dd = biz.tables[dt_][[dkey, dc]].dropna(subset=[dkey]).drop_duplicates(subset=[dkey])
    O["total_due"] = O["order_id"].astype(str).map(dict(zip(dd[dkey].astype(str), pd.to_numeric(dd[dc], errors="coerce"))))
    # cac lan tra
    pt, pc = pay
    pkey = _order_key(biz, pt, ids, pc)
    pdate = _date_col(biz, pt, pc)
    if pkey is None or pdate is None:
        return None
    P = pd.DataFrame({
        "order_id": biz.tables[pt][pkey].astype(str),
        "paid_at": pd.to_datetime(biz.tables[pt][pdate], errors="coerce"),
        "amount": pd.to_numeric(biz.tables[pt][pc], errors="coerce"),
    }).dropna(subset=["paid_at", "amount"])
    P["paid_date"] = P["paid_at"].dt.normalize()
    notes = []
    # don khong phai khach no: huy, COD, ban qua san TMDT (§4)
    excl = O["status"].eq("huỷ")
    if "payment" in O:
        pm = O["payment"].map(norm_key)
        excl |= pm.str.contains(r"\bcod\b|giao hàng thu tiền|thu hộ", regex=True, na=False)
    if "channel" in O:
        ch = O["channel"].map(lambda v: norm_key(v) if isinstance(v, str) else "")
        from .parsing import strip_accents

        excl |= ch.map(strip_accents).map(lambda s: any(k in s for k in _EXCLUDED_CHANNEL))
    O["debt_eligible"] = ~excl
    # don ghi no: hinh thuc thanh toan ghi no, hoac chua tra du ngay trong ngay mua
    paid_same_day = P.merge(O[["order_id", "order_date"]].astype({"order_id": str}), on="order_id", how="left")
    paid_same_day = paid_same_day[paid_same_day["paid_date"] <= paid_same_day["order_date"]].groupby("order_id")["amount"].sum()
    O["paid_on_purchase_day"] = O["order_id"].astype(str).map(paid_same_day).fillna(0.0)
    on_credit = (O["total_due"] - O["paid_on_purchase_day"]) > 0.5
    if "payment" in O:
        on_credit |= O["payment"].map(norm_key).str.contains(r"debt|ghi nợ|công nợ|nợ|trả góp", regex=True, na=False)
    O["on_credit"] = on_credit & O["debt_eligible"]
    mapping = {
        "total_due": f"{dt_}.{dc} (nối qua {dkey})",
        "payments": f"{pt}.{pc} theo ngày {pt}.{pdate} (nối qua {pkey})",
    }
    return DebtModel(orders=O, payments=P, mapping=mapping, notes=notes)


def _to_ts(x):
    return pd.Timestamp(str(x)[:10])


def debt_as_of(model: DebtModel, day) -> pd.DataFrame:
    D = _to_ts(day)
    O = model.orders
    sel = O["debt_eligible"] & (O["order_date"] <= D) & O["total_due"].notna()
    paid = model.payments[model.payments["paid_date"] <= D].groupby("order_id")["amount"].sum()
    out = O.loc[sel].copy()
    out["paid_to_date"] = out["order_id"].astype(str).map(paid).fillna(0.0)
    out["con_no"] = out["total_due"] - out["paid_to_date"]
    out = out[out["con_no"] > 0.5]
    out["tuoi_no"] = (D - out["order_date"]).dt.days
    cols = [c for c in ("order_id", "order_date", "customer", "customer_key", "customer_id", "is_walk_in", "status", "total_due", "paid_to_date", "con_no", "tuoi_no") if c in out]
    return out[cols].sort_values("con_no", ascending=False).reset_index(drop=True)


def debt_by_customer(model: DebtModel, day) -> pd.DataFrame:
    d = debt_as_of(model, day)
    if "is_walk_in" in d:
        d = d[~d["is_walk_in"].fillna(False)]
    key = "customer_id" if "customer_id" in d and d["customer_id"].notna().any() else "customer_key"
    if key not in d:
        return pd.DataFrame(columns=["khach", "con_no", "so_don"])
    name_col = "customer" if "customer" in d else key
    g = d.groupby(key).agg(khach=(name_col, "first"), con_no=("con_no", "sum"), so_don=("order_id", "nunique")).reset_index()
    return g.sort_values(["con_no", key], ascending=[False, True]).reset_index(drop=True)


def debt_added(model: DebtModel, start, end) -> pd.DataFrame:
    a, b = _to_ts(start), _to_ts(end)
    O = model.orders
    sel = O["on_credit"] & (O["order_date"] >= a) & (O["order_date"] <= b)
    out = O.loc[sel].copy()
    out["no_them"] = (out["total_due"] - out["paid_on_purchase_day"]).clip(lower=0)
    return out


def debt_paid(model: DebtModel, start, end) -> pd.DataFrame:
    a, b = _to_ts(start), _to_ts(end)
    O = model.orders.set_index(model.orders["order_id"].astype(str))
    P = model.payments.copy()
    P = P[P["order_id"].isin(O.index[O["on_credit"]])]
    P["order_date"] = P["order_id"].map(O["order_date"])
    P = P[(P["paid_date"] > P["order_date"]) & (P["paid_date"] >= a) & (P["paid_date"] <= b)]
    for c in ("customer", "customer_key", "customer_id"):
        if c in O:
            P[c] = P["order_id"].map(O[c])
    return P


def describe(model: DebtModel) -> str:
    O = model.orders
    return "\n".join([
        "## Công nợ dựng sẵn (§4): hàm debt_as_of(ngày), debt_by_customer(ngày), debt_added(từ, đến), debt_paid(từ, đến)",
        "- debt_as_of(D): mỗi đơn còn nợ > 0 tính đến hết ngày D (cột con_no, tuoi_no = số ngày từ ngày đặt; đã bỏ đơn huỷ, đơn COD, đơn bán qua sàn TMĐT; đơn trả hàng vẫn tính nếu còn nợ). Tổng nợ = .con_no.sum() (gồm cả khách lẻ).",
        "- debt_by_customer(D): nợ theo khách (bỏ khách lẻ), cột khach, con_no, so_don, giảm dần. Số khách còn nợ = len(...).",
        "- debt_added(a, b): đơn ghi nợ đặt trong kỳ, cột no_them = phần chưa trả ngay trong ngày mua. debt_paid(a, b): các lần trả nợ SAU ngày mua có ngày trả trong kỳ (cột amount).",
        "- Nguồn: " + "; ".join(f"{k} ← {v}" for k, v in model.mapping.items())
        + f"; {int(O['on_credit'].sum())} đơn ghi nợ / {len(O)} đơn.",
    ])
