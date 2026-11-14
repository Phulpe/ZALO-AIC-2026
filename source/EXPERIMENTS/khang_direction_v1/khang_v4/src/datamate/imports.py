"""Bang chuan nhap hang N (moi dong mot dong hang nhap), dung tu README (khong dung model).

Cot: receipt_id, date, item, item_code, unit, qty, unit_price, line_total, supplier, cancelled.
Rulebook §5: tien nhap trong ky = tong tien cac phieu co ngay phieu trong ky, bo phieu huy;
gia nhap binh quan = binh quan theo so luong nhap (tong tien / tong so luong), theo don vi nhap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .loader import META_COLS, Business, _split_name
from .parsing import compact_key

ROLES = {
    "receipt_id": (("mã phiếu nhập",), ("ma phieu nhap", "so phieu", "ma nhap hang", "ma don nhap", "so phieu nhap", "maphieunhap", "phieu nhap")),
    "date": (("ngày nhập",), ("ngay nhap", "ngay nhap kho", "thoi gian", "ngay")),
    "item": (("tên hàng",), ("ten hang", "product", "hang", "hang hoa", "ten mat hang", "ten san pham", "mat hang")),
    "item_code": (("mã hàng",), ("ma hang", "ma sku", "ma hang hoa", "ma sp", "ma mon", "ma hh")),
    "unit": (("đơn vị nhập", "đơn vị tính"), ("dvt nhap", "dvt", "don vi tinh", "don vi nhap", "dv nhap", "don vi")),
    "qty": (("số lượng nhập", "số lượng"), ("sl nhap", "so luong", "sl", "so luong nhap")),
    "unit_price": (("giá nhập một đơn vị",), ("gia nhap", "don gia", "don gia nhap", "dongianhap")),
    "line_total": (("tiền nhập của cả dòng",), ("thanh tien",)),
    "receipt_total": (("tổng tiền phiếu nhập", "tổng tiền cả phiếu", "tổng tiền của cả phiếu"), ("tong tien phieu", "tong gia tri nhap", "can tra ncc", "tong tien")),
    "supplier": (("nhà cung cấp", "mã nhà cung cấp"), ("nha cung cap", "ncc", "ma ncc", "ma nha cung cap")),
    "status": (("trạng thái phiếu nhập",), ("trang thai", "tinh trang", "trang thai phieu")),
    "note": (("ghi chú",), ("ghi chu", "note", "luu y")),
}


@dataclass
class ImportModel:
    N: pd.DataFrame
    receipts: pd.DataFrame
    mapping: dict[str, str]
    notes: list[str]


def _desc(biz, t, c):
    ci = biz.info[t].columns.get(c)
    return (ci.description or "").strip().lower() if ci else ""


def _kind(biz, t, c):
    ci = biz.info[t].columns.get(c)
    return ci.kind if ci else "text"


def _norm(c):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(c)).split())


_CODE = re.compile(r"^[A-Za-z]{1,6}[-_]?\d{1,6}$")


def _supplier_names(biz: Business, N: pd.DataFrame, mapping: dict) -> None:
    """Phieu nhap chi ghi ma NCC (NCC002): noi sang ten trong danh sach nha cung cap, giu ma o supplier_code."""
    vals = N["supplier"].dropna().astype(str)
    if vals.empty or vals.map(lambda v: bool(_CODE.match(v.strip()))).mean() < 0.9:
        return
    want = set(vals.str.strip())
    for t, df in biz.tables.items():
        for cc in df.columns:
            if cc in META_COLS:
                continue
            v = df[cc].dropna().astype(str).str.strip()
            if v.empty or v.duplicated().any() or pd.Series(list(want)).isin(set(v)).mean() < 0.8:
                continue
            texts = [c for c in df.columns if c != cc and c not in META_COLS and df[c].dtype == object
                     and df[c].dropna().astype(str).map(lambda x: not _CODE.match(x.strip())).mean() > 0.9
                     and df[c].nunique() >= 0.8 * len(v)]
            pref = [c for c in texts if any(k in _norm(c).split() for k in ("ten", "name", "tengoi"))
                    or _norm(c).startswith(("ten", "nha cung cap", "ncc"))]
            pref = [c for c in pref if not any(k in _norm(c) for k in ("dien thoai", "dia chi", "email", "sdt", "lien he"))]
            name = (pref or [None])[0]
            if name is None:
                continue
            m = df.assign(_k=df[cc].astype(str).str.strip()).drop_duplicates("_k").set_index("_k")[name]
            N["supplier_code"] = N["supplier"]
            N["supplier"] = N["supplier"].astype(str).str.strip().map(m).fillna(N["supplier"])
            mapping["supplier"] = f"{mapping.get('supplier', '')} -> {t}.{name} (nối qua mã NCC)"
            return


def _roles(biz: Business, t: str) -> dict[str, str]:
    df = biz.tables[t]
    out: dict[str, str] = {}
    for role, (descs, names) in ROLES.items():
        for c in df.columns:
            if c in META_COLS or c in out.values():
                continue
            d = _desc(biz, t, c)
            if d and any(d.startswith(p) for p in descs):
                if role == "qty" and "đổi ra" in d:
                    continue
                out[role] = c
                break
        if role in out:
            continue
        for c in df.columns:
            if c in META_COLS or c in out.values() or _desc(biz, t, c):
                continue
            if _norm(c) in names:
                out[role] = c
                break
    for role in ("qty", "unit_price", "line_total", "receipt_total"):
        if role in out and _kind(biz, t, out[role]) != "number":
            out.pop(role)
    if "date" in out and _kind(biz, t, out["date"]) not in ("date", "datetime"):
        out.pop("date")
    return out


def _is_import(biz, t, r) -> bool:
    descs = " ".join(_desc(biz, t, c) for c in biz.tables[t].columns)
    names = " ".join(_norm(c) for c in biz.tables[t].columns) + " " + _norm(t)
    return "phiếu nhập" in descs or "giá nhập" in descs or "nhap" in names.split() or "ngay nhap" in names or "gia nhap" in names


def build_imports(biz: Business) -> ImportModel | None:
    roles = {t: _roles(biz, t) for t in biz.tables}
    lines = [(t, r) for t, r in roles.items() if _is_import(biz, t, r) and "receipt_id" in r and ("item" in r or "item_code" in r) and "qty" in r]
    if not lines:
        return None
    lt, lr = max(lines, key=lambda x: len(x[1]))
    L = biz.tables[lt]
    N = pd.DataFrame({k: L[v] for k, v in lr.items()})
    mapping = {k: f"{lt}.{v}" for k, v in lr.items()}
    notes = []
    # dau phieu o bang khac (ngay, nha cung cap, trang thai, tong tien phieu)
    ids = set(N["receipt_id"].dropna().astype(str))
    for t, r in roles.items():
        if t == lt or "receipt_id" not in r:
            continue
        head = biz.tables[t]
        key = head[r["receipt_id"]].astype(str)
        if key.duplicated().any() or key.isin(ids).mean() < 0.5:
            continue
        h = head.set_index(key)
        for role in ("date", "supplier", "status", "receipt_total", "note"):
            if role in r and role not in N:
                N[role] = N["receipt_id"].astype(str).map(h[r[role]])
                mapping[role] = f"{t}.{r[role]} (nối qua mã phiếu)"
        break
    if "item" not in N and "item_code" in N:
        for t, r in roles.items():
            pass
        for t, df in biz.tables.items():
            if t == lt:
                continue
            code = next((c for c in df.columns if df[c].astype(str).isin(set(N["item_code"].astype(str))).mean() > 0.8 and not df[c].duplicated().any()), None)
            if code is None:
                continue
            # cot ten hang: chu, gan nhu moi ma mot ten; uu tien 'Ten ...'; khong lay 'Nhom hang', 'Loai hang'
            n_codes = max(1, df[code].nunique())
            texts = [c for c in df.columns if c != code and _kind(biz, t, c) == "text" and df[c].nunique() >= 0.8 * n_codes
                     and not any(x in _norm(c).split() for x in ("nhom", "loai", "nganh", "danh", "muc", "vi", "tri", "thuong", "hieu"))]
            pref = [c for c in texts if _norm(c).startswith("ten") or _norm(c) in ("mat hang", "san pham", "hang hoa", "mon")]
            name = (pref or [c for c in texts if any(k in _norm(c).split() for k in ("ten", "hang", "mon", "pham"))] or [None])[0]
            if name:
                N["item"] = N["item_code"].map(df.drop_duplicates(subset=[code]).set_index(code)[name])
                mapping["item"] = f"{t}.{name} (nối qua mã hàng)"
                break
    if "supplier" in N:
        _supplier_names(biz, N, mapping)
    if "date" not in N:
        return None
    N["date"] = pd.to_datetime(N["date"], errors="coerce").dt.normalize()
    if "line_total" not in N and "unit_price" in N:
        N["line_total"] = N["qty"] * N["unit_price"]
        mapping["line_total"] = "tính = qty × unit_price"
    if "line_total" not in N:
        N["line_total"] = float("nan")
        notes.append("shop không ghi tiền nhập theo dòng (line_total trống): xem bảng gốc")
    canc = pd.Series(False, index=N.index)
    for c in ("status", "note"):
        if c in N:
            canc |= N[c].astype(str).str.lower().str.contains("huỷ|hủy|huy|cancel", regex=True, na=False)
    N["cancelled"] = canc
    if canc.any():
        notes.append(f"{int(N.loc[canc, 'receipt_id'].nunique())} phiếu nhập huỷ (cancelled=True, bỏ khi tính)")
    if "item" in N:
        N["item_key"] = N["item"].map(compact_key)
    R = N.groupby("receipt_id", sort=False).agg(
        date=("date", "first"),
        cancelled=("cancelled", "any"),
        tien_nhap=("line_total", lambda x: x.sum(min_count=1)),
        **({"supplier": ("supplier", "first")} if "supplier" in N else {}),
        **({"receipt_total": ("receipt_total", "first")} if "receipt_total" in N else {}),
    ).reset_index()
    return ImportModel(N=N.reset_index(drop=True), receipts=R, mapping=mapping, notes=notes)


def describe(model: ImportModel) -> str:
    N, R = model.N, model.receipts
    return "\n".join([
        "## Nhập hàng dựng sẵn: `N` (mỗi dòng một dòng hàng nhập) và `R` (mỗi phiếu nhập) — dùng cho câu hỏi nhập hàng (§5)",
        "- N: receipt_id, date (ngày phiếu), item, unit (đơn vị nhập), qty, unit_price (giá nhập 1 đơn vị nhập), line_total (tiền nhập cả dòng), supplier, cancelled (phiếu huỷ).",
        "- Shop không ghi tiền/giá nhập (line_total toàn NaN) hoặc mặt hàng không có dòng nhập trong kỳ -> unanswerable, không trả 0.",
        "- Tiền nhập trong kỳ = N[~cancelled & trong_kỳ].line_total.sum() (hoặc R.tien_nhap). Giá nhập bình quân 1 <đơn vị> của mặt hàng trong kỳ = tổng line_total / tổng qty của các dòng mặt hàng đó với unit = <đơn vị>, bỏ phiếu huỷ (bình quân theo số lượng, không phải trung bình các mức giá), làm tròn về đồng.",
        "- Nhà cung cấp nhập nhiều nhất = supplier có tổng tiền nhập cao nhất; nhà cung cấp chính = supplier có nhiều phiếu nhập nhất (bỏ phiếu huỷ).",
        "- Nguồn: " + "; ".join(f"{k} ← {v}" for k, v in model.mapping.items()) + (("; " + "; ".join(model.notes)) if model.notes else ""),
        f"- N: {len(N)} dòng; R: {len(R)} phiếu; ngày từ {N['date'].min():%Y-%m-%d} đến {N['date'].max():%Y-%m-%d}.",
    ])


class ImportMetrics:
    """Ham mot dong cho cau hoi nhap hang (§5)."""

    def __init__(self, model: ImportModel):
        self.N = model.N[~model.N["cancelled"]]

    def _sel(self, a, b=None, **filters):
        from .metrics import _filters, _period

        d = self.N
        return d[_period(d, a, b, col="date") & _filters(d, filters)]

    def import_cost(self, a, b=None, **filters):
        """Tien nhap hang trong ky (tong tien cac dong nhap, bo phieu huy). None neu shop khong ghi tien nhap."""
        from .answers import round_money

        v = self._sel(a, b, **filters)["line_total"].sum(min_count=1)
        return None if v != v else round_money(v)

    def avg_import_price(self, a, b=None, item: str = None, unit: str | None = None):
        if item is None and isinstance(b, str):  # avg_import_price(THANG_NAY, 'Bia', 'thùng')
            item, unit, b = b, unit, None
        """Gia nhap binh quan 1 <unit> cua mat hang trong ky = tong tien / tong so luong (cac dong nhap theo
        don vi do), bo phieu huy, lam tron ve dong. None neu khong co dong nhap / khong ghi gia."""
        from .answers import round_money

        f = {"item": item}
        if unit and "unit" in self.N.columns:
            f["unit"] = unit
        d = self._sel(a, b, **f)
        q = d["qty"].sum()
        t = d["line_total"].sum(min_count=1)
        return None if not q or t != t else round_money(t / q)

    def top_suppliers(self, a, b=None, by: str = "amount"):
        """Nha cung cap trong ky: by='amount' (tien nhap nhieu nhat) hoac 'receipts' (so phieu nhap - NCC chinh)."""
        d = self._sel(a, b)
        if "supplier" not in d:
            return None
        g = d.groupby("supplier").agg(tien_nhap=("line_total", "sum"), so_phieu=("receipt_id", "nunique")).reset_index()
        col = "tien_nhap" if by == "amount" else "so_phieu"
        return g.sort_values([col, "supplier"], ascending=[False, True]).reset_index(drop=True)


IMPORT_DOC = """\
- import_cost(a, b, **lọc) -> tiền nhập hàng trong kỳ (bỏ phiếu huỷ); avg_import_price(a, b, item, unit) -> giá nhập bình quân 1 <unit> của mặt hàng (theo số lượng, §5); top_suppliers(a, b, by='amount'|'receipts') -> nhà cung cấp xếp hạng (nhập nhiều tiền nhất / nhà cung cấp chính). Trả None -> shop không ghi thông tin đó (unanswerable)."""
