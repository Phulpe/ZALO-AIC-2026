"""Ham chi so tren bang chuan S/O (rulebook §1, §3) de agent goi mot dong, it cho sai.

Moi ham nhan ky [a, b] (tinh ca hai dau, theo NGAY DAT cua don) va bo loc dang cot=gia_tri tren S
(vd item='Cà phê sữa (M)', group='cà phê', dm_hoat_chat='Paracetamol', channel='Zalo', ca='chiều').
Gia tri bo loc so khop khong phan biet hoa thuong / khoang trang / dau cau; truyen list de lay nhieu gia tri.
"""

from __future__ import annotations

import pandas as pd

from .answers import round2, round_money
from .dates import to_date
from .parsing import compact_key


def _ab(a, b):
    """revenue(THANG_NAY) hoac revenue(a, b): ky la tuple (dau, cuoi)."""
    if b is None and isinstance(a, (tuple, list)) and len(a) == 2 and not all(isinstance(x, (int, float)) for x in a):
        return a[0], a[1]
    if b is None:
        return a, a
    return a, b


def _period(df: pd.DataFrame, a, b=None, col: str = "order_date") -> pd.Series:
    a, b = _ab(a, b)
    d = pd.to_datetime(df[col]).dt.normalize()
    return (d >= pd.Timestamp(to_date(a))) & (d <= pd.Timestamp(to_date(b)))


class FilterError(ValueError):
    pass


def _resolve(series: pd.Series, val):
    """Gia tri loc -> tap khoa khop. Khong khop chinh xac: neu dung MOT gia tri chua cum da cho thi lay gia tri do;
    khong thi bao loi kem goi y (khong am tham tra 0)."""
    import difflib

    from .parsing import norm_key_ascii

    uniq = pd.Series(series.dropna().astype(str).unique())
    keys = set(uniq.map(compact_key))
    k = compact_key(val)
    if k in keys:
        return {k}
    a = compact_key(norm_key_ascii(val))
    contain = [u for u in uniq if k and (k in compact_key(u) or a in compact_key(norm_key_ascii(u)))]
    if len(contain) == 1:
        return {compact_key(contain[0])}
    if not contain:  # moi tu cua gia tri loc deu co trong dung mot ten (vd 'rau cu kho quet' ~ 'Rau củ kho quẹt (lớn)')
        import re

        def toks(x):
            return set(t for t in re.split(r"[^0-9a-z]+", norm_key_ascii(x)) if t)

        qt = toks(val)
        sub = [u for u in uniq if len(qt) >= 2 and qt <= toks(u)]
        if len(sub) == 1:
            return {compact_key(sub[0])}
        contain = sub
    sugg = contain[:8] or difflib.get_close_matches(str(val), uniq.tolist(), n=6, cutoff=0.5)
    raise FilterError(
        f"không có giá trị {val!r} trong cột '{series.name}'"
        + (f"; các giá trị gần giống: {sugg}" if sugg else "; dùng find_names(cột, 'từ khoá') để dò")
        + ". Không có thật thì trả lời unanswerable."
    )


def _filters(df: pd.DataFrame, filters: dict) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col not in df.columns:
            raise KeyError(f"bảng không có cột '{col}'. Các cột: {', '.join(df.columns)}")
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        if col == "item" and "item_code" in df.columns:
            codes = set(df["item_code"].dropna().astype(str).map(compact_key))
            if vals and all(compact_key(v) in codes for v in vals):  # model truyen ma hang vao item
                col = "item_code"
        if df[col].dtype == object:
            keys = set()
            errors = []
            for v in vals:
                if isinstance(v, bool):
                    keys.add(compact_key(v))
                    continue
                try:
                    keys |= _resolve(df[col], v)
                except FilterError as exc:
                    errors.append(str(exc))
            if errors and (not keys or len(errors) == len(vals)):
                raise FilterError(errors[0])
            m &= df[col].map(lambda x: compact_key(x) in keys if x is not None else False)
        else:
            m &= df[col].isin(vals)
    return m


class Metrics:
    def __init__(self, S: pd.DataFrame, O: pd.DataFrame):
        self.S, self.O = S, O
        self.SR = S[S["revenue_ok"] & S.get("revenue_line", True)] if len(S) else S
        self.OR = O[O["revenue_ok"]] if len(O) else O
        if len(self.SR) and "cogs" in self.SR and "amount" in self.SR:
            self.SR = self.SR.assign(profit=self.SR["amount"] - self.SR["cogs"])

    def _need_profit(self):
        if "profit" not in self.SR:
            raise FilterError("shop không ghi giá vốn trên dòng bán -> không tính được lãi gộp (trả lời unanswerable, §7)")

    def gross_profit(self, a, b=None, **filters) -> int:
        """Lai gop (§5) = doanh thu - gia von dong (qty x cost tren dong ban), don tinh doanh thu."""
        self._need_profit()
        d = self.SR
        return round_money(d.loc[_period(d, a, b) & _filters(d, filters), "profit"].sum())

    def revenue(self, a, b=None, **filters) -> int:
        """Doanh thu (§1) trong ky = tong thanh tien cac dong hang cua don khong huy, khong tra."""
        if len(self.S):
            d = self.SR
            return round_money(d.loc[_period(d, a, b) & _filters(d, filters), "amount"].sum())
        d = self.OR
        return round_money(d.loc[_period(d, a, b) & _filters(d, filters), "revenue"].sum())

    def n_orders(self, a, b=None, **filters) -> int:
        """So don (khong huy, khong tra) dat trong ky; bo loc tren cot cua S (vd item=...) hoac O."""
        if filters and len(self.S) and all(c in self.S.columns for c in filters):
            d = self.SR
            return int(d.loc[_period(d, a, b) & _filters(d, filters), "order_id"].nunique())
        d = self.OR
        if not len(d):
            return None  # shop khong ghi ma don: khong dem duoc so don
        return int((_period(d, a, b) & _filters(d, filters)).sum())

    def order_values(self, a, b=None, **filters) -> pd.Series:
        """Tien hang tung don trong ky (§1: don khong huy, khong tra; tong tien dong hang, khong gom phi ship / voucher), theo ma don.
        Dem don dat nguong: (order_values(ky) >= 500000).sum()."""
        if not len(self.S):
            d = self.OR
            return d[_period(d, a, b) & _filters(d, filters)].groupby("order_id")["revenue"].sum()
        d = self.SR
        return d[_period(d, a, b) & _filters(d, filters)].groupby("order_id")["amount"].sum()

    def qty_unit(self, a, b=None, unit: str = None, **filters) -> float:
        """Tong so luong ban tinh theo DON VI BAN <unit>, cong moi mat hang ban theo don vi do (don khong huy, khong tra):
        "hom qua ban duoc bao nhieu ly", "cong moi mat hang tinh theo binh". So don vi khong phan biet dau."""
        from .parsing import norm_key_ascii

        A = self.SR
        if "unit" not in A or unit is None:
            raise KeyError("bảng bán không ghi đơn vị bán (cột unit)")
        # mat hang co don vi co ban la <unit>: co dong ban theo <unit> voi he so 1 (qty_base == qty); ban theo thung / loc
        # cua mat hang do van quy ve <unit> qua qty_base
        u = A["unit"].map(norm_key_ascii) == norm_key_ascii(unit)
        base = set(A.loc[u & (A["qty_base"] == A["qty"]), "item"].dropna()) if "qty_base" in A else set(A.loc[u, "item"].dropna())
        d = A[_period(A, a, b) & _filters(A, filters)]
        col = "qty_base" if "qty_base" in d else "qty"
        v = d.loc[d["item"].isin(base), col].sum()
        return int(v) if float(v).is_integer() else float(v)

    def unit_is_base(self, item: str, unit: str) -> bool | None:
        """<unit> co phai don vi co ban cua mat hang khong (co dong ban theo <unit> voi he so 1). None: shop khong ghi don vi ban.
        Dung truoc khi tra so luong / don gia / ton theo don vi co ban cho cau hoi ghi don vi."""
        from .parsing import norm_key_ascii

        A = self.SR
        if not len(A) or "unit" not in A or "item" not in A:
            return None
        d = A[_filters(A, {"item": item})]
        if not len(d):
            return None
        one = d[d["qty_base"] == d["qty"]] if "qty_base" in d and "qty" in d else d
        return norm_key_ascii(unit) in set(one["unit"].dropna().map(norm_key_ascii))

    def qty_sold(self, a, b=None, **filters) -> float:
        """So luong ban quy ve don vi co ban (don khong huy, khong tra)."""
        d = self.SR
        v = d.loc[_period(d, a, b) & _filters(d, filters), "qty_base"].sum()
        return int(v) if float(v).is_integer() else float(v)

    def top_items(self, a, b=None, k: int = 3, by: str = "qty_base", key: str = "item", **filters) -> pd.DataFrame:
        """Xep hang mat hang (bo Phi phuc vu/Voucher): by='qty_base' (ban chay), 'amount' (doanh thu),
        'orders' (so don co mat hang - ua chuong), 'profit' (lai gop). key= cot xep hang khac item
        (vd key='dong_may', 'hang', 'group'); dong khong co gia tri cot do (phu kien voi dong_may) bi bo.
        Bang nhau xep theo ma/ten tang dan."""
        if isinstance(b, int) and isinstance(a, (tuple, list)):  # top_items(THANG_NAY, 3)
            a, b, k = a, None, b
        if by in ("qty", "so_luong", "quantity"):
            by = "qty_base"
        if by in ("revenue", "doanh_thu"):
            by = "amount"
        if by in ("lai", "lai_gop", "gross_profit"):
            by = "profit"
        if by == "profit":
            self._need_profit()
        if not len(self.SR):
            raise FilterError("shop không ghi chi tiết mặt hàng (chỉ có danh sách hoá đơn) -> không xếp hạng mặt hàng được (unanswerable, §7)")
        d = self.SR[self.SR.get("is_item", True)]
        d = d[_period(d, a, b) & _filters(d, filters)]
        if key not in d.columns:
            raise FilterError(f"không có cột {key!r} để xếp hạng; các cột: {', '.join(c for c in d.columns if not c.startswith('_'))}")
        d = d[d[key].notna()]
        if by == "orders":
            g = d.groupby(key)["order_id"].nunique()
        else:
            g = d.groupby(key)[by].sum()
        out = g.rename("value").reset_index()
        tie = "item_code" if key == "item" and "item_code" in d.columns else key
        if tie != key:
            out[tie] = out[key].map(d.drop_duplicates(key).set_index(key)[tie])
        out = out.sort_values(["value", tie], ascending=[False, True]).reset_index(drop=True)
        return out.head(k)

    def cancel_rate(self, a, b=None, **filters) -> float | None:
        """Ty le don huy (%) tren MOI don dat trong ky (moi trang thai), 2 chu so."""
        if not len(self.O):
            return None
        d = self.O[_period(self.O, a, b) & _filters(self.O, filters)]
        return round2(100.0 * (d["status"] == "huỷ").sum() / len(d)) if len(d) else None

    def n_cancelled(self, a, b=None, **filters) -> int | None:
        if not len(self.O):
            return None
        d = self.O[_period(self.O, a, b) & _filters(self.O, filters)]
        return int((d["status"] == "huỷ").sum())

    def avg_order_value(self, a, b=None, **filters) -> int | None:
        """Gia tri trung binh mot don = doanh thu / so don (don khong huy, khong tra), lam tron ve dong."""
        if not len(self.OR):
            return None
        d = self.OR[_period(self.OR, a, b) & _filters(self.OR, filters)]
        return round_money(d["revenue"].sum() / len(d)) if len(d) else None

    def unsold_items(self, a, b=None) -> list:
        """Mat hang e (§5): da tung ban truoc ky (don tinh doanh thu) nhung trong ky khong co don nao mua."""
        a, b = _ab(a, b)
        d = self.SR[self.SR.get("is_item", True)] if len(self.SR) else self.SR
        od = pd.to_datetime(d["order_date"]).dt.normalize()
        before = set(d.loc[od < pd.Timestamp(to_date(a)), "item"].dropna())
        during = set(d.loc[_period(d, a, b), "item"].dropna())
        return sorted(before - during)

    def by_day(self, a, b=None) -> pd.DataFrame:
        """Theo ngay dat trong ky: doanh_thu, so_don (don tinh doanh thu). Ngay ban nhieu tien nhat = max doanh_thu;
        ngay dat khach nhat = max so_don (bang nhau lay ngay som hon)."""
        if not len(self.OR):  # shop khong ghi ma don: tinh theo dong hang, khong dem duoc so don
            d = self.SR[_period(self.SR, a, b)]
            g = d.groupby("order_date").agg(doanh_thu=("amount", "sum")).reset_index()
            g["so_don"] = None
            return g.sort_values("order_date").reset_index(drop=True)
        d = self.OR[_period(self.OR, a, b)]
        g = d.groupby("order_date").agg(doanh_thu=("revenue", "sum"), so_don=("order_id", "nunique")).reset_index()
        return g.sort_values("order_date").reset_index(drop=True)

    def by_hour(self, a, b=None) -> pd.DataFrame:
        """So don va doanh thu theo khung 1 gio tron (gio bat dau) theo gio tao don; gio cao diem = max so_don."""
        if not len(self.OR) or "hour" not in self.OR:
            return pd.DataFrame()
        d = self.OR[_period(self.OR, a, b)]
        g = d.groupby("hour").agg(so_don=("order_id", "nunique"), doanh_thu=("revenue", "sum")).reset_index()
        return g.sort_values(["so_don", "hour"], ascending=[False, True]).reset_index(drop=True)

    def price_on(self, day, item: str = None, **kw):
        """Gia ban mot don vi co ban cua mat hang ngay D: don gia tren cac dong ban cua don dat ngay D,
        truoc chiet khau dong, tinh ca don huy / tra. Nhieu muc gia -> muc gap nhieu nhat."""
        d = self.S[_period(self.S, day, None) & _filters(self.S, {"item": item})]
        if not len(d) or "price" not in d:
            return None
        f = (d["qty_base"] / d["qty"]).where(d["qty"] != 0, 1.0) if "qty_base" in d and "qty" in d else 1.0
        unit = (d["price"] / f).round(2)
        v = unit.mode().iloc[0]
        return int(v) if float(v).is_integer() else float(v)

    def adjustments(self, a, b=None, kind: str = "voucher") -> int:
        """Tien giam voucher / tien giam nho dung diem / phi ship (kind='voucher'|'diem'|'ship'): SO DUONG, theo ngay hoa don,
        chi hoa don khong huy, khong tra (rulebook Me va be)."""
        S = self.S
        if not len(S) or "revenue_line" not in S:
            return 0
        k = {"voucher": "voucher", "diem": "diem", "điểm": "diem", "points": "diem", "ship": "ship", "phi ship": "ship", "phí ship": "ship"}.get(str(kind).lower(), str(kind).lower())
        name = S["item"].astype(str).map(lambda x: compact_key(x))
        from .parsing import strip_accents

        name = name.map(strip_accents)
        sel = ~S["revenue_line"] & S["revenue_ok"] & name.str.contains(k) & _period(S, a, b)
        return round_money(abs(S.loc[sel, "amount"].sum()))

    def price_change_date(self, item: str, new_price: float):
        """Ngay tang gia len gia moi (rulebook Hieu ung tang gia): theo bang gia / lich su gia = ngay ap dung dau tien co gia moi
        cao hon gia lien truoc; shop khong co bang gia -> ngay dau tien ban o gia moi (gia 1 don vi co ban tren dong ban). Khong co -> None."""
        H = getattr(self, "price_hist", None)
        if H is not None and len(H):
            try:
                m = _filters(H, {"item": item})
            except (FilterError, KeyError):
                m = None
            if m is not None and m.any():
                h = H[m]
                hit = h[((h["price"] - float(new_price)).abs() < 0.5) & h["old_price"].notna() & (h["price"] > h["old_price"])]
                if len(hit):
                    return pd.Timestamp(hit["date"].min()).date()
        d = self.S[_filters(self.S, {"item": item})]
        if not len(d) or "price" not in d:
            return None
        f = (d["qty_base"] / d["qty"]).where(d["qty"] != 0, 1.0) if "qty_base" in d and "qty" in d else 1.0
        unit = d["price"] / f
        hit = d.loc[(unit - float(new_price)).abs() < 0.5, "order_date"]
        return None if hit.empty else pd.Timestamp(hit.min()).date()

    def price_effect(self, day, n: int, item: str = None, **filters):
        """Hieu ung doi gia (§5): % thay doi so luong ban N ngay ke tu ngay D (tinh ca D) so voi N ngay ngay truoc D.
        Trung binh moi ngay cung ra cung %, vi hai ky cung N ngay."""
        import datetime as _dt

        from .helpers import pct_change

        D = to_date(day[0] if isinstance(day, (tuple, list)) else day)
        n = int(n)
        if item is not None:
            filters["item"] = item
        after = self.qty_sold(D, D + _dt.timedelta(days=n - 1), **filters)
        before = self.qty_sold(D - _dt.timedelta(days=n), D - _dt.timedelta(days=1), **filters)
        return pct_change(after, before)

    def customers(self, a, b=None, gom_huy: bool = False) -> pd.DataFrame:
        """Moi khach (bo khach le) co don tinh doanh thu trong ky: so don, so ngay co mua, doanh thu, gia tri TB mot don, so don huy.
        gom_huy=True: them ca khach chi co don huy trong ky (cau hoi khach hay doi y nhat)."""
        if not len(self.O):
            return pd.DataFrame()
        O = self.O[_period(self.O, a, b)]
        if "is_walk_in" in O:
            O = O[~O["is_walk_in"].fillna(False)]
        key = "customer_id" if "customer_id" in O and O["customer_id"].notna().any() else "customer_key"
        if key not in O:
            return pd.DataFrame()
        ok = O[O["revenue_ok"]]
        g = ok.groupby(key).agg(khach=("customer" if "customer" in ok else key, "first"), so_don=("order_id", "nunique"),
                                so_ngay_mua=("order_date", "nunique"), doanh_thu=("revenue", "sum"))
        g["gia_tri_tb_don"] = (g["doanh_thu"] / g["so_don"]).map(round_money)
        # khach chi co don huy (hay doi y nhat) van co dong: so_don = 0, doanh_thu = 0
        huy = O[O["status"] == "huỷ"].groupby(key).agg(khach=("customer" if "customer" in O else key, "first"), so_don_huy=("order_id", "nunique"))
        if gom_huy:
            g = g.reindex(g.index.union(huy.index))
        g["khach"] = g["khach"].fillna(huy["khach"])
        g[["so_don", "so_ngay_mua", "doanh_thu"]] = g[["so_don", "so_ngay_mua", "doanh_thu"]].fillna(0)
        g[["so_don", "so_ngay_mua"]] = g[["so_don", "so_ngay_mua"]].astype(int)
        g["so_don_huy"] = huy["so_don_huy"].reindex(g.index).fillna(0).astype(int)
        g.index.name = key
        g = g.reset_index()
        # §9: bang nhau xep theo ma tang dan (khong co ma thi theo ten)
        return g.sort_values(["doanh_thu", g.columns[0]], ascending=[False, True]).reset_index(drop=True)



_PH_DATE = ("ngay ap dung", "ngay hieu luc", "hieu luc tu", "tu ngay", "ngay doi gia", "ngay thay doi gia", "ngay thay doi", "ngay ap dung gia")
_PH_NEW = ("gia moi", "gia ban le", "gia ban moi", "don gia moi", "gia ban", "gia niem yet")
_PH_OLD = ("gia cu", "gia truoc", "gia ban cu", "don gia cu")


def build_price_history(tables: dict, S: pd.DataFrame) -> pd.DataFrame | None:
    """Bang gia / lich su gia (rulebook Hieu ung tang gia: 'lan tang gia = ngay D gia moi cao hon gia lien truoc theo bang gia').
    -> item (ten trong S), date, price, old_price (gia lien truoc: cot gia cu, khong co thi dong truoc cua cung mat hang)."""
    from .inventory import _norm as _cn

    if S is None or not len(S) or "item" not in S:
        return None
    names = {compact_key(x): x for x in S["item"].dropna().astype(str).unique()}
    code2name = {}
    if "item_code" in S:
        for c, n in S[["item_code", "item"]].dropna().drop_duplicates("item_code").itertuples(index=False):
            code2name[compact_key(c)] = n
    for t, df in tables.items():
        if not len(df) or len(df) > 20000:
            continue
        cols = {c: _cn(c) for c in df.columns}
        dcol = next((c for c, v in cols.items() if v in _PH_DATE), None)
        pcol = next((c for v0 in _PH_NEW for c, v in cols.items() if v == v0), None)
        if dcol is None or pcol is None:
            continue
        ocol = next((c for c, v in cols.items() if v in _PH_OLD), None)
        key = None
        for kind, ref in (("code", code2name), ("name", names)):
            for c in df.columns:
                if c in (dcol, pcol, ocol) or df[c].dtype != object or not ref:
                    continue
                k = df[c].dropna().astype(str).map(compact_key)
                if len(k) and k.isin(ref).mean() > 0.8:
                    key = (kind, c)
                    break
            if key:
                break
        if key is None:
            continue
        ref = code2name if key[0] == "code" else names
        H = pd.DataFrame({"item": df[key[1]].astype(str).map(compact_key).map(ref),
                          "date": pd.to_datetime(df[dcol], errors="coerce").dt.normalize(),
                          "price": pd.to_numeric(df[pcol], errors="coerce")})
        if ocol:
            H["old_price"] = pd.to_numeric(df[ocol], errors="coerce")
        H = H.dropna(subset=["item", "date", "price"]).sort_values(["item", "date"], kind="stable")
        prev = H.groupby("item")["price"].shift(1)
        H["old_price"] = H["old_price"].fillna(prev) if "old_price" in H else prev
        H.attrs["source"] = f"{t}: {key[1]} / {dcol} / {pcol}" + (f" / {ocol}" if ocol else "")
        return H.reset_index(drop=True)
    return None

DOC = """\
- revenue(a, b, **lọc) -> doanh thu (§1) kỳ [a, b] theo ngày đặt; lọc theo cột của S, vd revenue(a, b, group='cà phê'), revenue(a, b, item='Cà phê sữa (M)'), revenue(a, b, ca='chiều'), revenue(a, b, channel='Zalo'), revenue(a, b, customer_group='sỉ') (đơn khách lẻ không thuộc nhóm nào), revenue(a, b, dm_hoat_chat='Cetirizin'), revenue(a, b, dm_thuoc_ke_don=True). Giá trị lọc không có trong dữ liệu -> báo lỗi kèm gợi ý (không trả 0).
- order_values(a, b, **lọc) -> tiền hàng từng đơn (đơn không huỷ, không trả); đếm đơn có tiền hàng từ X trở lên: int((order_values(kỳ) >= X).sum()).
- unit_is_base(item, unit) -> True nếu <unit> là đơn vị cơ bản của mặt hàng (qty_sold, price_on, stock_at tính theo đơn vị cơ bản).
- qty_unit(a, b, unit='ly') -> tổng số lượng bán theo đơn vị bán đó, cộng mọi mặt hàng bán theo đơn vị đó ("hôm qua bán được bao nhiêu ly").
- n_orders(a, b, **lọc) -> số đơn không huỷ, không trả; qty_sold(a, b, **lọc) -> số lượng bán quy về đơn vị cơ bản; avg_order_value(a, b) -> giá trị TB một đơn (làm tròn đồng).
- top_items(a, b, k, by='qty_base'|'amount'|'orders'|'profit', key='item', **lọc) -> bảng mặt hàng xếp hạng (cột item, value), bằng nhau theo mã tăng dần; key='dong_may'/'hang'/'group' xếp hạng theo cột đó (cột <key>, value). Bán chạy = 'qty_base'; doanh thu cao nhất = 'amount'; được ưa chuộng (nhiều đơn) = 'orders'; lời/lãi nhất = 'profit'.
- gross_profit(a, b, **lọc) -> lãi gộp (§5) = doanh thu − giá vốn dòng; shop không ghi giá vốn -> báo lỗi (trả lời unanswerable).
- cancel_rate(a, b) -> tỉ lệ đơn huỷ (%) trên mọi đơn đặt trong kỳ; n_cancelled(a, b) -> số đơn huỷ.
- unsold_items(kỳ) -> list mặt hàng ế (đã từng bán trước kỳ, trong kỳ không bán được); by_day(kỳ) -> doanh_thu, so_don theo ngày; by_hour(kỳ) -> so_don theo giờ tạo đơn (giờ cao điểm = dòng đầu).
- adjustments(kỳ, kind='voucher'|'diem'|'ship') -> tiền giảm voucher / tiền giảm nhờ dùng điểm / phí ship, SỐ DƯƠNG, theo ngày hoá đơn, chỉ hoá đơn không huỷ, không trả.
- expenses(kỳ, nhom=None) -> tổng chi trong sổ chi phí (mọi khoản, kể cả sửa chữa, thay thiết bị); net_profit(tháng) -> lãi ròng = gross_profit − expenses (rulebook). Không có sổ chi phí -> báo lỗi (unanswerable).
- price_effect(ngày_đổi_giá, N, item=...) -> % thay đổi số lượng bán N ngày kể từ ngày đổi giá (tính cả ngày đó) so với N ngày ngay trước (§5). Ngày đổi giá = ngày câu hỏi ghi; câu hỏi không ghi ngày thì price_change_date(item, giá_mới) (theo bảng giá: ngày áp dụng đầu tiên giá mới cao hơn giá liền trước; shop không có bảng giá thì ngày đầu tiên bán ở giá mới).
- price_on(ngày, item) -> giá bán 1 đơn vị cơ bản của mặt hàng trên các dòng bán ngày đó (trước chiết khấu dòng, kể cả đơn huỷ/trả).
- customers(a, b) -> mỗi khách (bỏ khách lẻ): cột đầu là mã/khoá khách, khach (tên), so_don, so_ngay_mua, doanh_thu, gia_tri_tb_don, so_don_huy; sắp theo doanh_thu giảm dần. Xếp theo tiêu chí khác thì sort_values([tiêu_chí, cột đầu], ascending=[False, True]) (bằng nhau theo mã tăng dần, §9). Khách trung thành nhất = nhiều so_ngay_mua nhất; sộp nhất = gia_tri_tb_don cao nhất trong khách có so_don >= 3; hay đổi ý nhất = nhiều so_don_huy nhất, dùng customers(a, b, gom_huy=True) (gồm cả khách chỉ có đơn huỷ).
- S, O đã lọc sẵn (chỉ dòng/đơn tính doanh thu): đếm đơn, khách, doanh thu đều dùng S/O hoặc các hàm trên. S_gom_huy_tra, O_gom_huy_tra gồm cả đơn huỷ/trả: CHỈ dùng khi câu hỏi hỏi về huỷ/trả hàng.
"""
