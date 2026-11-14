"""Ton kho theo rulebook §2/§5, chi tra ket qua khi chac chan (khong dung model).

- Ngay hoi = ngay cuoi du lieu (README 'Du lieu co den het'): ton = cot ton cuoi cua bang ton kho.
- Shop co the kho (moi dong ghi ton sau giao dich): ton tai het ngay D = ton o dong cuoi cua mat hang tinh den het D.
- Truong hop khac: tra None (agent tu tinh tu bang goc hoac tra loi khong duoc).
Gia tri ton = ton (don vi co ban) x gia von 1 don vi co ban ghi trong danh muc hang, chi tinh hang con ton.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .loader import META_COLS, Business, _split_name
from .parsing import compact_key


def _desc(biz, t, c):
    ci = biz.info[t].columns.get(c)
    return (ci.description or "").strip().lower() if ci else ""


def _kind(biz, t, c):
    ci = biz.info[t].columns.get(c)
    return ci.kind if ci else "text"


def _norm(c):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(c)).split())


@dataclass
class InventoryModel:
    final: pd.DataFrame | None  # item, item_code, ton (don vi co ban), unit
    card: pd.DataFrame | None  # the kho: item_code, at, ton_sau
    cost: pd.Series | None  # item_key -> gia von 1 don vi co ban (danh muc)
    data_end: pd.Timestamp | None
    mapping: dict[str, str]
    item_names: dict[str, str]
    lots: pd.DataFrame | None = None  # lot, item_code, item, qty_in, expiry, receipt_date, unit_cost
    lot_moves: pd.DataFrame | None = None  # lot, order_id, order_date, status, qty_out, qty_ret


def _find_final(biz: Business):
    conv = biz.readme.text.lower()
    for t, df in biz.tables.items():
        stock = None
        for c in df.columns:
            d = _desc(biz, t, c)
            n = _norm(c)
            if d.startswith("tồn kho (đơn vị cơ bản)") or f"**{c.lower()}** là tồn cuối ngày cuối có dữ liệu" in conv or (not d and n in ("ton cuoi", "ton kho", "ton cuoi ky", "ton thuc", "ton")):
                if _kind(biz, t, c) == "number":
                    stock = c
                    break
        if stock is None:
            continue
        name = next((c for c in df.columns if _desc(biz, t, c).startswith(("tên hàng", "tên")) or (not _desc(biz, t, c) and _norm(c) in ("ten hang", "ten san pham", "hang hoa", "hang", "ten hang hoa", "san pham", "hang ban", "ten mon"))), None)
        code = next((c for c in df.columns if _desc(biz, t, c).startswith("mã hàng") or (not _desc(biz, t, c) and _norm(c) in ("ma hang", "ma sp", "ma sku", "id sp", "ma hang hoa", "ma vach"))), None)
        unit = next((c for c in df.columns if _desc(biz, t, c).startswith("đơn vị cơ bản") or (not _desc(biz, t, c) and _norm(c) in ("dvt", "don vi tinh", "don vi", "dv goc"))), None)
        if name or code:
            return t, stock, name, code, unit
    return None


def build_inventory(biz: Business, sales=None) -> InventoryModel | None:
    mapping = {}
    final = None
    f = _find_final(biz)
    names: dict[str, str] = {}
    if f:
        t, stock, name, code, unit = f
        df = biz.tables[t]
        final = pd.DataFrame({"ton": pd.to_numeric(df[stock], errors="coerce")})
        if name:
            final["item"] = df[name]
        if code:
            final["item_code"] = df[code]
        if unit:
            final["unit"] = df[unit]
        mapping["ton_cuoi"] = f"{t}.{stock}"
    # the kho (sqlite): cot ton sau giao dich + thoi diem + ma hang
    card = None
    for t, df in biz.tables.items():
        tc = next((c for c in df.columns if _norm(c) in ("ton cuoi", "ton sau") and "the kho" in _norm(t) or _norm(c) == "ton cuoi" and "kho" in _norm(t)), None)
        at = next((c for c in df.columns if _kind(biz, t, c) == "datetime"), None)
        code = next((c for c in df.columns if _norm(c) in ("ma hang hoa", "ma hang", "ma sp")), None)
        if tc and at and code and "the kho" in _norm(t):
            card = pd.DataFrame({"item_code": df[code], "at": pd.to_datetime(df[at], errors="coerce"), "ton_sau": pd.to_numeric(df[tc], errors="coerce")}).dropna(subset=["at"])
            card = card.reset_index().sort_values(["at", "index"]).drop(columns="index")
            mapping["the_kho"] = f"{t}.{tc}"
            break
    # gia von 1 don vi co ban trong danh muc
    cost = None
    for t, df in biz.tables.items():
        cc = next((c for c in df.columns if _desc(biz, t, c) == "giá vốn" or (not _desc(biz, t, c) and _norm(c) in ("gia von", "gia von hien tai", "gia von hang"))), None)
        if cc is None or len(df) > 2000:
            continue
        nm = next((c for c in df.columns if _desc(biz, t, c).startswith(("tên", "tên hàng")) or (not _desc(biz, t, c) and _norm(c) in ("ten hang", "ten hang hoa", "ten san pham", "hang", "hang ban", "ten goi", "mon", "ten mon"))), None)
        cd = next((c for c in df.columns if _desc(biz, t, c).startswith("mã hàng") or (not _desc(biz, t, c) and _norm(c) in ("ma hang", "ma hang hoa", "ma sp", "ma sku", "ma vach"))), None)
        keys = {}
        for _, r in df.iterrows():
            v = pd.to_numeric(r[cc], errors="coerce")
            if v != v:
                continue
            for k in (nm, cd):
                if k and isinstance(r[k], str):
                    keys[compact_key(r[k])] = float(v)
                    if k == cd and nm and isinstance(r[nm], str):
                        names[compact_key(r[k])] = r[nm]
        if keys:
            cost = pd.Series(keys)
            mapping["gia_von_danh_muc"] = f"{t}.{cc}"
            break
    # ma hang -> ten hang tu moi danh muc (khong phu thuoc co cot gia von)
    for t, df in biz.tables.items():
        if len(df) > 5000:
            continue
        cd = _pick(biz, t, ("mã hàng",), ("ma hang", "ma hang hoa", "ma sp", "ma sku", "ma vach", "ma mon", "ma hh"))
        nm = _pick(biz, t, ("tên hàng", "tên"), ("ten hang", "ten hang hoa", "ten san pham", "hang", "hang ban", "ten goi", "mon", "ten mon"))
        if cd and nm and not df[cd].duplicated().any():
            for c, n in zip(df[cd], df[nm]):
                if isinstance(c, str) and isinstance(n, str):
                    names.setdefault(compact_key(c), n)
    lots, moves = _build_lots(biz, sales, names, mapping)
    if final is None and card is None and lots is None:
        return None
    end = pd.Timestamp(biz.readme.data_end) if biz.readme.data_end else None
    return InventoryModel(final=final, card=card, cost=cost, data_end=end, mapping=mapping, item_names=names, lots=lots, lot_moves=moves)


def _pick(biz, t, descs=(), names=(), kind=None, exclude=()):
    df = biz.tables[t]
    for c in df.columns:
        if c in exclude or c in META_COLS:
            continue
        d = _desc(biz, t, c)
        if (d and any(d.startswith(x) for x in descs)) or (not d and _norm(c) in names):
            if kind is None or _kind(biz, t, c) in kind:
                return c
    return None


def _to_dates(x: pd.Series) -> pd.Series:
    """Cot ngay; chuoi 'dd/mm/yyyy' doc ngay truoc thang (khong de pandas doan thang truoc)."""
    if pd.api.types.is_datetime64_any_dtype(x):
        return x
    s = x.astype(str).str.strip()
    for rx, fmt in ((r"\d{1,2}/\d{1,2}/\d{4}", "%d/%m/%Y"), (r"\d{1,2}-\d{1,2}-\d{4}", "%d-%m-%Y"), (r"\d{1,2}\.\d{1,2}\.\d{4}", "%d.%m.%Y")):
        if s.str.fullmatch(rx).mean() > 0.9:
            return pd.to_datetime(s, format=fmt, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _date_like(df: pd.DataFrame, c) -> bool:
    s = df[c].dropna()
    if s.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    return s.dtype == object and _to_dates(s.head(300)).notna().mean() > 0.9


def _base_qty(biz, t, c) -> bool:
    """Cot so luong nhap da quy ve don vi co ban (khong phai don vi nhap: hop, thung…)."""
    from .parsing import strip_accents

    txt = _norm(c) + " " + strip_accents(_desc(biz, t, c)).lower()
    return any(k in txt for k in ("quy doi", "co ban", "quy ra le", "quydoi"))


_LOT_NAMES = ("ma lo", "lo", "so lo", "malo")
_EXP_NAMES = ("han su dung", "hsd", "hansudung")


def _build_lots(biz: Business, sales, names: dict, mapping: dict):
    """Lo nhap (moi dong nhap mot lo) + luong ban lay tu lo / khach tra ve lo (§5 nha thuoc).
    Nhieu bang co lo + so luong nhap + han dung (vd so nhap theo hop va bang lo theo don vi co ban): chon bang co
    so luong theo don vi co ban va ma hang (ton lo = nhap - ban, ban ghi theo don vi co ban)."""
    cands, mv = [], None
    for t in biz.tables:
        lot = _pick(biz, t, ("số lô",), _LOT_NAMES)
        if lot is None:
            continue
        df = biz.tables[t]
        qin = _pick(biz, t, ("số lượng nhập",), ("sl nhap quy doi", "sl nhap dvt co ban", "slnhapquydoi", "so luong nhap", "sl nhap"), ("number",))
        exp = _pick(biz, t, ("hạn sử dụng",), _EXP_NAMES)
        if exp is not None and _kind(biz, t, exp) not in ("date", "datetime") and not _date_like(df, exp):
            exp = None
        qout = _pick(biz, t, ("số lượng bán lấy từ lô",), ("sl quy doi", "so luong", "sl quy ra le", "slquydoi", "sl"), ("number",), exclude=(qin,) if qin else ())
        if qin and exp:
            code = _pick(biz, t, ("mã hàng",), ("ma hang", "ma sku", "ma hang hoa", "ma sp", "ma mon", "ma hh"))
            price = _pick(biz, t, ("giá nhập",), ("don gia nhap", "gia nhap", "dongianhap"), ("number",))
            cands.append((2 * _base_qty(biz, t, qin) + bool(code) + 0.5 * bool(price), -len(cands), t, lot, qin, exp))
        elif qout and mv is None and not qin:
            ret = next((c for c in biz.tables[t].columns if _desc(biz, t, c).startswith("số lượng khách trả lại") or (not _desc(biz, t, c) and "tra" in _norm(c).split())), None)
            mv = (t, lot, qout, ret)
    if not cands:
        return None, None
    _, _, t, lot, qin, exp = max(cands)
    df = biz.tables[t]
    code = _pick(biz, t, ("mã hàng",), ("ma hang", "ma sku", "ma hang hoa", "ma sp", "ma mon", "ma hh"))
    rid = _pick(biz, t, ("mã phiếu nhập",), ("ma phieu nhap", "so phieu", "ma don nhap", "maphieunhap"))
    price = _pick(biz, t, ("giá nhập",), ("don gia nhap", "gia nhap", "dongianhap"), ("number",))
    L = pd.DataFrame({"lot": df[lot].astype(str), "qty_in": pd.to_numeric(df[qin], errors="coerce"), "expiry": _to_dates(df[exp])})
    if code:
        L["item_code"] = df[code]
        L["item"] = L["item_code"].map(lambda c: names.get(compact_key(c)) if isinstance(c, str) else None)
    if price:
        L["unit_cost"] = pd.to_numeric(df[price], errors="coerce")
    # ngay nhap cua lo: cot ngay trong bang lo, hoac ngay cua phieu nhap
    dcol = next((c for c in df.columns if _kind(biz, t, c) in ("date", "datetime") and c != exp), None)
    if dcol:
        L["receipt_date"] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
    elif rid:
        L["receipt_id"] = df[rid].astype(str)
        for t2, d2 in biz.tables.items():
            if t2 == t:
                continue
            k2 = next((c for c in d2.columns if d2[c].astype(str).isin(set(L["receipt_id"])).mean() > 0.5 and not d2[c].duplicated().any()), None)
            dd = next((c for c in d2.columns if _kind(biz, t2, c) in ("date", "datetime")), None) if k2 else None
            if k2 and dd:
                L["receipt_date"] = L["receipt_id"].map(d2.set_index(d2[k2].astype(str))[dd]).dt.normalize()
                break
    if "unit_cost" not in L or "receipt_date" not in L:
        # bang lo khong co gia / ngay nhap: lay tu so nhap ghi cung ma lo (gia 1 don vi co ban = thanh tien cua lo / so luong quy doi)
        lots = set(L["lot"])
        for t2, d2 in biz.tables.items():
            lot2 = _pick(biz, t2, ("số lô",), _LOT_NAMES) if t2 != t else None
            if lot2 is None:
                continue
            k2 = d2[lot2].astype(str)
            if k2.isin(lots).mean() < 0.9 or L["lot"].isin(set(k2)).mean() < 0.9:
                continue
            if "unit_cost" not in L:
                tot = _pick(biz, t2, ("thành tiền",), ("thanh tien", "tong tien", "tien nhap"), ("number",))
                if tot:
                    money = pd.to_numeric(d2[tot], errors="coerce").groupby(k2).sum(min_count=1)
                    L["unit_cost"] = L["lot"].map(money) / L["lot"].map(L.groupby("lot")["qty_in"].sum())
                    mapping["gia_nhap_lo"] = f"{t2}.{tot} / {t}.{qin}"
            if "receipt_date" not in L:
                exp2 = _pick(biz, t2, ("hạn sử dụng",), _EXP_NAMES)
                dd = next((c for c in d2.columns if c != exp2 and _kind(biz, t2, c) in ("date", "datetime")), None)
                if dd:
                    L["receipt_date"] = L["lot"].map(pd.to_datetime(d2[dd], errors="coerce").groupby(k2).min()).dt.normalize()
            break
    mapping["lo_nhap"] = f"{t}.{lot}/{qin}/{exp}"
    moves = None
    if mv is not None and sales is not None and len(sales.O):
        t2, lot2, qout, ret = mv
        d2 = biz.tables[t2]
        oid = next((c for c in d2.columns if d2[c].astype(str).isin(set(sales.O["order_id"].astype(str))).mean() > 0.5), None)
        if oid:
            moves = pd.DataFrame({"lot": d2[lot2].astype(str), "order_id": d2[oid].astype(str), "qty_out": pd.to_numeric(d2[qout], errors="coerce").fillna(0)})
            moves["qty_ret"] = pd.to_numeric(d2[ret], errors="coerce").fillna(0) if ret else 0.0
            O = sales.O.set_index(sales.O["order_id"].astype(str))
            moves["order_date"] = moves["order_id"].map(O["order_date"])
            moves["status"] = moves["order_id"].map(O["status"])
            mapping["ban_theo_lo"] = f"{t2}.{lot2}/{qout}" + (f"/{ret}" if ret else "")
    return L, moves


def lot_stock_at(model: InventoryModel, day) -> pd.DataFrame | None:
    """Ton tung lo luc het ngay D = nhap (lo nhap truoc/het D) - ban (don dat truoc/het D, khong huy) + khach tra ve lo."""
    if model.lots is None or model.lot_moves is None:
        return None
    D = pd.Timestamp(str(day)[:10])
    L = model.lots.copy()
    if "receipt_date" in L:
        L = L[L["receipt_date"].isna() | (L["receipt_date"] <= D)]
    M = model.lot_moves
    M = M[(M["order_date"] <= D) & (M["status"] != "huỷ")]
    net = M.groupby("lot")["qty_out"].sum() - M.groupby("lot")["qty_ret"].sum()
    L["ton"] = L["qty_in"] - L["lot"].map(net).fillna(0)
    return L.reset_index(drop=True)


def _day(x) -> pd.Timestamp:
    from .dates import to_date

    return pd.Timestamp(to_date(x[0] if isinstance(x, (tuple, list)) else x))


def expiring_lots(model: InventoryModel, day, a=None, b=None, item=None) -> pd.DataFrame | None:
    """Lo CON TON (ton > 0) luc het ngay D co han dung trong [a, b] (mac dinh tu D, khong gioi han tren), sap han som truoc
    (cung han theo ma hang, ma lo). Cot von = ton x gia nhap 1 don vi co ban cua chinh lo (rulebook: tien von hang sap het han)."""
    L = lot_stock_at(model, _day(day).date())
    if L is None:
        return None
    if isinstance(a, (tuple, list)) and b is None:
        a, b = a[0], a[1]
    lo = _day(a) if a is not None else _day(day)
    exp = L["expiry"].dt.normalize()
    m = (L["ton"] > 0) & (exp >= lo)
    if b is not None:
        m &= exp <= _day(b)
    L = L[m].copy()
    if item is not None:
        from .metrics import _filters

        if "item" not in L and "item_code" not in L:
            raise ValueError("bảng lô không ghi mã/tên hàng -> không lọc theo mặt hàng được")
        L = L[_filters(L, {"item" if "item" in L else "item_code": item})]
    if "unit_cost" in L:
        L["von"] = L["ton"] * L["unit_cost"]
    keys = ["expiry"] + [c for c in ("item_code", "lot") if c in L]
    return L.sort_values(keys).reset_index(drop=True)


def expiring_items(model: InventoryModel, day, a=None, b=None) -> list | None:
    """Mat hang con lo ton co han dung trong [a, b], thu tu rulebook §9: han som nhat cua mat hang truoc, cung han theo ma."""
    L = expiring_lots(model, day, a, b)
    if L is None:
        return None
    key = "item" if "item" in L and L["item"].notna().all() else "item_code"
    g = L.groupby(key).agg(expiry=("expiry", "min"), **({"code": ("item_code", "first")} if "item_code" in L else {})).reset_index()
    g = g.sort_values(["expiry", "code" if "code" in g else key])
    return g[key].tolist()


def nearest_expiry(model: InventoryModel, day, item) -> str | None:
    """Han dung cua lo gan het han nhat cua mot mat hang (rulebook): trong cac lo con ton chua qua han tai D
    (han dung >= D; han dung dung ngay D chua qua han), lo co han som nhat -> 'YYYY-MM-DD'."""
    L = expiring_lots(model, day, None, None, item=item)
    if L is None or L.empty:
        return None
    return L["expiry"].min().date().isoformat()


def stock_at(model: InventoryModel, day) -> pd.DataFrame | None:
    D = pd.Timestamp(str(day)[:10])
    if model.card is not None:
        sub = model.card[model.card["at"] < D + pd.Timedelta(days=1)]
        last = sub.groupby("item_code")["ton_sau"].last().reset_index().rename(columns={"ton_sau": "ton"})
        last["item"] = last["item_code"].map(lambda c: model.item_names.get(compact_key(c)))
        return last
    if model.final is not None and model.data_end is not None and D.normalize() == model.data_end.normalize():
        return model.final.copy()
    ls = lot_stock_at(model, D)
    if ls is not None and "item_code" in ls:
        g = ls.groupby("item_code", as_index=False)["ton"].sum()
        g["item"] = g["item_code"].map(lambda c: model.item_names.get(compact_key(c)))
        return g
    return None


def stock_value_at(model: InventoryModel, day) -> float | None:
    st = stock_at(model, day)
    if st is None or model.cost is None:
        return None
    st = st[st["ton"] > 0]
    key = st["item_code"] if "item_code" in st and st["item_code"].notna().any() else st["item"]
    unit_cost = key.map(lambda k: model.cost.get(compact_key(k)) if isinstance(k, str) else None)
    if unit_cost.isna().any():
        alt = st["item"].map(lambda k: model.cost.get(compact_key(k)) if isinstance(k, str) else None) if "item" in st else unit_cost
        unit_cost = unit_cost.fillna(alt)
    return float((st["ton"] * unit_cost).sum()) if unit_cost.notna().all() else None


def describe(model: InventoryModel) -> str:
    return "\n".join([
        "## Tồn kho dựng sẵn: stock_at(ngày) và stock_value_at(ngày)",
        "- stock_at(D): bảng tồn (item, item_code, ton theo đơn vị cơ bản) lúc hết ngày D. Chỉ có kết quả khi D là ngày cuối dữ liệu (dùng cột tồn cuối) hoặc shop có thẻ kho; ngược lại trả None (tự tính từ bảng gốc hoặc trả lời không được).",
        "- stock_value_at(D): giá trị hàng tồn = Σ tồn × giá vốn 1 đơn vị cơ bản ghi trong danh mục (chỉ hàng còn tồn, §5); None nếu không tính được.",
        "- Nhà thuốc có lô: lot_stock_at(D) -> mỗi lô (lot, item, qty_in, expiry, receipt_date, unit_cost?, ton) lúc hết ngày D = nhập − bán (đơn không huỷ) + khách trả về lô. Lô gần hết hạn nhất của một thuốc = lô còn tồn (ton > 0), chưa quá hạn tại D, expiry sớm nhất.",
        "- Dựng sẵn theo rulebook: expiring_lots(D, kỳ hạn dùng) = lô còn tồn có hạn dùng trong kỳ (cột von = ton × giá nhập 1 đơn vị cơ bản của lô); "
        "expiring_items(D, kỳ) = danh sách mặt hàng theo hạn sớm nhất; nearest_expiry(D, item=...) = hạn dùng 'YYYY-MM-DD' của lô gần hết hạn nhất.",
        "- Tồn theo đơn vị lớn: [số đơn vị lớn nguyên, số đơn vị cơ bản lẻ] = [ton // hệ số, ton % hệ số] (hệ số ở bảng quy đổi / README).",
        "- Nguồn: " + "; ".join(f"{k} ← {v}" for k, v in model.mapping.items()),
    ])
