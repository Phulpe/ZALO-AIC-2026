"""So no + so tra gop tu ghi (cua hang dien thoai). Rulebook muc "Cua hang dien thoai" va §4.

- So no: dong dau moi don ghi ngay mua, khach, **No** = so con no sau khi tra luc mua; cac lan tra sau ghi
  **Da tra** / **Ngay tra** (lan dau o dong dau don, moi lan sau mot dong).
- So gop: dong dau moi don ghi ngay mua, STT don, khach, may, **Gia**, **Tra truoc**, **So ky**, **Gop/thang**,
  **Han gop (ngay)**; moi lan gop ghi **Ngay gop** / **Da gop**. Ky 1 co han vao ngay "Han gop" cua thang ngay sau thang mua,
  ky sau cung ngay thang ke tiep.
- Rulebook: don tra gop la don ghi no; phan gop chua tra la tien khach con no, tien gop la tien tra no. Ky co han TRUOC ngay
  hoi ma chua gop du la qua han; ky co han dung ngay hoi thi chua qua han.

Ca hai so do vao DebtModel (debt.py) nen debt_as_of / debt_by_customer / debt_added / debt_paid dung chung.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass, field

import pandas as pd

from .debt import DebtModel
from .loader import META_COLS, Business, _split_name
from .parsing import norm_key


def _n(c: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(c)).split())


def _col(df: pd.DataFrame, *keys: str, exact: bool = False):
    for c in df.columns:
        if c in META_COLS:
            continue
        n = _n(c)
        if any((n == k) if exact else (k in n) for k in keys):
            return c
    return None


def _ckey(name, phone) -> str:
    ph = re.sub(r"\D", "", str(phone)) if phone is not None and not (isinstance(phone, float) and pd.isna(phone)) else ""
    return norm_key(name) + (f"|{ph}" if ph else "")


def _due_date(purchase: dt.date, k: int, day: int) -> dt.date:
    y, m = purchase.year, purchase.month + k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return dt.date(y, m, min(int(day), calendar.monthrange(y, m)[1]))


@dataclass
class InstallmentModel:
    contracts: pd.DataFrame  # contract_id, order_date, stt, customer, phone, customer_key, may, gia, tra_truoc, so_ky, gop_thang, han_ngay, con_lai_ban_dau
    payments: pd.DataFrame  # contract_id, paid_date, amount
    notes: list = field(default_factory=list)


def _groups(df: pd.DataFrame, head_col: str) -> pd.Series:
    """So dong dau don (o head_col co gia tri) -> ma nhom cho moi dong (cac dong sau thuoc don phia tren)."""
    return df[head_col].notna().cumsum()


def build_phone_debt(biz: Business) -> tuple[DebtModel | None, InstallmentModel | None]:
    inst_t = debt_t = None
    for t, df in biz.tables.items():
        if _col(df, "so ky") and _col(df, "gop"):
            inst_t = t
        elif _col(df, "da tra") and _col(df, "ngay tra") and _col(df, "no", exact=True):
            debt_t = t
    if inst_t is None and debt_t is None:
        return None, None
    orders, pays, notes = [], [], []
    inst = None
    if inst_t is not None:
        df = biz.tables[inst_t].reset_index(drop=True)
        c_date, c_cus, c_ph = _col(df, "ngay", exact=True), _col(df, "khach"), _col(df, "sdt", "dien thoai")
        c_gia, c_tt, c_ky, c_gop = _col(df, "gia", exact=True), _col(df, "tra truoc"), _col(df, "so ky"), _col(df, "gop thang", "gop moi thang")
        c_han, c_pd, c_pa = _col(df, "han gop"), _col(df, "ngay gop"), _col(df, "da gop")
        c_stt, c_may = _col(df, "stt", exact=True), _col(df, "may", exact=True)
        g = _groups(df, c_gia)
        rows = []
        for gid, part in df.groupby(g):
            if gid == 0:
                continue
            h = part.iloc[0]
            od = pd.Timestamp(h[c_date]).date() if pd.notna(h[c_date]) else None
            if od is None:
                continue
            cid = f"gop:{int(h[c_stt]) if c_stt and pd.notna(h[c_stt]) else gid}:{od:%Y-%m}"
            gia, tt = float(h[c_gia] or 0), float(h[c_tt] or 0) if c_tt and pd.notna(h[c_tt]) else 0.0
            rows.append({"contract_id": cid, "order_date": pd.Timestamp(od), "stt": h[c_stt] if c_stt else None,
                         "customer": h[c_cus], "phone": h[c_ph] if c_ph else None, "customer_key": _ckey(h[c_cus], h[c_ph] if c_ph else None),
                         "may": h[c_may] if c_may else None, "gia": gia, "tra_truoc": tt, "so_ky": int(h[c_ky]), "gop_thang": float(h[c_gop]),
                         "han_ngay": int(h[c_han]), "con_lai_ban_dau": gia - tt})
            for _, r in part.iterrows():
                if c_pd and pd.notna(r[c_pd]) and c_pa and pd.notna(r[c_pa]):
                    pays.append({"order_id": cid, "paid_date": pd.Timestamp(r[c_pd]).normalize(), "amount": float(r[c_pa])})
        C = pd.DataFrame(rows)
        P_inst = pd.DataFrame([p for p in pays if p["order_id"].startswith("gop:")], columns=["order_id", "paid_date", "amount"])
        inst = InstallmentModel(contracts=C, payments=P_inst.rename(columns={"order_id": "contract_id"}),
                                notes=[f"{len(C)} hợp đồng trả góp ({inst_t})"])
        for r in rows:
            orders.append({"order_id": r["contract_id"], "order_date": r["order_date"], "customer": r["customer"], "customer_key": r["customer_key"],
                           "total_due": r["con_lai_ban_dau"], "paid_on_purchase_day": 0.0, "kind": "trả góp"})
        notes.append(f"{len(C)} đơn trả góp ({inst_t}): nợ = Giá − Trả trước, trả nợ = các lần góp")
    if debt_t is not None:
        df = biz.tables[debt_t].reset_index(drop=True)
        c_date, c_cus, c_ph = _col(df, "ngay", exact=True), _col(df, "khach"), _col(df, "sdt", "dien thoai")
        c_no, c_paid, c_pd, c_note = _col(df, "no", exact=True), _col(df, "da tra"), _col(df, "ngay tra"), _col(df, "ghi chu")
        g = _groups(df, c_no)
        n0 = len(orders)
        for gid, part in df.groupby(g):
            if gid == 0:
                continue
            h = part.iloc[0]
            if pd.isna(h[c_date]):
                continue
            od = pd.Timestamp(h[c_date]).normalize()
            stt = re.search(r"\d+", str(h[c_note])) if c_note and pd.notna(h[c_note]) else None
            oid = f"no:{stt.group(0) if stt else gid}:{od:%Y-%m}"
            orders.append({"order_id": oid, "order_date": od, "customer": h[c_cus], "customer_key": _ckey(h[c_cus], h[c_ph] if c_ph else None),
                           "total_due": float(h[c_no]), "paid_on_purchase_day": 0.0, "kind": "sổ nợ"})
            for _, r in part.iterrows():
                if pd.notna(r[c_pd]) and pd.notna(r[c_paid]):
                    pays.append({"order_id": oid, "paid_date": pd.Timestamp(r[c_pd]).normalize(), "amount": float(r[c_paid])})
        notes.append(f"{len(orders) - n0} đơn ghi nợ ({debt_t}): Nợ là số còn nợ sau khi trả lúc mua")
    O = pd.DataFrame(orders)
    O["status"] = "hoàn thành"
    O["debt_eligible"] = True
    O["on_credit"] = True
    O["is_walk_in"] = False
    P = pd.DataFrame(pays, columns=["order_id", "paid_date", "amount"])
    P["paid_at"] = P["paid_date"]
    model = DebtModel(orders=O, payments=P, mapping={"so_no": debt_t, "tra_gop": inst_t}, notes=notes)
    return model, inst


def installments_as_of(model: InstallmentModel, day) -> pd.DataFrame:
    """Moi hop dong tra gop luc het ngay D: so_ky_toi_han (ky co han <= D, ke ca ky han dung ngay D), da_gop,
    con_lai (ca ky chua toi han), tien_qua_han (ky co han TRUOC D ma chua gop du), qua_han (bool)."""
    D = pd.Timestamp(str(day)[:10]).date()
    C = model.contracts[model.contracts["order_date"].dt.date <= D].copy()
    paid = model.payments[model.payments["paid_date"].dt.date <= D].groupby("contract_id")["amount"].sum()
    out = []
    for r in C.itertuples():
        od = r.order_date.date()
        dues = [_due_date(od, k, r.han_ngay) for k in range(1, r.so_ky + 1)]
        amt = [r.gop_thang] * (r.so_ky - 1) + [max(0.0, r.con_lai_ban_dau - r.gop_thang * (r.so_ky - 1))]
        n_due = sum(d <= D for d in dues)
        due_before = sum(a for d, a in zip(dues, amt) if d < D)
        p = float(paid.get(r.contract_id, 0.0))
        out.append({"contract_id": r.contract_id, "order_date": r.order_date, "customer": r.customer, "customer_key": r.customer_key, "may": r.may,
                    "so_ky": r.so_ky, "gop_thang": r.gop_thang, "so_ky_toi_han": n_due, "da_gop": p,
                    "con_lai": max(0.0, r.con_lai_ban_dau - p), "tien_qua_han": max(0.0, due_before - p)})
    df = pd.DataFrame(out)
    if len(df):
        df["qua_han"] = df["tien_qua_han"] > 0.5
    return df


def describe(model: InstallmentModel) -> str:
    C = model.contracts
    return "\n".join([
        "## Trả góp dựng sẵn (rulebook mục Cửa hàng điện thoại)",
        f"- {len(C)} hợp đồng; installments_as_of(ngày) -> mỗi hợp đồng lúc hết ngày đó: customer, may, so_ky, gop_thang, so_ky_toi_han "
        "(kỳ có hạn đến hết ngày đó, KỂ CẢ kỳ hạn đúng ngày đó), da_gop, con_lai (gồm cả kỳ chưa tới hạn), tien_qua_han (kỳ có hạn TRƯỚC ngày đó mà chưa góp đủ), qua_han.",
        "- Hạn kỳ k = ngày 'Hạn góp' của tháng thứ k sau tháng mua. Số khách quá hạn = số khách (customer_key) có qua_han.",
        "- Công nợ (debt_as_of, debt_by_customer, debt_added, debt_paid) đã gồm cả đơn trả góp (cột kind = 'trả góp' / 'sổ nợ'), đúng rulebook: câu hỏi khách nợ, trả nợ gồm cả trả góp.",
    ])


# ---------------------------------------------------------------------------- so ban dien thoai: khach tra, bot, thu may cu


def build_phone_orders(biz: Business, sales) -> pd.DataFrame | None:
    """Moi don cua so ban: khach_tra (tien ca don, da bot va da tru may cu), bot, thu_may_cu (tong gia thu may cu tru vao don),
    kem trang thai tu O. None neu so ban khong co cot 'Khach tra'."""
    if sales is None or not len(getattr(sales, "O", [])):
        return None
    O = sales.O
    line_t = None
    oids = set(O["order_id"].astype(str))
    for t, df in biz.tables.items():
        if _col(df, "khach tra") and _col(df, "stt", exact=True):
            line_t = t
            break
    if line_t is None:
        return None
    df = biz.tables[line_t]
    c_stt, c_pay, c_bot = _col(df, "stt", exact=True), _col(df, "khach tra"), _col(df, "bot", exact=True)
    stt = pd.to_numeric(df[c_stt], errors="coerce").astype("Int64").astype(str)
    # ma don cua O: '<file>/<sheet>#STT' khi moi sheet danh lai STT, hoac STT tron (so mot sheet danh lien)
    key = df["_file"].astype(str) + "/" + df["_sheet"].astype(str) + "#" + stt if "_sheet" in df and "_file" in df else stt
    if key.isin(oids).mean() < 0.9:
        return None
    agg = {"khach_tra": (c_pay, "first")}
    if c_bot:
        agg["bot"] = (c_bot, "first")
    c_tmc = _col(df, "thu may cu")
    if c_tmc:
        agg["thu_may_cu"] = (c_tmc, "first")
    po = df.assign(_k=key).groupby("_k").agg(**agg)
    po = po.reindex(O["order_id"].astype(str))
    out = pd.DataFrame({"order_id": O["order_id"].values, "order_date": O["order_date"].values, "status": O["status"].values,
                        "revenue_ok": O["revenue_ok"].values, "khach_tra": pd.to_numeric(po["khach_tra"], errors="coerce").values,
                        "bot": pd.to_numeric(po["bot"], errors="coerce").fillna(0).values if "bot" in po else 0.0})
    out["thu_may_cu"] = pd.to_numeric(po["thu_may_cu"], errors="coerce").fillna(0).values if "thu_may_cu" in po else 0.0
    for t, tm in ({} if "thu_may_cu" in po else biz.tables).items():  # so thu may cu rieng: gia thu tru vao don co STT o so ban, cung ngay
        c_price, c_link, c_date = _col(tm, "gia thu"), _col(tm, "tru vao"), _col(tm, "ngay", exact=True)
        if not (c_price and c_link and c_date):
            continue
        stt_of = out["order_id"].astype(str).str.extract(r"#(\d+)$")[0]
        idx = {(pd.Timestamp(d).normalize(), s): i for i, (d, s) in enumerate(zip(out["order_date"], stt_of))}
        for _, r in tm.iterrows():
            if pd.isna(r[c_link]) or pd.isna(r[c_date]):
                continue
            i = idx.get((pd.Timestamp(r[c_date]).normalize(), str(int(r[c_link]))))
            if i is not None:
                out.iat[i, out.columns.get_loc("thu_may_cu")] += float(r[c_price] or 0)
        break
    return out


def phone_order_total(po: pd.DataFrame, a, b, col: str) -> int:
    """Tong cot cua cac don khong huy, khong tra (§1) dat trong ky."""
    from .answers import round_money
    from .metrics import _period

    d = po[po["revenue_ok"] & _period(po, a, b)]
    return round_money(pd.to_numeric(d[col], errors="coerce").fillna(0).sum())


# ---------------------------------------------------------------------------- so bao hanh dien thoai


def build_phone_warranty(biz: Business) -> pd.DataFrame | None:
    """So bao hanh: moi lan khach mang may toi mot dong (ngay_nhan, imei, dong_may, ngay_mua). None neu khong co."""
    for t, df in biz.tables.items():
        c_recv, c_imei = _col(df, "ngay nhan"), _col(df, "imei")
        if not (c_recv and c_imei) or _col(df, "gia thu"):
            continue
        W = pd.DataFrame({"ngay_nhan": pd.to_datetime(df[c_recv], errors="coerce").dt.normalize().values, "imei": df[c_imei].astype(str).values})
        for role, keys in (("dong_may", ("may",)), ("ngay_mua", ("ngay mua",)), ("customer", ("khach",)), ("loi", ("loi",)), ("xu_ly", ("xu ly",))):
            c = _col(df, *keys, exact=(role == "dong_may"))
            if c is not None:
                W[role] = df[c].values if role != "ngay_mua" else pd.to_datetime(df[c], errors="coerce").dt.normalize().values
        return W.dropna(subset=["ngay_nhan"]).reset_index(drop=True)
    return None
