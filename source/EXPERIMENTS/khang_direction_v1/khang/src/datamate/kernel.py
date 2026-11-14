"""Sandbox chay code do model sinh ra.

Moi cua hang co mot tien trinh worker (don luong) giu san cac bang da lam sach. Moi cau hoi la
mot "phien" co namespace rieng (bien giu qua cac buoc). Bang duoc chia se o che do Copy-on-Write
cua pandas nen code cua phien nay sua bang khong anh huong phien khac.

Gioi han: moi lan chay co timeout mem (SIGALRM) va timeout cung (runner giet worker, nap lai).
/data chi doc; file tam ghi vao /tmp.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import multiprocessing as mp
import os
import pickle
import re
import signal
import threading
import time
import traceback
from typing import Any

MAX_OUTPUT = 6000


# ------------------------------------------------------------------ cache cua hang


# Tang khi doi danh sach phan dung san trong load_business_cached (cache cu thieu thuoc tinh moi se het han)
CACHE_VERSION = "2026-11-07.4"


def _fingerprint(path: str) -> str:
    h = hashlib.sha256(CACHE_VERSION.encode())
    for f in sorted(os.listdir(path)):
        p = os.path.join(path, f)
        if os.path.isfile(p):
            st = os.stat(p)
            h.update(f"{f}|{st.st_size}|{int(st.st_mtime)}".encode())
    src = os.path.dirname(os.path.abspath(__file__))
    for f in ("loader.py", "parsing.py", "readme.py", "semantic.py", "helpers.py", "debt.py", "imports.py", "inventory.py", "repair.py", "installment.py"):  # metrics.py khong anh huong du lieu cache  # doi code -> cache cu het han
        h.update(open(os.path.join(src, f), "rb").read())
    return h.hexdigest()[:16]


def load_business_cached(path: str, cache_dir: str):
    from .loader import load_business

    os.makedirs(cache_dir, exist_ok=True)
    bid = os.path.basename(os.path.normpath(path))
    cpath = os.path.join(cache_dir, f"{bid}-{_fingerprint(path)}.pkl")
    if os.path.exists(cpath):
        try:
            with open(cpath, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    biz = load_business(path)
    try:
        from .semantic import build_sales

        biz.sales = build_sales(biz)
    except Exception as exc:  # bang chuan la phan phu: loi thi bo qua, agent dung bang goc
        biz.sales = None
        biz.warnings.append(f"không dựng được bảng chuẩn S/O: {type(exc).__name__}: {exc}")
    try:
        from .imports import build_imports

        biz.imports = build_imports(biz)
    except Exception as exc:
        biz.imports = None
        biz.warnings.append(f"không dựng được bảng nhập hàng: {type(exc).__name__}: {exc}")
    try:
        from .inventory import build_inventory

        biz.inventory = build_inventory(biz, biz.sales)
    except Exception as exc:
        biz.inventory = None
        biz.warnings.append(f"không dựng được tồn kho: {type(exc).__name__}: {exc}")
    try:
        from .debt import build_debt

        biz.debt = build_debt(biz, biz.sales)
    except Exception as exc:
        biz.debt = None
        biz.warnings.append(f"không dựng được công nợ: {type(exc).__name__}: {exc}")
    biz.installments = None
    if biz.debt is None:  # so no / so gop tu ghi (cua hang dien thoai): tra gop la cong no
        try:
            from .installment import build_phone_debt

            biz.debt, biz.installments = build_phone_debt(biz)
        except Exception as exc:
            biz.warnings.append(f"không dựng được sổ nợ / trả góp: {type(exc).__name__}: {exc}")
    try:
        from .installment import build_phone_warranty

        biz.phone_warranty = build_phone_warranty(biz)
    except Exception as exc:
        biz.phone_warranty = None
        biz.warnings.append(f"không dựng được sổ bảo hành: {type(exc).__name__}: {exc}")
    try:
        from .installment import build_phone_orders

        biz.phone_orders = build_phone_orders(biz, biz.sales)
    except Exception as exc:
        biz.phone_orders = None
        biz.warnings.append(f"không dựng được bảng khách trả / thu máy cũ: {type(exc).__name__}: {exc}")
    try:
        from .repair import build_repair

        biz.repair = build_repair(biz)
    except Exception as exc:
        biz.repair = None
        biz.warnings.append(f"không dựng được sổ sửa xe: {type(exc).__name__}: {exc}")
    tmp = cpath + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(biz, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cpath)
    # bo ban cache cu cua cung cua hang (ma/du lieu da doi): tranh day dia khi chay lai nhieu lan
    for f in os.listdir(cache_dir):
        if f.startswith(f"{bid}-") and f.endswith(".pkl") and os.path.join(cache_dir, f) != cpath and f[len(bid) + 1:-4].isalnum():
            with contextlib.suppress(OSError):
                os.remove(os.path.join(cache_dir, f))
    return biz


def warm_cache(path: str, cache_dir: str) -> str:
    load_business_cached(path, cache_dir)
    return path


# ------------------------------------------------------------------ worker


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _expense_table(tables: dict):
    """So chi phi: bang ten co 'chi phi' (hoac cot 'Tien chi'), co cot ngay va cot tien -> (bang, cot ngay, cot tien, cot nhom)."""
    import pandas as pd

    from .loader import _split_name
    from .parsing import strip_accents

    for t, df in tables.items():
        tn = strip_accents(str(t)).lower().replace("_", "").replace(" ", "")
        cols = {c: " ".join(_split_name(c).split()) for c in df.columns if not str(c).startswith("_")}
        if "chiphi" not in tn and not any(v in ("tien chi", "so tien chi", "chi phi") for v in cols.values()):
            continue
        c_date = next((c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])), None)
        c_amt = next((c for c, v in cols.items() if pd.api.types.is_numeric_dtype(df[c]) and any(k in v for k in ("tien", "chi", "so tien"))), None)
        c_grp = next((c for c, v in cols.items() if df[c].dtype == object and any(k in v.split() for k in ("nhom", "loai", "khoan", "muc"))), None)
        if c_date and c_amt:
            return t, c_date, c_amt, c_grp
    return None


_BUILT_TABLES = ("S", "O", "S_gom_huy_tra", "O_gom_huy_tra", "N", "R", "P", "D_giao", "PO", "W_bh")


class _Tables(dict):
    """T: bang goc theo ten sheet/file, cong them bang dung san (S, O, P, D_giao…) cung la bien rieng."""

    raw: list = []
    built: list = []

    def __missing__(self, key):
        raise KeyError(f"{key!r} không có trong T. Bảng gốc: {', '.join(self.raw)}. Bảng dựng sẵn (dùng thẳng tên biến, cũng có trong T): "
                       f"{', '.join(self.built) or 'không có'}")


def _new_namespace(tables: dict, readme: str, today: str, lunar_path: str | None, sales=None, debt=None, imports=None, inventory=None, conversions=None, repair=None, installments=None, phone_orders=None, phone_warranty=None) -> dict:
    import datetime as dt

    import numpy as np
    import pandas as pd

    from . import helpers
    from .answers import AnswerError, normalize
    from .dates import LunarCalendar

    ns: dict[str, Any] = {"pd": pd, "np": np, "dt": dt, "datetime": dt}
    ns["T"] = {k: v.copy(deep=False) for k, v in tables.items()}
    for k, v in ns["T"].items():
        if k.isidentifier():
            ns.setdefault(k, v)
    for name in (
        "between", "norm", "same_name", "find_names", "status_of", "pct", "pct_change", "to_num",
        "parse_conversions", "days_in", "count_weekday", "round_money", "round2", "month_range",
        "quarter_range", "week_range", "shift_month",
    ):
        ns[name] = getattr(helpers, name)
    if sales is not None:
        from .metrics import Metrics

        S_all = sales.S.copy(deep=False)
        O_all = sales.O.copy(deep=False)
        m = Metrics(S_all, O_all)
        # bang gia / lich su gia (ngay tang gia theo rulebook): tinh mot lan moi worker
        ph = getattr(sales, "_price_hist", "chua")
        if isinstance(ph, str):
            try:
                from .metrics import build_price_history

                ph = build_price_history(tables, sales.S)
            except Exception:
                ph = None
            try:
                sales._price_hist = ph
            except Exception:
                pass
        m.price_hist = ph
        # mac dinh S/O la ban DA LOC (chi dong/don tinh doanh thu); ban gom don huy/tra dat ten ro nghia
        # (ten "O_all" lam model dung nham cho cau "tong bao nhieu don")
        # so ban chi ghi ma nhan vien (NV001): them cot employee_name lay ho ten tu bang nhan vien
        from .staff import employee_names

        emap = employee_names(tables, S_all["employee"]) if "employee" in S_all else None
        if emap:
            for df in (S_all, O_all, m.SR, m.OR):
                if "employee" in df:
                    df["employee_name"] = df["employee"].astype(str).map(emap)
        ns["S_gom_huy_tra"], ns["O_gom_huy_tra"] = S_all, O_all
        ns["S"], ns["O"] = m.SR, m.OR
        ns["SR"], ns["OR"] = m.SR, m.OR
        for fn in ("revenue", "n_orders", "qty_sold", "top_items", "cancel_rate", "n_cancelled", "avg_order_value", "customers", "price_on", "unsold_items", "order_values", "qty_unit", "unit_is_base", "by_day", "by_hour", "gross_profit", "price_effect", "price_change_date", "adjustments"):
            ns[fn] = getattr(m, fn)
    from .dates import date_args

    exp = _expense_table(tables)
    if exp is not None:
        t_exp, c_date, c_amt, c_grp = exp

        def expenses(a, b=None, nhom=None):
            """Tong chi trong so chi phi (moi khoan) co ngay trong ky; loc nhom= (cot nhom chi)."""
            from .answers import round_money
            from .metrics import _filters, _period

            d = tables[t_exp]
            m = _period(d, a, b, col=c_date)
            if nhom is not None and c_grp:
                m &= _filters(d.rename(columns={c_grp: "nhom"}), {"nhom": nhom}).values
            return round_money(pd.to_numeric(d.loc[m, c_amt], errors="coerce").sum())

        ns["expenses"] = expenses
        if "gross_profit" in ns:
            def net_profit(a, b=None):
                """Lai rong (rulebook) = lai gop trong ky - moi khoan chi trong so chi phi cung ky."""
                return ns["gross_profit"](a, b) - expenses(a, b)

            ns["net_profit"] = net_profit
    if repair is not None:
        from .repair import RepairMetrics

        rm = RepairMetrics(repair)
        ns["P"] = repair.P.copy(deep=False)
        for fn in ("tickets", "n_tickets", "avg_ticket_value", "warranty_rate", "n_vehicles", "regular_vehicles", "by_tho", "commission", "by_dong_xe"):
            ns[fn] = getattr(rm, fn)
        if sales is None:  # tiem chi co so sua xe: ten ham quen thuoc tro toi ham cua so sua xe
            ns.update(revenue=rm.revenue, n_orders=rm.n_tickets, avg_order_value=rm.avg_ticket_value, by_day=rm.by_day)
    if inventory is not None:
        from . import inventory as _inv

        def stock_at(*d):
            """Ton kho luc het ngay D: stock_at('2026-06-12') hoac stock_at(2026, 6, 12)."""
            return _inv.stock_at(inventory, date_args(d, 1, "stock_at")[0])

        def stock_value_at(*d):
            """Gia tri hang ton theo gia von danh muc luc het ngay D."""
            return _inv.stock_value_at(inventory, date_args(d, 1, "stock_value_at")[0])

        def lot_stock_at(*d):
            """Ton tung lo luc het ngay D."""
            return _inv.lot_stock_at(inventory, date_args(d, 1, "lot_stock_at")[0])

        def stock_units(*d, item=None, unit=None):
            """Ton cua mat hang luc het ngay D quy ra [so <unit> nguyen, so le theo don vi co ban]:
            stock_units(HOM_NAY, item='Cát xây tô', unit='xe 4m3') -> [19, 2]."""
            if item is None and len(d) >= 3 and isinstance(d[-1], str) and isinstance(d[-2], str):
                d, item, unit = d[:-2], d[-2], d[-1]
            day = date_args(d, 1, "stock_units")[0]
            st = _inv.stock_at(inventory, day)
            if st is None or item is None or unit is None:
                return None
            from .metrics import _filters
            from .parsing import norm_key

            row = st[_filters(st, {"item": item})]
            if row.empty:
                return None
            ton = float(row["ton"].sum())
            names = {norm_key(x) for x in row["item"].dropna().astype(str)} | {norm_key(x) for x in row.get("item_code", pd.Series(dtype=str)).dropna().astype(str)}
            u = norm_key(unit)
            fac = None
            for (k, cu), f in (conversions or {}).items():
                if k in names and (cu == u or cu.startswith(u + " ") or u.startswith(cu + " ")):
                    fac = f
                    break
            if not fac:  # cau hoi go khong dau ("cuon 10m"): so khong dau
                from .parsing import norm_key_ascii
                ua = norm_key_ascii(unit)
                for (k, cu), f in (conversions or {}).items():
                    ca = norm_key_ascii(cu)
                    if k in names and (ca == ua or ca.startswith(ua + " ") or ua.startswith(ca + " ")):
                        fac = f
                        break
            if not fac:
                raise ValueError(f"không thấy hệ số quy đổi '{unit}' của {item!r}; xem bảng quy đổi / README")
            big = int(ton // fac)
            rest = ton - big * fac
            return [big, int(rest) if float(rest).is_integer() else round(rest, 2)]

        ns.update(stock_at=stock_at, stock_value_at=stock_value_at, lot_stock_at=lot_stock_at, stock_units=stock_units)
        if inventory.lots is not None and inventory.lot_moves is not None:
            def _lot_args(d):
                if not d:
                    raise ValueError("cần ngày tính tồn D, vd expiring_lots(HOM_NAY, ('2026-06-28', '2026-10-26'))")
                day, rest = d[0], d[1:]
                return day, (rest[0] if rest else None), (rest[1] if len(rest) > 1 else None)

            def expiring_lots(*d, item=None):
                """Lo con ton luc het ngay D co han dung trong ky: expiring_lots(D, (a, b)) hoac expiring_lots(D, a, b);
                cot von = ton x gia nhap 1 don vi co ban cua lo; sap han som truoc."""
                return _inv.expiring_lots(inventory, *_lot_args(d), item=item)

            def expiring_items(*d):
                """Mat hang con lo ton co han dung trong ky, thu tu han som nhat cua mat hang (cung han theo ma)."""
                return _inv.expiring_items(inventory, *_lot_args(d))

            def nearest_expiry(*d, item=None):
                """Han dung 'YYYY-MM-DD' cua lo gan het han nhat (con ton, chua qua han tai D) cua mot mat hang: nearest_expiry(D, item='...')."""
                if item is None and d and isinstance(d[-1], str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", d[-1]):
                    d, item = d[:-1], d[-1]
                if item is None:
                    raise ValueError("cần item=<tên mặt hàng>")
                return _inv.nearest_expiry(inventory, d[0] if d else today, item)

            ns.update(expiring_lots=expiring_lots, expiring_items=expiring_items, nearest_expiry=nearest_expiry)
    if imports is not None:
        from .imports import ImportMetrics

        im = ImportMetrics(imports)
        ns.update(import_cost=im.import_cost, avg_import_price=im.avg_import_price, top_suppliers=im.top_suppliers)
        ns["N"] = imports.N.copy(deep=False)
        ns["R"] = imports.receipts.copy(deep=False)

        def net_import(*d, item=None, unit=None):
            """Nhap tru ban (§5) cua mat hang trong ky, quy ve <unit> (mac dinh don vi co ban): so luong nhap (bo phieu huy,
            moi dong quy ve don vi co ban theo bang quy doi) tru so luong ban (don tinh doanh thu). Khong quy doi duoc -> loi."""
            from .metrics import _filters, _period
            from .parsing import norm_key, norm_key_ascii

            a, b = date_args(d, 2, "net_import") if not (len(d) == 1 and isinstance(d[0], (tuple, list))) else date_args(tuple(d[0]), 2, "net_import")
            Nn = imports.N
            x = Nn[_period(Nn, a, b, col="date") & _filters(Nn, {"item": item})]
            if "cancelled" in x:
                x = x[~x["cancelled"].fillna(False).astype(bool)]
            names = {norm_key(v) for v in x["item"].dropna().astype(str)} | {norm_key(v) for v in x.get("item_code", pd.Series(dtype=str)).dropna().astype(str)}

            def fac(u):
                if u is None or (isinstance(u, float) and u != u):
                    return 1.0
                ua = norm_key_ascii(u)
                SR = ns.get("S")
                if SR is not None and len(SR) and "unit" in SR:  # don vi co ban theo dong ban (he so 1), roi he so tu dong ban
                    sr = SR[_filters(SR, {"item": item})]
                    if "qty_base" in sr and len(sr):
                        r = (sr["qty_base"] / sr["qty"]).where(sr["qty"] != 0)
                        byu = r.groupby(sr["unit"].map(norm_key_ascii)).median()
                        if ua in byu.index and byu[ua] == byu[ua]:
                            return float(byu[ua])
                for (k, cu), f in (conversions or {}).items():
                    if k in names and norm_key_ascii(cu) == ua:
                        return float(f)
                raise ValueError(f"không thấy hệ số quy đổi '{u}' của {item!r}")

            imp = float(sum(q * fac(u) for q, u in zip(x["qty"], x["unit"] if "unit" in x else [None] * len(x))))
            sold = float(ns["qty_sold"](a, b, item=item)) if "qty_sold" in ns else 0.0
            v = (imp - sold) / (fac(unit) if unit else 1.0)
            return int(v) if float(v).is_integer() else round(v, 2)

        ns["net_import"] = net_import
    if debt is not None:
        from . import debt as _debt

        def debt_as_of(*d):
            """Moi don con no > 0 luc het ngay D: debt_as_of('2025-08-24') hoac debt_as_of(2025, 8, 24)."""
            return _debt.debt_as_of(debt, date_args(d, 1, "debt_as_of")[0])

        def debt_by_customer(*d):
            """No theo khach (bo khach le) luc het ngay D, giam dan."""
            return _debt.debt_by_customer(debt, date_args(d, 1, "debt_by_customer")[0])

        def debt_added(*d):
            """No them trong ky [a, b]: debt_added('2025-08-01', '2025-08-31')."""
            a, b = date_args(d, 2, "debt_added")
            return _debt.debt_added(debt, a, b)

        def debt_paid(*d):
            """Cac lan tra no (sau ngay mua) co ngay tra trong ky [a, b]."""
            a, b = date_args(d, 2, "debt_paid")
            return _debt.debt_paid(debt, a, b)

        def debt_aging(*d, edges=(7, 15)):
            """No con lai luc het ngay D chia theo tuoi no (so ngay tu ngay dat): mac dinh [0-7, 8-15, tren 15 ngay] -> list tien."""
            from .answers import round_money

            x = _debt.debt_as_of(debt, date_args(d, 1, "debt_aging")[0])
            lo, out = 0, []
            for e in edges:
                out.append(round_money(x.loc[x["tuoi_no"].between(lo, e), "con_no"].sum()))
                lo = e + 1
            out.append(round_money(x.loc[x["tuoi_no"] >= lo, "con_no"].sum()))
            return out

        ns.update(debt_as_of=debt_as_of, debt_by_customer=debt_by_customer, debt_added=debt_added, debt_paid=debt_paid, debt_aging=debt_aging)
    from .delivery import build_delivery

    D_giao = build_delivery(tables, sales.O if sales is not None else None)
    if D_giao is not None:
        from .delivery import delivery_rounds as _rounds

        ns["D_giao"] = D_giao

        def delivery_rounds(a, b=None):
            """So dot giao co ngay giao trong ky (tinh den het hom nay), ke ca dot cua don sau do tra hang."""
            from .metrics import _ab

            a, b = _ab(a, b)
            return _rounds(D_giao, a, b, today)

        ns["delivery_rounds"] = delivery_rounds
    if phone_orders is not None:
        from .installment import phone_order_total

        ns["PO"] = phone_orders.copy(deep=False)

        def customer_pays(a, b=None):
            """Tong khach phai tra (rulebook dien thoai): tien don sau bot va sau tru may cu, gom ca phan con no; don khong huy, khong tra."""
            from .metrics import _ab

            a, b = _ab(a, b)
            return phone_order_total(phone_orders, a, b, "khach_tra")

        def trade_in_total(a, b=None):
            """Tong tien thu may cu tru vao cac don khong huy, khong tra dat trong ky (khong tru vao doanh thu)."""
            from .metrics import _ab

            a, b = _ab(a, b)
            return phone_order_total(phone_orders, a, b, "thu_may_cu")

        def discount_total(a, b=None):
            """Tong tien bot cua cac don khong huy, khong tra dat trong ky."""
            from .metrics import _ab

            a, b = _ab(a, b)
            return phone_order_total(phone_orders, a, b, "bot")

        ns.update(customer_pays=customer_pays, trade_in_total=trade_in_total, discount_total=discount_total)
    if phone_warranty is not None:
        ns["W_bh"] = phone_warranty.copy(deep=False)

        def phone_warranty_rate(a, b=None):
            """Ti le bao hanh (%) = so luot bao hanh co ngay nhan trong ky / so may ban trong ky (don khong huy, khong tra;
            may = hang co IMEI: dm_loai chua 'may'). Luot bao hanh cua may ban tu ky truoc van tinh."""
            from .answers import round2
            from .metrics import _ab, _period

            a, b = _ab(a, b)
            W = phone_warranty
            n_w = int(_period(W, a, b, col="ngay_nhan").sum())
            S_ = ns.get("S")
            if S_ is None or "dm_loai" not in S_:
                raise ValueError("không xác định được máy bán (thiếu loại hàng)")
            m = S_[_period(S_, a, b) & S_["dm_loai"].astype(str).str.contains("máy|may", case=False)]
            n_m = float(m["qty_base"].sum()) if "qty_base" in m else float(m["qty"].sum())
            return round2(100.0 * n_w / n_m) if n_m else None

        ns["phone_warranty_rate"] = phone_warranty_rate
    if installments is not None:
        from . import installment as _inst

        def installments_as_of(*d):
            """Moi hop dong tra gop luc het ngay D (so ky toi han, con lai, tien qua han)."""
            return _inst.installments_as_of(installments, date_args(d, 1, "installments_as_of")[0])

        ns["installments_as_of"] = installments_as_of
    ns["README"] = readme
    ns["TODAY"] = pd.Timestamp(today)
    lunar = LunarCalendar.load(lunar_path) if lunar_path and os.path.exists(lunar_path) else None

    def tet_range(year: int):
        if lunar is None:
            raise ValueError("khong co calendar.csv")
        r = lunar.tet_range(int(year))
        if r is None:
            raise ValueError(f"lich am khong co Tet {year}")
        return r

    from .dates import period_vars

    # ky trong sandbox la pd.Timestamp (khong phai datetime.date) de so sanh truc tiep voi cot ngay cua pandas:
    # N['date'] >= THANG(1, 2026)[0] chay duoc; cac ham van nhan ca date / chuoi ISO.
    def _ts(x):
        if isinstance(x, dt.date) and not isinstance(x, pd.Timestamp):
            return pd.Timestamp(x)
        if isinstance(x, tuple):
            return tuple(_ts(v) for v in x)
        return x

    def _wrap(fn):
        import functools

        @functools.wraps(fn)
        def inner(*a, **k):
            return _ts(fn(*a, **k))

        return inner

    ns["tet_range"] = _wrap(tet_range)
    for name in ("month_range", "quarter_range", "week_range"):
        ns[name] = _wrap(ns[name])
    ns.update({k: (_wrap(v) if callable(v) else _ts(v)) for k, v in period_vars(today, lunar).items()})
    ns["__answer__"] = None

    def answer(type_, value=None, reason=None):
        try:
            new = normalize(type_, value, reason)
        except AnswerError as exc:
            raise AnswerError(f"answer() sai format: {exc}") from None
        prev = ns.get("__answer__")
        if prev is not None and prev != new:
            # public: model goi answer() hai lan cho cau hoi hai so do, lan sau de lan truoc
            raise AnswerError("answer() gọi lần thứ hai trong cùng lượt với đáp án khác: mỗi câu chỉ nộp MỘT đáp án. Câu hỏi nhiều số đo "
                              "(lần lượt, từng bên, X và Y) thì nộp một list đúng thứ tự câu hỏi, vd answer('list', [a, b]).")
        ns["__answer__"] = new
        print("ĐÃ NỘP:", new)

    ns["answer"] = answer
    # model hay tim bang dung san trong T (T["P"], T["D_giao"]): dua cac bang dung san vao T (khong de ten bang goc),
    # ten khong co thi bao ro danh sach bang goc va bang dung san thay vi KeyError tran
    raw = list(ns["T"])
    built = [k for k in _BUILT_TABLES if isinstance(ns.get(k), pd.DataFrame)]
    T = _Tables(ns["T"])
    for k in built:
        T.setdefault(k, ns[k])
    T.raw, T.built = raw, built
    ns["T"] = T
    return ns


_PROTECTED = ("T", "S", "O", "S_gom_huy_tra", "O_gom_huy_tra", "SR", "OR", "N", "R", "P", "D_giao", "PO", "W_bh")
_READERS = ("read_csv", "read_excel", "read_parquet", "read_json", "read_table", "read_sql", "read_sql_query", "read_pickle", "read_feather", "read_fwf")


class _GuardError(Exception):
    pass


def _guard(tree: ast.AST, ns: dict) -> None:
    """Chan code thay du lieu that bang du lieu tu tao: gan lai T/S/O/bang goc (khong tu tham chieu chinh no) hoac doc file.
    Da gap tren bo public: model tao lai T bang pd.read_csv('...csv') roi pd.DataFrame so lieu bia."""
    raw = getattr(ns.get("T"), "raw", None) or []
    prot = set(_PROTECTED) | {k for k in raw if isinstance(k, str) and k.isidentifier() and k in ns}

    def names(t):
        if isinstance(t, ast.Name):
            yield t.id
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                yield from names(e)
        elif isinstance(t, ast.Starred):
            yield from names(t.value)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            for t in targets:
                for name in names(t):
                    if name in prot and not (used & prot):  # ve phai khong lay tu bang that nao: du lieu tu tao
                        raise _GuardError(
                            f"không gán lại biến {name}: {name} là dữ liệu THẬT của shop đã nạp sẵn, không tự tạo bảng hay số liệu. "
                            f"Đặt tên biến khác, vd {name}_loc = {name}[...]; xem cột bằng print({name}.columns) hoặc T.raw.")
        elif isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr in _READERS) or (isinstance(f, ast.Name) and f.id in ("open",) + _READERS):
                raise _GuardError("không đọc file: dữ liệu đã nạp sẵn trong T (bảng gốc) và S, O… (bảng dựng sẵn); dùng thẳng các biến đó.")


def _run_cell(ns: dict, code: str, soft_timeout: float) -> dict:
    out = io.StringIO()
    err = None
    ns["__answer__"] = None
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, soft_timeout)
    t0 = time.time()
    try:
        tree = ast.parse(code, "<code>", "exec")
        _guard(tree, ns)
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(tree, "<code>", "exec"), ns)
            if last_expr is not None:
                val = eval(compile(last_expr, "<code>", "eval"), ns)
                if val is not None:
                    if type(val).__name__ == "DataFrame":
                        print(val.to_string(max_rows=40, max_cols=20))
                    elif hasattr(val, "to_string") and type(val).__name__ != "str":
                        print(val.to_string(max_rows=40))  # Series.to_string khong co max_cols
                    else:
                        print(repr(val))
    except _Timeout:
        err = f"TimeoutError: code chạy quá {soft_timeout:.0f} giây; hãy viết cách tính nhanh hơn (vector hoá, lọc trước)."
    except _GuardError as exc:
        err = f"LỖI QUY TẮC: {exc}"
    except SystemExit:  # exit()/quit()/sys.exit() trong code: chi dung cell, giu bien va dap an da nop (khong tat worker)
        pass
    except Exception as exc:  # loi trong code cua model: tra lai de model sua
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb = [line for line in tb if 'File "<code>"' in line or not line.startswith("  File")]
        err = "".join(tb[-6:]).strip()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    text = out.getvalue()
    if len(text) > MAX_OUTPUT:
        text = text[: MAX_OUTPUT // 2] + f"\n... (bỏ bớt {len(text) - MAX_OUTPUT} ký tự) ...\n" + text[-MAX_OUTPUT // 2 :]
    return {
        "stdout": text,
        "error": err,
        "answer": ns.get("__answer__"),
        "seconds": round(time.time() - t0, 3),
    }


def worker_main(conn, biz_path: str, cache_dir: str, lunar_path: str | None):
    import pandas as pd

    pd.options.mode.copy_on_write = True
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    biz = load_business_cached(biz_path, cache_dir)
    tables, readme = biz.tables, biz.readme.text
    sales = getattr(biz, "sales", None)
    debt = getattr(biz, "debt", None)
    imports = getattr(biz, "imports", None)
    inventory = getattr(biz, "inventory", None)
    repair = getattr(biz, "repair", None)
    installments = getattr(biz, "installments", None)
    phone_orders = getattr(biz, "phone_orders", None)
    phone_warranty = getattr(biz, "phone_warranty", None)
    try:
        from .semantic import _conversion_table

        conversions = _conversion_table(biz)
    except Exception:  # khong co bang quy doi: stock_units bao loi ro rang
        conversions = {}
    sessions: dict[str, dict] = {}
    conn.send({"ready": True})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        op = msg.get("op")
        if op == "stop":
            break
        if op == "end":
            sessions.pop(msg["sid"], None)
            conn.send({"ok": True})
            continue
        if op == "exec":
            sid = msg["sid"]
            if sid not in sessions:
                sessions[sid] = _new_namespace(tables, readme, msg["today"], lunar_path, sales, debt, imports, inventory, conversions, repair, installments, phone_orders, phone_warranty)
            res = _run_cell(sessions[sid], msg["code"], msg.get("soft_timeout", 30.0))
            try:
                conn.send(res)
            except Exception as exc:  # ket qua khong pickle duoc
                conn.send({"stdout": res.get("stdout", ""), "error": f"loi gui ket qua: {exc}", "answer": None, "seconds": 0})


class BusinessKernel:
    """Phia runner: mot worker cho mot cua hang, chay tuan tu, tu khoi dong lai khi treo."""

    def __init__(self, biz_path: str, cache_dir: str, lunar_path: str | None, hard_timeout: float = 90.0):
        self.biz_path, self.cache_dir, self.lunar_path = biz_path, cache_dir, lunar_path
        self.hard_timeout = hard_timeout
        self.lock = threading.Lock()
        self.proc = None
        self.conn = None
        self.generation = 0

    def _start(self):
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(child, self.biz_path, self.cache_dir, self.lunar_path), daemon=True)
        self.proc.start()
        child.close()
        self.conn = parent
        if not parent.poll(600):
            raise RuntimeError(f"worker {self.biz_path} khong khoi dong duoc")
        parent.recv()
        self.generation += 1

    def _kill(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.join(5)
        self.proc = self.conn = None

    def execute(self, sid: str, code: str, today: str, soft_timeout: float = 30.0) -> dict:
        with self.lock:
            if self.proc is None or not self.proc.is_alive():
                self._start()
            try:
                self.conn.send({"op": "exec", "sid": sid, "code": code, "today": today, "soft_timeout": soft_timeout})
                if self.conn.poll(self.hard_timeout):
                    return self.conn.recv()
                err = f"TimeoutError: code chạy quá {self.hard_timeout:.0f} giây và bị dừng; biến của phiên đã mất, hãy chạy lại từ đầu với cách tính nhanh hơn."
            except (EOFError, BrokenPipeError, OSError) as exc:
                err = f"Sandbox lỗi ({exc}); biến của phiên đã mất, hãy chạy lại."
            self._kill()
            return {"stdout": "", "error": err, "answer": None, "seconds": self.hard_timeout}

    def end(self, sid: str):
        with self.lock:
            if self.proc is not None and self.proc.is_alive():
                try:
                    self.conn.send({"op": "end", "sid": sid})
                    if self.conn.poll(10):
                        self.conn.recv()
                except Exception:
                    self._kill()

    def close(self):
        with self.lock:
            if self.proc is not None and self.proc.is_alive():
                try:
                    self.conn.send({"op": "stop"})
                except Exception:
                    pass
            self._kill()


class KernelPool:
    def __init__(self, cache_dir: str, lunar_path: str | None, workers_per_business: int = 2):
        self.cache_dir, self.lunar_path = cache_dir, lunar_path
        self.n = workers_per_business
        self.kernels: dict[str, list[BusinessKernel]] = {}
        self.lock = threading.Lock()
        self.rr: dict[str, int] = {}

    def get(self, biz_path: str, sid: str) -> BusinessKernel:
        with self.lock:
            ks = self.kernels.setdefault(biz_path, [BusinessKernel(biz_path, self.cache_dir, self.lunar_path) for _ in range(self.n)])
        # cung mot phien luon ve cung worker (giu bien)
        idx = int(hashlib.md5(sid.encode()).hexdigest(), 16) % len(ks)
        return ks[idx]

    def close(self):
        for ks in self.kernels.values():
            for k in ks:
                k.close()
