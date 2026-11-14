"""Nap du lieu mot cua hang thanh cac bang pandas da lam sach.

Cac buoc (deu co dinh, khong dung model):
1. Doc moi file csv / xlsx (moi sheet) / sqlite (moi bang).
2. Tim dong header theo ten cot README liet ke (bo dong tieu de, header 2 tang).
3. Gop cac file/sheet cung mot muc README hoac cung bo cot (file theo thang, sheet theo thang,
   danh sach dai tach nhieu file).
4. Doan kieu tung cot: ngay, gio, so, dung/sai, chuoi; chuyen '90k', '12tr460', '1 thg 3 2025'...
5. Luat §8 rulebook: o trong o cac dong sau cua cung mot don lay theo dong dau don;
   hai dong cung don giong het nhau la loi xuat file, tinh mot lan.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .parsing import (
    clean_str,
    is_blank,
    norm_key_ascii,
    parse_bool,
    parse_datetime,
    parse_number,
    parse_time,
    strip_accents,
    year_from_name,
)
from .readme import Readme, Section, parse_readme

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

META_COLS = ("_file", "_sheet")


@dataclass
class ColumnInfo:
    name: str
    kind: str  # date | datetime | time | number | bool | text | empty
    description: str | None = None
    parse_rate: float = 1.0
    examples: list[tuple[Any, Any]] = field(default_factory=list)  # (raw, clean) khi co doi
    unit_column: str | None = None
    failed: list[Any] = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    sources: list[tuple[str, str | None]]
    header_rows: list[int]
    n_rows: int
    columns: dict[str, ColumnInfo]
    key: str | None = None
    ffilled: list[str] = field(default_factory=list)
    duplicates_dropped: int = 0
    dropped_rows: int = 0
    canonicalized: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # cot -> [(cach viet khac, ten chuan)]


@dataclass
class Business:
    id: str
    path: str
    readme: Readme
    tables: dict[str, pd.DataFrame]
    info: dict[str, TableInfo]
    warnings: list[str] = field(default_factory=list)
    sales: object = None  # semantic.SalesModel (bang chuan S/O) neu dung duoc
    debt: object = None  # debt.DebtModel (cong no) neu dung duoc
    imports: object = None  # imports.ImportModel (nhap hang N/R) neu dung duoc
    inventory: object = None  # inventory.InventoryModel (ton kho) neu dung duoc

    @property
    def data_end(self) -> dt.date | None:
        return self.readme.data_end


# ----------------------------------------------------------------- doc file tho


def _read_xlsx(path: str) -> list[tuple[str, list[list[Any]]]]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        out.append((ws.title, rows))
    wb.close()
    return out


def _read_csv(path: str) -> list[list[Any]]:
    raw = open(path, "rb").read()
    text = None
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sample = (text or "")[:20000]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    return [row for row in csv.reader(io.StringIO(text or ""), delimiter=delim)]


def _read_sqlite(path: str) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
        return {n: pd.read_sql_query(f'select * from "{n}"', con) for n in names if not n.startswith("sqlite_")}
    finally:
        con.close()


# ----------------------------------------------------------------- header


def _cell_str(x: Any) -> str:
    x = clean_str(x)
    return "" if is_blank(x) else str(x)


def _find_header(rows: list[list[Any]], expected: set[str]) -> int | None:
    best, best_i = 0, None
    exp_norm = {norm_key_ascii(e) for e in expected}
    for i, row in enumerate(rows[:40]):
        cells = [_cell_str(c) for c in row]
        score = sum(1 for c in cells if c and (c in expected or norm_key_ascii(c) in exp_norm))
        if score > best:
            best, best_i = score, i
    if best_i is not None and best >= max(1, min(2, len(expected))):
        return best_i
    # khong co mo ta README: dong dau tien co >= 2 o chu, chiem >= nua hang, dong sau co du lieu
    for i, row in enumerate(rows[:40]):
        cells = [c for c in row if not is_blank(c)]
        if len(cells) >= 2 and all(isinstance(c, str) for c in cells) and len(cells) >= 0.5 * len(row):
            if i + 1 < len(rows) and any(not is_blank(c) for c in rows[i + 1]):
                return i
    return 0 if rows else None


def _rows_to_frame(rows: list[list[Any]], h: int) -> tuple[pd.DataFrame, int]:
    header = [_cell_str(c) for c in rows[h]]
    width = len(header)
    names, seen = [], {}
    for j, name in enumerate(header):
        name = name or f"cot_{j + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    data, dropped = [], 0
    for r in rows[h + 1 :]:
        r = list(r)[:width] + [None] * max(0, width - len(r))
        if all(is_blank(c) for c in r):
            continue
        if [_cell_str(c) for c in r] == header:  # header lap lai giua file
            dropped += 1
            continue
        data.append([clean_str(c) for c in r])
    df = pd.DataFrame(data, columns=names, dtype=object)
    for name in names:
        if name.startswith("cot_") and df[name].map(is_blank).all():
            df = df.drop(columns=[name])
    return df, dropped


# ----------------------------------------------------------------- kieu cot

_RE_ID_DESC = re.compile(
    r"(^|[\s(])(mã|số điện thoại|biển số|số lô|barcode|mã vạch|imei|số phiếu|số hoá đơn|số hóa đơn|số đơn(?! vị)|số thứ tự đơn)",
    re.IGNORECASE,
)
_ID_TOKENS = {"ma", "id", "sdt", "imei", "barcode", "msp", "mahh", "masp", "mst", "cccd", "cmnd", "sodienthoai", "dienthoai"}
_ID_PHRASES = ("dien thoai", "bien so", "so phieu", "so hd", "ma vach", "so don ", "so hoa don", "ma lo", "so lo")
_DATE_HINT = ("ngay", "date", "thoi gian", "thoi diem", "time", "gio", "han", "luc", "sinh nhat")
_NOT_DATE_DESC = ("số ngày", "kỳ hạn", "hệ số", "số giờ")


def _split_name(name: str) -> str:
    """'ThoiGianLap' -> 'thoi gian lap'; 'Mã KH mua' -> 'ma kh mua'."""
    chars = []
    for i, ch in enumerate(name):
        if i and ch.isupper() and (
            name[i - 1].islower() or (name[i - 1].isupper() and i + 1 < len(name) and name[i + 1].islower())
        ):  # 'ThoiGian' -> 'Thoi Gian'; 'SLQuyDoi' -> 'SL Quy Doi'
            chars.append(" ")
        chars.append(ch)
    return norm_key_ascii("".join(chars)).replace("_", " ")


def _is_id_like(name: str, desc: str | None, vals: pd.Series) -> bool:
    d = (desc or "").strip()
    if d and _RE_ID_DESC.search(d) and not d.lower().startswith(("số lượng", "tên")):
        return True
    n = _split_name(name)
    toks = set(n.split())
    if toks & _ID_TOKENS or any(p in n + " " for p in _ID_PHRASES):
        return True
    strs = vals[vals.map(lambda x: isinstance(x, str))]
    if len(strs) and strs.str.match(r"^0\d{8,10}$").mean() > 0.5:  # so dien thoai
        return True
    return False


def _date_hinted(name: str, desc: str | None) -> bool:
    d = (desc or "").lower()
    if any(k in d for k in _NOT_DATE_DESC):
        return False
    n = _split_name(name)
    dd = strip_accents(d)
    return any(re.search(rf"(^|\s){k}", n) for k in _DATE_HINT) or any(
        k in dd for k in ("ngay", "thoi diem", "thoi gian", "han ", "gio ")
    )


def _uniq_map(series: pd.Series, fn) -> pd.Series:
    """Ap fn len gia tri duy nhat (nhanh hon nhieu voi bang lon)."""
    cache: dict[Any, Any] = {}
    out = []
    for x in series:
        try:
            k = (type(x), x)
            if k not in cache:
                cache[k] = fn(x)
            out.append(cache[k])
        except TypeError:
            out.append(fn(x))
    return pd.Series(out, index=series.index, dtype=object)


def _infer_column(series: pd.Series, name: str, desc: str | None, year_hint: pd.Series | None):
    blank = series.map(is_blank)
    vals = series[~blank]
    info = ColumnInfo(name=name, kind="text", description=desc)
    if len(vals) == 0:
        info.kind = "empty"
        return pd.Series([None] * len(series), index=series.index, dtype=object), None, info
    # ---- ngay / gio
    if _date_hinted(name, desc) or vals.map(lambda x: isinstance(x, (dt.date, dt.datetime))).mean() > 0.5:
        yh = year_hint if year_hint is not None else pd.Series([None] * len(series), index=series.index)
        pairs = pd.Series(list(zip(series, yh)), index=series.index)
        parsed = _uniq_map(pairs, lambda t: None if is_blank(t[0]) else parse_datetime(t[0], t[1]))
        ok = parsed[~blank].notna().mean()
        if ok >= 0.9:
            p = parsed.dropna()
            has_time = any((d.hour, d.minute, d.second) != (0, 0, 0) for d in p.head(5000))
            info.kind = "datetime" if has_time else "date"
            info.parse_rate = float(ok)
            info.examples = _examples(series, parsed)
            info.failed = list(series[parsed.isna() & ~blank].head(5))
            return pd.to_datetime(parsed), None, info
        times = _uniq_map(vals, parse_time)
        if times.notna().mean() >= 0.9:
            info.kind = "time"
            out = _uniq_map(series, lambda x: (lambda t: t.strftime("%H:%M:%S") if t else None)(parse_time(x)))
            info.parse_rate = float(times.notna().mean())
            info.examples = _examples(series, out)
            return out, None, info
    # ---- dung / sai
    if vals.map(lambda x: isinstance(x, bool) or (isinstance(x, str) and x.strip().lower() in ("true", "false"))).mean() >= 0.95:
        info.kind = "bool"
        return series.map(parse_bool), None, info
    # ---- so
    if not _is_id_like(name, desc, vals):
        parsed = _uniq_map(vals, parse_number)
        nums = parsed.map(lambda t: t[0])
        units = parsed.map(lambda t: t[1])
        ok = nums.notna().mean()
        has_unit = units.notna()
        # dia chi '118 Tinh Lo 10' khong phai so: phan chu qua da dang
        rest = units[has_unit]
        diverse_rest = bool(len(rest)) and has_unit.mean() > 0.2 and (
            rest.map(lambda u: len(str(u).split())).mean() >= 2.5 or rest.nunique() > 60
        )
        if ok >= 0.95 and not diverse_rest:
            full = pd.Series([None] * len(series), index=series.index, dtype=object)
            full.loc[vals.index] = nums
            info.kind = "number"
            info.parse_rate = float(ok)
            info.examples = _examples(series, full)
            info.failed = list(vals[nums.isna()].head(5))
            unit_series = None
            if has_unit.any():
                unit_series = pd.Series([None] * len(series), index=series.index, dtype=object)
                unit_series.loc[vals.index] = units
                info.unit_column = f"{name}__dv"
            return pd.to_numeric(full, errors="coerce"), unit_series, info

    def _txt(x):
        if is_blank(x):
            return None
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        if isinstance(x, (dt.datetime, dt.date)):
            return x.isoformat()
        return str(x)

    return _uniq_map(series, _txt), None, info


def _examples(raw: pd.Series, clean: pd.Series, k: int = 3) -> list[tuple[Any, Any]]:
    out, seen = [], set()
    for r, c in zip(raw, clean):
        if is_blank(r) or is_blank(c) or not isinstance(r, str):
            continue
        cs = c.isoformat() if hasattr(c, "isoformat") else c
        if str(r) != str(cs) and r not in seen:
            seen.add(r)
            out.append((r, cs))
            if len(out) >= k:
                break
    return out


# ----------------------------------------------------------------- don hang (§8)

_KEY_DESC = ("mã đơn", "số hoá đơn", "số hóa đơn", "mã hoá đơn", "mã hóa đơn", "số thứ tự đơn", "mã phiếu nhập", "số phiếu")
_KEY_NAME = ("ma don", "so don", "ma hoa don", "so hoa don", "ma hd", "so hd", "so phieu", "ma phieu", "order", "don so", "madon",
             "ma giao dich", "ma bill", "so bill", "so ct", "ma chung tu")


def _find_key(df: pd.DataFrame, descs: dict[str, str | None]) -> str | None:
    for c in df.columns:
        d = (descs.get(c) or "").lower()
        if any(d.startswith(k) for k in _KEY_DESC):
            return c
    for c in df.columns:
        n = _split_name(c)
        if any(k in n for k in _KEY_NAME) and "dong" not in n and "tra" not in n.split():
            return c
    return None


def _apply_group_rules(df: pd.DataFrame, raw_blank: pd.DataFrame, key: str, info: TableInfo) -> pd.DataFrame:
    """Luat §8: o trong (o file goc) o dong sau cua cung don = o dong dau don; dong trung trong cung don tinh 1 lan."""
    if df[key].isna().all() or not df[key].duplicated().any():
        return df
    df = df.reset_index(drop=True)
    raw_blank = raw_blank.reset_index(drop=True)
    if df[key].isna().any():
        # dong co du lieu nhung trong ma don = dong sau cua don ngay tren (khong vuot qua ranh gioi file/sheet)
        part = df[[c for c in META_COLS if c in df.columns]].astype(str).agg("|".join, axis=1) if any(
            c in df.columns for c in META_COLS
        ) else pd.Series("", index=df.index)
        filled = df.groupby(part, sort=False)[key].ffill()
        if (filled.notna() & df[key].isna()).any():
            df[key] = filled
            info.ffilled.append(key)
    runs = (df[key] != df[key].shift()).cumsum()
    is_first = (runs != runs.shift()).to_numpy()
    first_pos = pd.Series(np.where(is_first, np.arange(len(df)), np.nan)).ffill().astype(int).to_numpy()
    size = df.groupby(runs, sort=False)[key].transform("size").to_numpy()
    multi = size > 1
    data_cols = [c for c in df.columns if c not in META_COLS and c != key and not c.endswith("__dv")]
    for c in data_cols:
        if c not in raw_blank.columns:
            continue
        rb = raw_blank[c].to_numpy()
        fill_mask = rb & ~is_first & multi
        if not fill_mask.any():
            continue
        # chi dien cot cap don: trong moi don nhieu dong, cac o khac trong co cung gia tri
        sub = pd.DataFrame({"g": runs.to_numpy()[multi], "v": df[c].to_numpy()[multi]}).dropna()
        if sub.empty or (sub.groupby("g")["v"].nunique() <= 1).mean() < 0.98:
            continue
        vals = df[c].to_numpy(copy=True)
        vals[fill_mask] = vals[first_pos[fill_mask]]
        df[c] = vals
        info.ffilled.append(c)
    cmp_cols = [c for c in df.columns if c not in META_COLS]
    dup = df.duplicated(subset=cmp_cols, keep="first")
    if dup.any():
        info.duplicates_dropped = int(dup.sum())
        df = df.loc[~dup].reset_index(drop=True)
    return df


# ----------------------------------------------------------------- ten bang


def _safe_name(s: str) -> str:
    s = re.sub(r"\.(xlsx|xlsm|xls|csv|tsv|txt|sqlite3?|db)$", "", s, flags=re.I)
    s = re.sub(r"<[^>]*>", "", s)
    s = strip_accents(s)
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
    return s or "bang"


def _base_name(name: str) -> str:
    """Bo nam / so thu tu file: 'So_chi_tiet_ban_hang_2026' -> 'So_chi_tiet_ban_hang'."""
    s = re.sub(r"(?<![A-Za-z])(20\d{2}|\d{1,2})(?![A-Za-z0-9])", "", name)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or name


# ----------------------------------------------------------------- nap cua hang


def _clean_frame(df: pd.DataFrame, descs: dict[str, str | None], year_hint: pd.Series | None, info: TableInfo) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        if c in META_COLS:
            out[c] = df[c]
            continue
        col, unit, cinfo = _infer_column(df[c], c, descs.get(c), year_hint)
        out[c] = col
        if unit is not None:
            out[cinfo.unit_column] = unit
        info.columns[c] = cinfo
    res = pd.DataFrame(out, index=df.index)
    for c in res.columns:
        if res[c].dtype == object:
            res[c] = res[c].where(res[c].notna(), None)
    # dong tong / ghi chu cuoi sheet trong so ghi tay ('Tong thang', 'Tong cong', 'Ghi chu: ...')
    footer = pd.Series(False, index=df.index)
    for c, ci in info.columns.items():
        if ci.kind in ("date", "datetime", "number", "time"):
            raw = df[c]
            failed = res[c].isna() & ~raw.map(is_blank)
            if failed.any():
                txt = raw[failed].map(lambda x: str(x).strip().lower())
                footer.loc[txt.index] |= txt.str.match(_FOOTER_RE)
    # dong tong xen giua du lieu ('Tong ngay 01/04' o cot ten, so tong o cot tien): it o duoc dien
    data_cols = [c for c in df.columns if c not in META_COLS]
    if data_cols:
        filled = (~df[data_cols].map(is_blank)).sum(axis=1)
        sparse = filled <= max(1, len(data_cols) // 2)
        key = _find_key(df, descs)
        cand = sparse.copy()
        if key is not None:
            kv = df[key]
            cand |= kv.map(is_blank)
            footer |= kv.map(lambda x: isinstance(x, str) and bool(re.match(_FOOTER_RE, x.strip().lower())))
        date_cols = [c for c, ci in info.columns.items() if ci.kind in ("date", "datetime")]
        if date_cols:
            cand |= df[date_cols].map(is_blank).all(axis=1)
        if cand.any():
            txt_hit = pd.Series(False, index=df.index)
            for c in data_cols:
                vals = df.loc[cand, c]
                m = vals.map(lambda x: isinstance(x, str) and bool(re.match(_FOOTER_RE, x.strip().lower())))
                txt_hit.loc[m.index] |= m
            footer |= cand & txt_hit
    if footer.any():
        info.dropped_rows += int(footer.sum())
        res = res.loc[~footer]
    return res


Part = tuple  # (raw_df, file, sheet, header_row, year_hint, section)

_FOOTER_RE = r"^(tổng|tong|cộng|cong|ghi chú|ghi chu|người lập|nguoi lap|ký tên|ky ten|lưu ý|luu y)\b"


def _match_section(rd: Readme, fname: str, sheet: str | None) -> Section | None:
    cands = [s for s in rd.sections if s.matches(fname, sheet)]
    if not cands:
        return None
    cands.sort(key=lambda s: (s.sheet is None, -len(s.pattern)))
    return cands[0]


def load_business(path: str) -> Business:
    bid = os.path.basename(os.path.normpath(path))
    readme_path = os.path.join(path, "README.md")
    text = open(readme_path, encoding="utf-8").read() if os.path.exists(readme_path) else ""
    rd = parse_readme(text)
    biz = Business(bid, path, rd, {}, {})
    parts: dict[str, list[Part]] = {}
    expected_all = rd.all_column_names()

    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        if fname == "README.md" or fname.startswith(".") or not os.path.isfile(fpath):
            continue
        low = fname.lower()
        try:
            if low.endswith((".xlsx", ".xlsm")):
                sheets = _read_xlsx(fpath)
            elif low.endswith((".csv", ".txt", ".tsv")):
                sheets = [(None, _read_csv(fpath))]
            elif low.endswith((".sqlite", ".db", ".sqlite3")):
                for tname, tdf in _read_sqlite(fpath).items():
                    tdf = tdf.astype(object).map(clean_str)
                    sec = _match_section(rd, fname, tname)
                    parts.setdefault(_safe_name(tname), []).append((tdf, fname, tname, -1, None, sec))
                continue
            else:
                continue
        except Exception as exc:  # file hong: ghi lai, khong dung ca he thong
            biz.warnings.append(f"khong doc duoc {fname}: {exc}")
            continue
        n_sheets = len(sheets)
        for sheet, rows in sheets:
            if not rows or all(all(is_blank(c) for c in r) for r in rows):
                continue
            sec = _match_section(rd, fname, sheet)
            expected = set(sec.columns) if sec else expected_all
            h = _find_header(rows, expected)
            if h is None:
                continue
            df, _ = _rows_to_frame(rows, h)
            if df.empty:
                continue
            yh = year_from_name(sheet or "") or year_from_name(fname)
            if sec is not None:
                tname = _safe_name(sec.pattern) + (f"__{_safe_name(sec.sheet)}" if sec.sheet else "")
            else:
                tname = _safe_name(fname) + (f"__{_safe_name(sheet)}" if sheet and n_sheets > 1 else "")
            parts.setdefault(tname, []).append((df, fname, sheet, h, yh, sec))

    # gop cac nhom khac ten nhung cung bo cot (danh sach dai tach nhieu file, file theo nam)
    merged: dict[str, list[Part]] = {}
    sig_owner: dict[frozenset, str] = {}
    for tname, plist in parts.items():
        sig = frozenset(plist[0][0].columns)
        owner = sig_owner.get(sig) if len(sig) >= 4 else None
        if owner is not None and plist[0][2] != "-sqlite-":
            merged[owner].extend(plist)
        else:
            merged[tname] = list(plist)
            sig_owner.setdefault(sig, tname)
    for tname in list(merged):
        files = {p[1] for p in merged[tname]}
        if len(files) > 1:
            base = _base_name(tname)
            if base != tname and base not in merged:
                merged[base] = merged.pop(tname)

    for tname, plist in merged.items():
        # cac phan co cot khac han nhau thi tach rieng
        groups: list[list[Part]] = []
        for p in plist:
            cols = set(p[0].columns)
            for g in groups:
                gcols = set(g[0][0].columns)
                if len(cols & gcols) >= 0.8 * max(len(cols), len(gcols)):
                    g.append(p)
                    break
            else:
                groups.append([p])
        groups.sort(key=lambda g: -sum(len(p[0]) for p in g))  # nhom lon nhat giu ten goc
        for gi, g in enumerate(groups):
            name = tname if gi == 0 else f"{tname}__{_safe_name(g[0][2] or g[0][1])}"
            files_g = {p[1] for p in g}
            if len(g) > 1 and len(files_g) == 1 and g[0][2] and name.endswith("__" + _safe_name(g[0][2])):
                stem = _safe_name(next(iter(files_g)))
                if stem not in merged and stem not in biz.tables:
                    name = stem  # nhieu sheet thang cua cung mot file: dat ten theo file
            frames = []
            for df, fname, sheet, h, yh, sec in g:
                df = df.copy()
                if len(g) > 1:
                    df.insert(0, "_file", fname)
                    df.insert(1, "_sheet", sheet)
                df["__yh"] = yh
                frames.append(df)
            raw = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0].reset_index(drop=True)
            yh_series = raw.pop("__yh")
            sec = next((x[5] for x in g if x[5] is not None), None)
            descs = dict(sec.columns) if sec else {c: rd.description(c) for c in raw.columns}
            info = TableInfo(
                name=name,
                sources=[(x[1], x[2]) for x in g],
                header_rows=[x[3] for x in g],
                n_rows=0,
                columns={},
            )
            raw_blank = raw.map(is_blank)
            clean = _clean_frame(raw, descs, yh_series, info)
            raw_blank = raw_blank.loc[clean.index]
            key = _find_key(clean, descs)
            if key is not None:
                info.key = key
                clean = _apply_group_rules(clean, raw_blank, key, info)
            info.n_rows = len(clean)
            biz.tables[name] = clean
            biz.info[name] = info
    _canonicalize_names(biz)
    _link_people(biz)
    return biz


_HONORIFICS = {"a", "anh", "chi", "c", "em", "e", "co", "chu", "bac", "ong", "ba", "thay", "tho", "nv", "ms", "mr"}


def _link_people(biz: Business) -> None:
    """Ten goi tat cua tho / nhan vien trong so ('a Duc', 'Duc') -> ho ten day du trong bang danh sach nguoi
    (ghep theo ten rieng = tu cuoi, chi khi dung mot nguoi khop). Them cot '<cot>__ho_ten'."""
    rosters = []
    for t, df in biz.tables.items():
        if len(df) > 100:
            continue
        for c in df.columns:
            d = (biz.info[t].columns.get(c).description or "").lower() if biz.info[t].columns.get(c) else ""
            n = _split_name(c)
            if d.startswith("họ tên") or n in ("ho ten", "ho va ten", "ten nhan vien", "ten tho", "ho ten tho"):
                names = [x for x in df[c].dropna().astype(str) if len(x.split()) >= 2]
                if names:
                    rosters.append((t, c, names))
    if not rosters:
        return
    by_last: dict[str, list[str]] = {}
    for _, _, names in rosters:
        for full in names:
            by_last.setdefault(strip_accents(full.split()[-1]).lower(), []).append(full)
    for t, df in biz.tables.items():
        if any(t == r[0] for r in rosters):
            continue
        for c in list(df.columns):
            d = (biz.info[t].columns.get(c).description or "").lower() if biz.info[t].columns.get(c) else ""
            n = _split_name(c)
            if not (any(k in d for k in ("thợ", "nhân viên")) or n in ("tho", "nhan vien", "nguoi ban", "nv ban", "ten nv", "tho sua")):
                continue
            if biz.info[t].columns.get(c) is None or biz.info[t].columns[c].kind != "text":
                continue
            mapping = {}
            for v in df[c].dropna().astype(str).unique():
                toks = [x for x in strip_accents(v).lower().replace(".", " ").split() if x not in _HONORIFICS]
                if not toks:
                    continue
                cands = by_last.get(toks[-1], [])
                full_hits = [f for f in cands if strip_accents(f).lower() == " ".join(toks)]
                if len(full_hits) == 1:
                    mapping[v] = full_hits[0]
                elif len(set(cands)) == 1:
                    mapping[v] = cands[0]
            if mapping and len(mapping) >= 0.5 * df[c].nunique():
                df[c + "__ho_ten"] = df[c].map(mapping)
                biz.info[t].columns[c + "__ho_ten"] = ColumnInfo(
                    name=c + "__ho_ten", kind="text",
                    description=f"họ tên đầy đủ của '{c}' (ghép tên gọi với danh sách người theo tên riêng; trống nếu không chắc)")


def _has_marks(s: str) -> bool:
    return strip_accents(s) != s


def _canonicalize_names(biz: Business) -> None:
    """Luat §8 'Ten goi, nhan': cac cach viet cung chi mot thu (khac hoa thuong, khoang trang, dau cau,
    hoac viet khong dau) duoc gop ve mot cach viet co trong file. Ap dung cho moi cot chuoi cua cua hang
    voi mot tu dien chung (de noi bang theo ten van dung)."""
    from collections import Counter, defaultdict

    from .parsing import compact_key as norm_key

    count: Counter = Counter()
    targets = []
    for tname, df in biz.tables.items():
        inf = biz.info[tname]
        for c in df.columns:
            ci = inf.columns.get(c)
            if c in META_COLS or ci is None or ci.kind != "text":
                continue
            vc = df[c].dropna().astype(str).value_counts()
            if vc.empty:
                continue
            targets.append((tname, c))
            for v, n in vc.items():
                count[v] += int(n)
    if not targets:
        return
    by_k1: dict[str, list[str]] = defaultdict(list)
    for sp in count:
        by_k1[norm_key(sp)].append(sp)

    def best(spellings):
        return max(spellings, key=lambda x: (_has_marks(x), not x.isupper(), count[x], x))

    canon_k1 = {k: best(v) for k, v in by_k1.items() if k}
    # gop nhom viet khong dau vao nhom co dau neu chi co dung mot nhom co dau tuong ung
    by_k2: dict[str, list[str]] = defaultdict(list)
    for k in canon_k1:
        by_k2[strip_accents(k)].append(k)
    k1_target = {k: k for k in canon_k1}
    for k2, ks in by_k2.items():
        marked = [k for k in ks if _has_marks(k)]
        plain = [k for k in ks if not _has_marks(k)]
        if len(marked) == 1 and plain:
            for k in plain:
                k1_target[k] = marked[0]
    mapping = {}
    for sp in count:
        k = norm_key(sp)
        if not k:
            continue
        tgt = canon_k1[k1_target[k]]
        if tgt != sp:
            mapping[sp] = tgt
    if not mapping:
        return
    for tname, c in targets:
        df = biz.tables[tname]
        col = df[c]
        hit = col.isin(mapping.keys())
        if hit.any():
            examples = list(dict.fromkeys((v, mapping[v]) for v in col[hit].head(2000)))[:3]
            df[c] = col.where(~hit, col.map(mapping))
            biz.info[tname].canonicalized[c] = examples
