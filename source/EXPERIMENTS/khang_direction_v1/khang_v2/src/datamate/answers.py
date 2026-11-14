"""Dinh dang dap an theo rulebook §9 va so sanh dap an (cham diem tai cho).

Moi type co mot format duy nhat; sai format la sai. Ham `normalize` dua gia tri model tinh ra ve
dung format; `same_answer` so dap an du doan voi dap an chuan (co alias).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

VALUE_TYPES = ("number", "percent", "date", "boolean", "string", "list", "table")
ALL_TYPES = VALUE_TYPES + ("unanswerable", "abstain")


class AnswerError(ValueError):
    pass


def round_half_up(x: float, nd: int = 0) -> float:
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def round_money(x) -> int:
    return int(round_half_up(float(x), 0))


def round2(x) -> float:
    return round_half_up(float(x), 2)


def _num(v: Any) -> float | int:
    if isinstance(v, bool):
        raise AnswerError("boolean khong phai number")
    if hasattr(v, "item") and not isinstance(v, (list, dict, str)):  # numpy scalar
        v = v.item()
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise AnswerError("gia tri NaN/inf")
        return v
    if isinstance(v, Decimal):
        return float(v)
    raise AnswerError(f"can JSON number, nhan {type(v).__name__}: {v!r}")


def _clean_number(v: Any) -> int | float:
    x = _num(v)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return round_half_up(x, 2) if abs(x * 100 - round(x * 100)) > 1e-9 else round(x, 2)
    return int(x)


def _scalar_for_list(v: Any) -> Any:
    if hasattr(v, "item") and not isinstance(v, (list, dict, str)):
        v = v.item()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _clean_number(v)
    if isinstance(v, (dt.datetime, dt.date)) or hasattr(v, "to_pydatetime"):
        return _date_str(v)
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, bool) or v is None:
        return v
    return str(v)


_MONTH_TXT = re.compile(r"^(?:th[aá]ng|thang|thg|th|t)?\s*(\d{1,2})\s*(?:/|-|\s+n[aă]m\s+|\s+)\s*(\d{4})$", re.I)
_DMY_TXT = re.compile(r"^(?:ng[aà]y\s*)?(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})$", re.I)
_YMD_TXT = re.compile(r"^(\d{4})\s*[/\-.]\s*(\d{1,2})(?:\s*[/\-.]\s*(\d{1,2}))?$")


def _date_text(s: str) -> str | None:
    """Chuoi ngay/thang theo cach viet thuong gap -> 'YYYY-MM-DD' / 'YYYY-MM'; khong phai ngay -> None."""
    s = unicodedata.normalize("NFC", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if m := _YMD_TXT.match(s):
        y, mo, d = int(m[1]), int(m[2]), m[3]
        if 1 <= mo <= 12:
            if d is None:
                return f"{y:04d}-{mo:02d}"
            try:
                return dt.date(y, mo, int(d)).isoformat()
            except ValueError:
                return None
    if m := _DMY_TXT.match(s):
        try:
            return dt.date(int(m[3]), int(m[2]), int(m[1])).isoformat()
        except ValueError:
            return None
    if m := _MONTH_TXT.match(s):
        mo, y = int(m[1]), int(m[2])
        if 1 <= mo <= 12 and 1900 < y < 2200:
            return f"{y:04d}-{mo:02d}"
    return None


def _is_date_like(v: Any) -> bool:
    if isinstance(v, (dt.datetime, dt.date)) or hasattr(v, "to_pydatetime"):
        return True
    return isinstance(v, str) and _date_text(v) is not None


def _date_str(v: Any, month: bool = False) -> str:
    if type(v).__name__ in ("Series", "DataFrame", "Index", "DatetimeIndex", "ndarray"):
        vals = list(v.values.ravel()) if hasattr(v, "values") else list(v.ravel())
        if len(vals) != 1:
            raise AnswerError(
                f"cần MỘT ngày nhưng nhận {type(v).__name__} {len(vals)} giá trị (vd df.idxmax() trên cả bảng). Lấy một ngày: "
                "df.loc[df['doanh_thu'].idxmax(), 'order_date'] hoặc df.sort_values(['doanh_thu', 'order_date'], ascending=[False, True]).order_date.iloc[0]")
        v = vals[0]
    if type(v).__name__ == "datetime64":
        import pandas as pd

        v = pd.Timestamp(v)
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()
    if type(v).__name__ == "Period":  # pd.Period thang -> 'YYYY-MM'
        v = str(v)
    if isinstance(v, (tuple, list)) and len(v) in (2, 3) and all(isinstance(x, int) for x in v):
        if len(v) == 2 and 1900 < v[0] < 2200 and 1 <= v[1] <= 12:  # (nam, thang)
            return f"{v[0]:04d}-{v[1]:02d}"
        if len(v) == 3:
            return dt.date(*v).isoformat()
    if isinstance(v, (tuple, list)) and len(v) == 2:  # ky (dau, cuoi): ca thang -> 'YYYY-MM'; mot ngay -> ngay
        a, b = (x.to_pydatetime() if hasattr(x, "to_pydatetime") else x for x in v)
        if isinstance(a, (dt.date, dt.datetime)) and isinstance(b, (dt.date, dt.datetime)):
            a = a.date() if isinstance(a, dt.datetime) else a
            b = b.date() if isinstance(b, dt.datetime) else b
            if a == b:
                return a.isoformat()
            nxt = (b + dt.timedelta(days=1))
            if a.day == 1 and nxt.day == 1 and (a.year, a.month) == (b.year, b.month):
                return f"{a.year:04d}-{a.month:02d}"
        raise AnswerError(f"ngay phai la mot ngay hoac mot thang, nhan {v!r}")
    if isinstance(v, (dt.datetime, dt.date)):
        return v.strftime("%Y-%m") if month else v.strftime("%Y-%m-%d")
    r = _date_text(str(v))
    if r is not None:
        return r
    raise AnswerError(f"ngay phai dang YYYY-MM-DD hoac YYYY-MM, nhan {v!r}")


def normalize(type_: str, value: Any = None, reason: str | None = None) -> dict:
    """Tra ve dict {"type", "answer"} dung format §9, hoac raise AnswerError."""
    t = (type_ or "").strip().lower()
    aliases = {"bool": "boolean", "str": "string", "int": "number", "float": "number", "pct": "percent", "percentage": "percent"}
    t = aliases.get(t, t)
    if t not in ALL_TYPES:
        raise AnswerError(f"type khong hop le: {type_!r}; dung mot trong {ALL_TYPES}")
    if t == "abstain":
        return {"type": "abstain"}
    if t == "unanswerable":
        out = {"type": "unanswerable"}
        if reason is not None:
            out["reason"] = str(reason)
        return out
    if t == "number":
        return {"type": t, "answer": _clean_number(value)}
    if t == "percent":
        x = float(_num(value))
        r = round_half_up(x, 2)
        return {"type": t, "answer": int(r) if r.is_integer() else r}
    if t == "date":
        return {"type": t, "answer": _date_str(value)}
    if t == "boolean":
        if hasattr(value, "item"):
            value = value.item()
        if not isinstance(value, bool):
            raise AnswerError(f"boolean phai la True/False, nhan {value!r}")
        return {"type": t, "answer": value}
    if t == "string":
        if hasattr(value, "item") and not isinstance(value, str) and not hasattr(value, "to_pydatetime"):
            value = value.item()
        if _is_date_like(value):  # "tháng 8/2025", "15/08/2026", Timestamp -> type date (§9)
            return {"type": "date", "answer": _date_str(value)}
        if isinstance(value, (list, tuple)):
            raise AnswerError("string khong duoc la list; dung type 'list' neu cau hoi hoi nhieu doi tuong")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise AnswerError("string rong")
        return {"type": t, "answer": unicodedata.normalize("NFC", str(value)).strip()}
    if t == "list":
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise AnswerError("list phai la mang; mot phan tu van la mang")
        for v in value:
            if isinstance(v, (list, tuple, dict)) or (hasattr(v, "__len__") and hasattr(v, "dtype") and not isinstance(v, str)):
                raise AnswerError("list phai phang: moi phan tu la mot gia tri (ten hoac so); ten kem so do thi dung type 'table', "
                                  "chi hoi ten thi chi lay cot ten")
            if v is None or (isinstance(v, float) and math.isnan(v)) or (hasattr(v, "item") and not isinstance(v, str) and v != v):
                raise AnswerError("list co phan tu rong (None/NaN): kiem tra lai phep noi bang / cot ten truoc khi nop")
        return {"type": t, "answer": [_scalar_for_list(v) for v in value]}
    if t == "table":
        if hasattr(value, "to_dict") and hasattr(value, "columns"):  # DataFrame
            value = {"columns": [str(c) for c in value.columns], "rows": value.values.tolist()}
        if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):  # df.to_dict("records")
            value = {"columns": list(value[0].keys()), "rows": value}
        if not isinstance(value, dict) or "columns" not in value or "rows" not in value:
            raise AnswerError('table phai la {"columns": [...], "rows": [[...], ...]}')
        cols = [str(c) for c in value["columns"]]
        raw_rows = value["rows"]
        if hasattr(raw_rows, "to_dict") and hasattr(raw_rows, "columns"):  # rows la DataFrame
            raw_rows = raw_rows.values.tolist()
        elif hasattr(raw_rows, "tolist"):
            raw_rows = raw_rows.tolist()
        fixed = []
        for r in raw_rows:
            if isinstance(r, dict):  # dong dang dict (to_dict("records")): lay gia tri theo ten cot, khong lay ten cot
                missing = [c for c in value["columns"] if c not in r and str(c) not in r]
                if missing:
                    raise AnswerError(f"dong {r} thieu cot {missing}")
                r = [r[c] if c in r else r[str(c)] for c in value["columns"]]
            elif hasattr(r, "tolist"):
                r = r.tolist()
            elif isinstance(r, tuple):
                r = list(r)
            fixed.append(r)
        rows = [[_scalar_for_list(c) for c in r] for r in fixed]
        if any(len(r) != len(cols) for r in rows):
            raise AnswerError("so o moi dong phai bang so cot")
        if len(cols) == 2 and rows and all(isinstance(r[0], str) and _is_date_like(r[0]) for r in rows) \
                and all(isinstance(r[1], (int, float)) and not isinstance(r[1], bool) for r in rows):
            # so do theo tung thang/ngay: §9 "cau hoi nhieu so thi ghi list" (bang chi dung cho doi tuong co ten)
            return {"type": "list", "answer": [r[1] for r in rows]}
        return {"type": t, "answer": {"columns": cols, "rows": rows}}
    raise AnswerError(t)


def answer_key(ans: dict) -> str:
    """Khoa de bieu quyet giua cac lan chay (bo reason; chuoi so khop khong phan biet hoa thuong)."""
    t = ans.get("type")
    if t in ("unanswerable", "abstain"):
        return t
    v = ans.get("answer")
    if t == "string":
        v = _norm_str(v)
    elif t == "list":
        v = [_norm_str(x) if isinstance(x, str) else x for x in v]
    elif t == "table":
        v = {"columns": [_norm_str(c) for c in v["columns"]], "rows": sorted(json.dumps([_norm_str(c) if isinstance(c, str) else c for c in r], ensure_ascii=False) for r in v["rows"])}
    return t + "|" + json.dumps(v, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------ cham diem (§9)


def _norm_str(s: Any) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[\W_]+", "", s)  # bo khoang trang, dau cau
    return s


def _str_match(pred: Any, gold: Any, alias: list | None) -> bool:
    if not isinstance(pred, str):
        return False
    p = _norm_str(pred)
    cands = [gold] + list(alias or [])
    return any(isinstance(c, str) and p == _norm_str(c) for c in cands)


def _num_match(p: Any, g: Any, percent: bool = False) -> bool:
    if isinstance(p, bool) or isinstance(g, bool):
        return False
    if not isinstance(p, (int, float)) or not isinstance(g, (int, float)):
        return False
    if percent:
        return abs(round(p, 2) - round(g, 2)) < 1e-6
    return abs(p - g) <= 1e-6 * max(1.0, abs(g))


def _cell_match(p: Any, g: Any, alias: list | None = None) -> bool:
    if isinstance(g, str):
        return _str_match(p, g, alias)
    if isinstance(g, bool):
        return p is g
    if isinstance(g, (int, float)):
        return _num_match(p, g)
    return p == g


def same_answer(pred: dict, gold: dict) -> bool:
    gt, pt = gold.get("type"), pred.get("type")
    if pt != gt:
        return False
    if gt == "unanswerable":
        return True
    g, p = gold.get("answer"), pred.get("answer")
    alias = gold.get("alias") or []
    if gt == "number":
        return _num_match(p, g)
    if gt == "percent":
        return _num_match(p, g, percent=True)
    if gt in ("date", "boolean"):
        return p == g
    if gt == "string":
        return _str_match(p, g, alias)
    if gt == "list":
        if not isinstance(p, list) or len(p) != len(g):
            return False
        for i, (pi, gi) in enumerate(zip(p, g)):
            al = alias[i] if i < len(alias) and isinstance(alias[i], list) else None
            if not _cell_match(pi, gi, al):
                return False
        return True
    if gt == "table":
        if not isinstance(p, dict) or len(p.get("columns", [])) != len(g["columns"]):
            return False
        gcols = [_norm_str(c) for c in g["columns"]]
        pcols = [_norm_str(c) for c in p["columns"]]
        if sorted(gcols) == sorted(pcols):
            perm = [pcols.index(c) for c in gcols]
        else:
            perm = list(range(len(gcols)))
        prows = [[r[j] for j in perm] for r in p.get("rows", [])]
        if len(prows) != len(g["rows"]):
            return False
        used = [False] * len(prows)
        for gi, grow in enumerate(g["rows"]):
            al = alias[gi] if gi < len(alias) and isinstance(alias[gi], list) else None
            found = False
            for pi, prow in enumerate(prows):
                if used[pi]:
                    continue
                if _cell_match(prow[0], grow[0], al) and all(_cell_match(a, b) for a, b in zip(prow[1:], grow[1:])):
                    used[pi] = found = True
                    break
            if not found:
                return False
        return True
    return False


def score(pred: dict | None, gold: dict) -> int:
    """+1 dung, -1 sai, 0 bo trong (abstain / khong co dong)."""
    if pred is None or pred.get("type") in (None, "abstain"):
        return 0
    return 1 if same_answer(pred, gold) else -1
