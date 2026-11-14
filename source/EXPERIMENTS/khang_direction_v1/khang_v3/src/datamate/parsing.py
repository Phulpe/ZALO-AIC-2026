"""Chuan hoa gia tri o trong file cua cua hang: tien, so luong, ngay gio, chuoi.

Du lieu cuoc thi co tinh "ban": tien ghi '90k', '12tr460', '105.000d', '84,000';
ngay ghi '1 thg 3 2025', '1/1' (nam lay theo sheet), so serial Excel; so luong kem chu
don vi ('2 cai'). Moi ham o day thuan tuy, khong phu thuoc model, co test rieng.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata
from typing import Any

EXCEL_EPOCH = dt.datetime(1899, 12, 30)

_SPACES = re.compile(r"\s+")


def is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    try:  # pandas NA / NaT
        import pandas as pd

        if x is pd.NaT or x is pd.NA:
            return True
    except Exception:  # pragma: no cover
        pass
    return False


def clean_str(x: Any) -> Any:
    """Bo khoang trang thua o dau/cuoi va giua chuoi; giu nguyen gia tri khong phai chuoi."""
    if isinstance(x, str):
        s = unicodedata.normalize("NFC", x).replace(" ", " ")
        s = _SPACES.sub(" ", s).strip()
        return s if s else None
    return x


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def norm_key(x: Any) -> str:
    """Khoa so khop ten: thuong, bo dau cau va khoang trang thua (giu dau tieng Viet)."""
    if is_blank(x):
        return ""
    s = unicodedata.normalize("NFC", str(x)).lower()
    s = re.sub(r"[^\w\s%+./-]", " ", s)
    return _SPACES.sub(" ", s).strip()


def compact_key(x: Any) -> str:
    """Khoa so khop theo §9: bo moi khoang trang va dau cau, khong phan biet hoa thuong (giu dau tieng Viet)."""
    if is_blank(x):
        return ""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", str(x)).lower())


def norm_key_ascii(x: Any) -> str:
    """Nhu norm_key nhung bo ca dau tieng Viet (dung khi so khop cach viet khong dau)."""
    return strip_accents(norm_key(x))


# --------------------------------------------------------------------------- so

_NUM_TOKEN = r"[-+]?\d[\d.,]*"
_RE_TR = re.compile(r"^([-+]?\d+(?:[.,]\d+)?)\s*(?:tr|triệu|trieu)\s*(\d{1,3})?$")
_RE_K = re.compile(r"^([-+]?\d+(?:[.,]\d+)?)\s*(?:k|nghìn|ngàn|nghin|ngan)$")
_RE_TY = re.compile(r"^([-+]?\d+(?:[.,]\d+)?)\s*(?:tỷ|tỉ|ty|ti)\s*(\d{1,3})?$")
_RE_THOUSANDS = re.compile(r"^[-+]?\d{1,3}(?:([.,])\d{3})(?:\1\d{3})*$")
_RE_LEAD_NUM = re.compile(r"^\s*(" + _NUM_TOKEN + r")\s*(.*)$")
_CURRENCY_WORDS = ("vnđ", "vnd", "đồng", "dong", "đ", "₫")


def _plain_number(s: str) -> float | None:
    """'84,000' -> 84000; '105.000' -> 105000; '1.234.567' -> 1234567; '2,5' -> 2.5; '18.6' -> 18.6."""
    s = s.strip().replace(" ", "")
    if not s:
        return None
    if _RE_THOUSANDS.match(s):
        return float(re.sub(r"[.,]", "", s))
    if s.count(",") and s.count("."):
        # 1.234,5 (kieu VN) hoac 1,234.5 (kieu Anh)
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(",") == 1:
        s = s.replace(",", ".")
    elif s.count(",") > 1 or s.count(".") > 1:
        s = s.replace(",", "").replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _scaled(num: str, scale: float, rest: str | None) -> float | None:
    # '3.045tr', '7,59tr', '1,5k': dau phan cach la dau thap phan
    n = num.replace(" ", "")
    if n.count(",") + n.count(".") == 1:
        try:
            base = float(n.replace(",", "."))
        except ValueError:
            return None
    else:
        base = _plain_number(n)
    if base is None:
        return None
    value = base * scale
    if rest:
        extra = int(rest.ljust(3, "0")) * (scale / 1000)
        value += -extra if num.strip().startswith("-") else extra
    return value


def parse_number(x: Any) -> tuple[float | None, str | None]:
    """Tra ve (gia tri, phan chu con lai). Khong doc duoc -> (None, None).

    Vi du: '90k' -> (90000, None); '12tr460' -> (12460000, None); '2 cái' -> (2, 'cái');
    '105.000đ' -> (105000, None); '(2)' -> (-2, None).
    """
    if is_blank(x):
        return None, None
    if isinstance(x, bool):
        return float(x), None
    if isinstance(x, (int, float)):
        return (None, None) if isinstance(x, float) and math.isnan(x) else (float(x), None)
    if isinstance(x, (dt.date, dt.datetime)):
        return None, None
    s = unicodedata.normalize("NFC", str(x)).strip().lower().replace(" ", " ")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    for w in _CURRENCY_WORDS:
        if s.endswith(w):
            s = s[: -len(w)].strip()
            break
    s = s.rstrip(".").strip()
    if not s:
        return None, None
    for rx, scale in ((_RE_TY, 1e9), (_RE_TR, 1e6), (_RE_K, 1e3)):
        m = rx.match(s.replace(" ", "") if rx is not _RE_K else s)
        if m:
            v = _scaled(m.group(1), scale, m.group(2) if m.lastindex and m.lastindex >= 2 else None)
            if v is not None:
                return (-v if neg else v), None
    v = _plain_number(s)
    if v is not None:
        return (-v if neg else v), None
    m = _RE_LEAD_NUM.match(s)
    if m:
        v = _plain_number(m.group(1))
        rest = m.group(2).strip()
        if v is not None and rest and re.match(r"^[^\W\d_]", rest):
            for rx, scale in ((_RE_TR, 1e6), (_RE_K, 1e3)):
                mm = rx.match((m.group(1) + rest).replace(" ", ""))
                if mm:
                    vv = _scaled(mm.group(1), scale, mm.group(2) if mm.lastindex and mm.lastindex >= 2 else None)
                    if vv is not None:
                        return (-vv if neg else vv), None
            return (-v if neg else v), rest
    return None, None


def to_number(x: Any) -> float | None:
    return parse_number(x)[0]


# --------------------------------------------------------------------------- ngay

_RE_YMD = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ t](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?)?$")
_RE_DMY = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?:[ ,]+(\d{1,2})[:h](\d{2})(?::(\d{2}))?)?$")
_RE_DM = re.compile(r"^(\d{1,2})[-/.](\d{1,2})$")
_RE_THG = re.compile(r"^(\d{1,2})\s*(?:thg|tháng|th)\s*(\d{1,2})(?:[, ]+(?:năm\s*)?(\d{4}))?(?:[ ,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
_RE_NGAY = re.compile(r"^ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})(?:\s*năm\s*(\d{4}))?$")
_RE_TIME = re.compile(r"^(\d{1,2})[:h](\d{2})(?::(\d{2}))?$")


def _mk(y: int, m: int, d: int, hh=None, mm=None, ss=None) -> dt.datetime | None:
    if y < 100:
        y += 2000
    try:
        return dt.datetime(y, m, d, int(hh or 0), int(mm or 0), int(ss or 0))
    except ValueError:
        return None


def parse_datetime(x: Any, year_hint: int | None = None) -> dt.datetime | None:
    """Doc ngay/gio. Ngay khong ghi nam lay year_hint (nam cua sheet, rulebook §8)."""
    if is_blank(x):
        return None
    if isinstance(x, dt.datetime):
        return x.replace(tzinfo=None)
    if isinstance(x, dt.date):
        return dt.datetime(x.year, x.month, x.day)
    if hasattr(x, "to_pydatetime"):
        return x.to_pydatetime().replace(tzinfo=None)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if 20000 <= float(x) <= 80000:  # serial Excel
            return EXCEL_EPOCH + dt.timedelta(days=float(x))
        return None
    s = unicodedata.normalize("NFC", str(x)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return None
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        return parse_datetime(float(s), year_hint)
    m = _RE_YMD.match(s)
    if m:
        return _mk(int(m[1]), int(m[2]), int(m[3]), m[4], m[5], m[6])
    m = _RE_DMY.match(s)
    if m:
        return _mk(int(m[3]), int(m[2]), int(m[1]), m[4], m[5], m[6])
    m = _RE_THG.match(s)
    if m:
        y = int(m[3]) if m[3] else year_hint
        return _mk(y, int(m[2]), int(m[1]), m[4], m[5], m[6]) if y else None
    m = _RE_NGAY.match(s)
    if m:
        y = int(m[3]) if m[3] else year_hint
        return _mk(y, int(m[2]), int(m[1])) if y else None
    m = _RE_DM.match(s)
    if m and year_hint:
        return _mk(year_hint, int(m[2]), int(m[1]))
    return None


def parse_time(x: Any) -> dt.time | None:
    if is_blank(x):
        return None
    if isinstance(x, dt.datetime):
        return x.time()
    if isinstance(x, dt.time):
        return x
    if isinstance(x, (int, float)) and 0 <= float(x) < 1:  # phan le ngay cua Excel
        secs = round(float(x) * 86400)
        return dt.time(secs // 3600 % 24, secs // 60 % 60, secs % 60)
    m = _RE_TIME.match(str(x).strip().lower())
    if m:
        try:
            return dt.time(int(m[1]), int(m[2]), int(m[3] or 0))
        except ValueError:
            return None
    d = parse_datetime(x)
    return d.time() if d else None


def year_from_name(name: str) -> int | None:
    m = re.findall(r"(?<!\d)(20\d{2})(?!\d)", name or "")
    return int(m[-1]) if m else None


def parse_bool(x: Any) -> bool | None:
    if is_blank(x):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x) if x in (0, 1) else None
    s = norm_key_ascii(x)
    if s in {"true", "1", "co", "x", "yes", "y", "dung", "v"}:
        return True
    if s in {"false", "0", "khong", "no", "n", "sai"}:
        return False
    return None
