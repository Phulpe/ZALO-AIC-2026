"""Ham tien ich nap san trong sandbox de code do model sinh ra goi (deu co dinh, co test)."""

from __future__ import annotations

import datetime as dt
import re
import unicodedata

import pandas as pd

from .answers import round2, round_half_up, round_money  # noqa: F401  (xuat ra cho sandbox)
from .dates import month_range, quarter_range, shift_month, to_date, week_range  # noqa: F401
from .parsing import norm_key, norm_key_ascii, parse_number, strip_accents


def between(series: pd.Series, start, end) -> pd.Series:
    """Mask cac dong co NGAY (bo gio) trong [start, end], tinh ca hai dau."""
    s = pd.to_datetime(series, errors="coerce")
    d = s.dt.normalize()
    return (d >= pd.Timestamp(to_date(start))) & (d <= pd.Timestamp(to_date(end)))


def norm(x) -> str:
    """Chuan hoa ten de so khop: thuong, bo dau cau va khoang trang thua (giu dau tieng Viet)."""
    return norm_key(x)


def same_name(series: pd.Series, name: str) -> pd.Series:
    """Mask cac dong cung chi mot ten voi `name` (khong phan biet hoa thuong, khoang trang, dau cau;
    neu khong dong nao khop thi so khop ca khi bo dau tieng Viet)."""
    target = norm_key(name)
    keys = series.map(norm_key)
    mask = keys == target
    if not mask.any():
        t2 = strip_accents(target)
        mask = keys.map(strip_accents) == t2
    return mask


def find_names(series: pd.Series, text: str, limit: int = 20) -> list[str]:
    """Liet ke cac gia tri khac nhau trong `series` chua cum `text` (khong dau, khong phan biet hoa thuong)."""
    t = norm_key_ascii(text)
    vals = pd.Series(series.dropna().astype(str).unique())
    hit = vals[vals.map(lambda v: t in norm_key_ascii(v))]
    return sorted(hit.tolist())[:limit]


_STATUS_RULES = (
    ("trả hàng", ("tra hang", "tra lai", "tra ve", "doi tra", "hoan hang", "hoan tra", "return", "refund", "mang tra", "hang tra")),
    ("huỷ", ("huy", "cancel", "khong giao duoc", "bo don", "khach bo", "void", "that bai", "failed")),
    ("đang giao", ("dang giao", "dang ship", "van chuyen", "shipping", "delivering", "in transit", "cho giao", "dang chuyen",
                   "dang di giao", "chua giao", "dang xu ly", "dang gui", "processing", "pending")),
    ("hoàn thành", ("hoan thanh", "xong", "complete", "done", "thanh cong", "da giao", "delivered", "success", "finished",
                    "da lay", "ok")),
)
# ghi chu cong no ('ghi so, khach tra het roi', 'no 1tr2, khach ck du roi') khong phai trang thai don
_PAYMENT_WORDS = ("no ", "ghi so", " ck", "chuyen khoan", "tien", "du roi", "tra du", "het roi", "thieu", "dua truoc", "tra truoc")


def status_of(x) -> str | None:
    """Doi chu trang thai don cua tung shop ve 1 trong 4 trang thai §8:
    'hoàn thành' | 'huỷ' | 'trả hàng' | 'đang giao'. Khong nhan ra -> None (tu xem xet)."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    s = norm_key_ascii(x)
    if not s:
        return None
    if s in ("khach tra", "da tra"):  # cot trang thai ghi ngan gon
        return "trả hàng"
    padded = f" {s} "
    is_payment_note = any(w in padded for w in _PAYMENT_WORDS) and not any(k in s for k in ("tra hang", "tra lai", "hoan hang", "doi tra"))
    for label, keys in _STATUS_RULES:
        if label == "trả hàng" and is_payment_note:
            continue
        if any(re.search(r"(?:^| )" + re.escape(k), s) for k in keys):  # khop tu dau tu ('chuyen' khong chua 'huy')
            return label
    return None


def pct(a, b) -> float | None:
    """Phan tram a/b * 100, lam tron 2 chu so (nua len). b = 0 -> None."""
    if b == 0 or b is None:
        return None
    return round2(100.0 * float(a) / float(b))


def pct_change(new, old) -> float | None:
    """Phan tram thay doi so voi ky truoc; giam ra so am."""
    if old == 0 or old is None:
        return None
    return round2(100.0 * (float(new) - float(old)) / float(old))


def to_num(x):
    """Doc so tu chuoi kieu '90k', '1tr2', '84,000', '2 cai' (tra ve float hoac None)."""
    return parse_number(x)[0]


def parse_conversions(text: str) -> pd.DataFrame:
    """Doc cac dong quy doi dang 'Ten hang: 1 thung = 24 lon' trong README -> bang
    (hang, don_vi_lon, he_so, don_vi_co_ban)."""
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*-\s*(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$", line)
        if m:
            k = parse_number(m.group(3))[0]
            rows.append((m.group(1).strip(), m.group(2).strip(), k, m.group(4).strip()))
    return pd.DataFrame(rows, columns=["hang", "don_vi_lon", "he_so", "don_vi_co_ban"])


def days_in(start, end) -> list[dt.date]:
    a, b = to_date(start), to_date(end)
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def count_weekday(start, end, weekday: int) -> int:
    """So ngay thu `weekday` (0 = thu Hai ... 6 = Chu nhat) trong [start, end]."""
    return sum(1 for d in days_in(start, end) if d.weekday() == weekday)


from .imports import IMPORT_DOC as _IMPORT_DOC
from .metrics import DOC as _METRICS_DOC

HELPER_DOC = """\
- between(series, start, end) -> mask các dòng có NGÀY (bỏ giờ) trong [start, end], tính cả hai đầu.
- month_range(y, m), quarter_range(y, q), week_range(d) -> (ngày đầu, ngày cuối) tính cả hai đầu; shift_month(y, m, k) -> (y, m) dời k tháng.
- tet_range(Y) -> (ngày đầu, ngày cuối) của Tết năm Y theo §2; days_in(a, b) -> list ngày; count_weekday(a, b, wd) -> số ngày thứ wd (0 = thứ Hai … 6 = Chủ nhật).
- norm(s) -> tên chuẩn hoá; same_name(series, name) -> mask các dòng cùng chỉ tên đó (không phân biệt hoa thường/khoảng trắng/dấu câu); find_names(series, "từ khoá") -> các tên có chứa từ khoá (để dò tên đúng trong dữ liệu).
- status_of(x) -> 'hoàn thành' | 'huỷ' | 'trả hàng' | 'đang giao' | None: đổi chữ trạng thái của shop về 4 trạng thái §8 (luôn kiểm tra lại bằng value_counts).
- round_money(x) -> int làm tròn về đồng (nửa đồng lên); round2(x) -> 2 chữ số thập phân (nửa lên); pct(a, b) = round2(100*a/b); pct_change(new, old) = round2(100*(new-old)/old).
- to_num(x) đọc số từ chuỗi kiểu '90k', '1tr2', '84,000'; parse_conversions(README) -> bảng quy đổi đơn vị ghi trong README dạng '- Tên: 1 thùng = 24 lon'.
- Khi hồ sơ có bảng chuẩn S/O, có thêm các hàm chỉ số (đã đúng rulebook, ưu tiên dùng):
{metrics_doc}- S, O: bảng chuẩn dòng hàng / đơn ĐÃ LỌC (chỉ tính doanh thu); S_gom_huy_tra, O_gom_huy_tra: đủ mọi trạng thái; N, R: dòng nhập hàng / phiếu nhập; debt_as_of, debt_by_customer, debt_added, debt_paid: công nợ; stock_at, stock_value_at, lot_stock_at: tồn kho (chỉ có khi hồ sơ cửa hàng ghi là đã dựng sẵn).
- README: chuỗi README của cửa hàng; TODAY: pd.Timestamp ngày hôm nay (as_of_date).
- answer(type, value) nộp đáp án; answer("unanswerable", reason="...") khi dữ liệu không đủ (§7).
"""

HELPER_DOC = HELPER_DOC.replace("{metrics_doc}", "".join("  " + line + "\n" for line in (_METRICS_DOC.strip() + "\n" + _IMPORT_DOC.strip()).splitlines()))
