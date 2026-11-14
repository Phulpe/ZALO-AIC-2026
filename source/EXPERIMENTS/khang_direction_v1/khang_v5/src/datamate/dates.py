"""Moc thoi gian theo rulebook §2, tinh co dinh tu as_of_date (khong de model tu tinh)."""

from __future__ import annotations

import calendar as _cal
import csv
import datetime as dt
from dataclasses import dataclass

WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]


def to_date(x) -> dt.date:
    if isinstance(x, (tuple, list)) and len(x) == 3 and all(isinstance(v, (int, float)) for v in x):
        return dt.date(int(x[0]), int(x[1]), int(x[2]))
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)) and hasattr(x, "dtype") and getattr(x, "ndim", 0) >= 1:
        # model hay truyen nham cot / mang dieu kien (vd between(P.date, a, b)) vao cho ky
        raise ValueError("kỳ phải là biến kỳ (THANG(m, y), QUY(q, y), NGAY(d, m, y), TUAN_TRUOC…) hoặc ngày 'YYYY-MM-DD', "
                         "không phải một cột hay mảng điều kiện; vd n_tickets(THANG(8, 2025), viec='sửa điện')")
    if hasattr(x, "to_pydatetime"):
        return x.to_pydatetime().date()
    try:
        return dt.date.fromisoformat(str(x)[:10])
    except ValueError:
        raise ValueError(f"không đọc được ngày/kỳ {str(x)[:40]!r}: dùng biến kỳ (THANG(m, y), NGAY(d, m, y)…) hoặc 'YYYY-MM-DD'") from None


def month_range(year: int, month: int) -> tuple[dt.date, dt.date]:
    return dt.date(year, month, 1), dt.date(year, month, _cal.monthrange(year, month)[1])


def quarter_range(year: int, q: int) -> tuple[dt.date, dt.date]:
    m0 = 3 * (q - 1) + 1
    return dt.date(year, m0, 1), month_range(year, m0 + 2)[1]


def week_range(d) -> tuple[dt.date, dt.date]:
    """Tuan (thu Hai -> Chu nhat) chua ngay d."""
    d = to_date(d)
    start = d - dt.timedelta(days=d.weekday())
    return start, start + dt.timedelta(days=6)


def shift_month(year: int, month: int, k: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + k
    return idx // 12, idx % 12 + 1


def month_year_for(month: int, today) -> int:
    """Thang M khong ghi nam: nam nay neu khong sau hom nay, sau hom nay thi nam truoc (§2)."""
    today = to_date(today)
    return today.year if month <= today.month else today.year - 1


def quarter_year_for(q: int, today) -> int:
    today = to_date(today)
    return today.year if 3 * (q - 1) + 1 <= today.month else today.year - 1


@dataclass
class LunarCalendar:
    rows: dict[dt.date, tuple[int, int, int, bool, str]]

    @classmethod
    def load(cls, path: str) -> "LunarCalendar":
        rows = {}
        with open(path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                d = dt.date.fromisoformat(r["date"])
                rows[d] = (
                    int(r["lunar_day"]),
                    int(r["lunar_month"]),
                    int(r["lunar_year"]),
                    str(r.get("lunar_leap", "")).strip().lower() == "true",
                    (r.get("holiday") or "").strip(),
                )
        return cls(rows)

    def solar(self, lunar_year: int, lunar_month: int, lunar_day: int, leap: bool = False) -> dt.date | None:
        for d, (ld, lm, ly, lp, _) in self.rows.items():
            if ly == lunar_year and lm == lunar_month and ld == lunar_day and lp == leap:
                return d
        return None

    def tet_range(self, year: int) -> tuple[dt.date, dt.date] | None:
        """Tet nam Y: 23 thang Chap (nam am Y-1) -> het mung 10 thang Gieng (nam am Y)."""
        start = self.solar(year - 1, 12, 23)
        end = self.solar(year, 1, 10)
        if start and end:
            return start, end
        return None

    def holidays(self) -> list[tuple[dt.date, str]]:
        return [(d, v[4]) for d, v in sorted(self.rows.items()) if v[4]]


def _fmt(a: dt.date, b: dt.date) -> str:
    return f"{a.isoformat()} → {b.isoformat()}"


def date_context(as_of, lunar: LunarCalendar | None = None) -> str:
    """Bang moc thoi gian dua vao prompt (model chi viec dung, khong tu tinh)."""
    D = to_date(as_of)
    lines = [f"- Hôm nay (as_of_date): {D.isoformat()} ({WEEKDAYS_VI[D.weekday()]}) [HOM_NAY]"]
    y = D - dt.timedelta(days=1)
    lines.append(f"- Hôm qua: {y.isoformat()} ({WEEKDAYS_VI[y.weekday()]}) [HOM_QUA]")
    ws, _ = week_range(D)
    lines.append(f"- Tuần này: {_fmt(ws, D)} [TUAN_NAY]")
    pws = ws - dt.timedelta(days=7)
    lines.append(f"- Tuần trước / tuần rồi: {_fmt(pws, pws + dt.timedelta(days=6))} [TUAN_TRUOC]")
    ms = D.replace(day=1)
    lines.append(f"- Tháng này: {_fmt(ms, D)} [THANG_NAY]")
    py, pm = shift_month(D.year, D.month, -1)
    p_start, p_end = month_range(py, pm)
    same_end = min(p_end, dt.date(py, pm, min(D.day, p_end.day)))
    lines.append(f"- Tháng trước (cả tháng): {_fmt(p_start, p_end)} [THANG_TRUOC]")
    lines.append(f"- Tháng này so với tháng trước / cùng kỳ tháng trước: cùng số ngày của tháng trước = {_fmt(p_start, same_end)} [THANG_TRUOC_CUNG_KY]")
    q = (D.month - 1) // 3 + 1
    qs, qe = quarter_range(D.year, q)
    lines.append(f"- Quý này: Q{q}/{D.year} = {_fmt(qs, D)} [QUY_NAY] (cả quý: {_fmt(qs, qe)})")
    pqy, pq = (D.year, q - 1) if q > 1 else (D.year - 1, 4)
    lines.append(f"- Quý trước: Q{pq}/{pqy} = {_fmt(*quarter_range(pqy, pq))} [QUY_TRUOC]")
    lines.append(f"- Từ đầu năm tới nay: {_fmt(dt.date(D.year, 1, 1), D)} [TU_DAU_NAM]")
    lines.append(f"- Năm nay: {D.year}; năm ngoái / năm trước: {D.year - 1}")
    for n in (7, 14, 30):
        lines.append(f"- {n} ngày gần nhất: {_fmt(D - dt.timedelta(days=n - 1), D)} [N_NGAY_GAN_NHAT({n})]")
    lines.append(
        f"- Tháng M không ghi năm: tháng 1..{D.month} là năm {D.year}; tháng {D.month + 1}..12 là năm {D.year - 1}"
        if D.month < 12
        else f"- Tháng M không ghi năm: năm {D.year}"
    )
    if lunar is not None:
        for yr in (D.year - 2, D.year - 1, D.year, D.year + 1):
            r = lunar.tet_range(yr)
            if r:
                n = (r[1] - r[0]).days + 1
                before = (r[0] - dt.timedelta(days=n), r[0] - dt.timedelta(days=1))
                lines.append(f"- Tết {yr}: {_fmt(*r)} ({n} ngày) [TET({yr})]; trước Tết {yr} (cùng số ngày): {_fmt(*before)} [TRUOC_TET({yr})]")
    return "\n".join(lines)


def date_args(args: tuple, n: int, fname: str = "hàm") -> list[dt.date]:
    """Doc n ngay tu tham so linh hoat: ('2025-08-24',), (2025, 8, 24), (date,), (Timestamp,), ((2025, 8, 24),)."""
    out: list[dt.date] = []
    i = 0
    args = list(args)
    if n == 2 and len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 2:  # debt_paid(THANG_NAY)
        args = list(args[0])
    if (
        n == 1 and len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 2
        and not all(isinstance(x, int) for x in args[0])
    ):  # debt_as_of(NGAY(28, 6, 2025)), stock_at(THANG(6, 2025)): tinh den het ngay cuoi ky
        args = [args[0][1]]
    while len(out) < n:
        if i >= len(args):
            raise TypeError(f"{fname} cần {n} ngày, vd {fname}('2025-08-24'" + (", '2025-08-31')" if n == 2 else ")"))
        a = args[i]
        if isinstance(a, int) and i + 2 < len(args) + 0 and all(isinstance(v, int) for v in args[i:i + 3]) and 1900 < a < 2200:
            out.append(dt.date(args[i], args[i + 1], args[i + 2]))
            i += 3
        else:
            out.append(to_date(a))
            i += 1
    if i != len(args):
        raise TypeError(f"{fname}: thừa tham số {args[i:]}")
    return out


def period_vars(as_of, lunar: "LunarCalendar | None" = None) -> dict:
    """Bien ky chuan (§2) nap san trong sandbox: moi ky la tuple (ngay_dau, ngay_cuoi) tinh ca hai dau."""
    D = to_date(as_of)
    ws, _ = week_range(D)
    pws = ws - dt.timedelta(days=7)
    py, pm = shift_month(D.year, D.month, -1)
    p_start, p_end = month_range(py, pm)
    same_end = min(p_end, dt.date(py, pm, min(D.day, p_end.day)))
    q = (D.month - 1) // 3 + 1
    pqy, pq = (D.year, q - 1) if q > 1 else (D.year - 1, 4)
    v = {
        "HOM_NAY": D,
        "HOM_QUA": D - dt.timedelta(days=1),
        "TUAN_NAY": (ws, D),
        "TUAN_TRUOC": (pws, pws + dt.timedelta(days=6)),
        "THANG_NAY": (D.replace(day=1), D),
        "THANG_TRUOC": (p_start, p_end),
        "THANG_TRUOC_CUNG_KY": (p_start, same_end),
        "QUY_NAY": (quarter_range(D.year, q)[0], D),
        "QUY_TRUOC": quarter_range(pqy, pq),
        "TU_DAU_NAM": (dt.date(D.year, 1, 1), D),
        "NAM_NAY": (dt.date(D.year, 1, 1), dt.date(D.year, 12, 31)),
        "NAM_TRUOC": (dt.date(D.year - 1, 1, 1), dt.date(D.year - 1, 12, 31)),
    }

    def N_NGAY_GAN_NHAT(n: int):
        """N ngay lien nhau ke ca hom nay."""
        return (D - dt.timedelta(days=int(n) - 1), D)

    def THANG(m: int, y: int | None = None):
        """Ca thang m/y; khong ghi nam: nam nay neu khong sau hom nay, sau hom nay thi nam truoc (§2)."""
        return month_range(int(y) if y else month_year_for(int(m), D), int(m))

    def QUY(qq: int, y: int | None = None):
        return quarter_range(int(y) if y else quarter_year_for(int(qq), D), int(qq))

    def NGAY(d: int, m: int, y: int | None = None):
        """Mot ngay; khong ghi nam: nam nay neu khong sau hom nay, sau hom nay thi nam truoc (§2)."""
        if not y:
            try:
                y = D.year if dt.date(D.year, int(m), int(d)) <= D else D.year - 1
            except ValueError:
                y = D.year - 1
        dd = dt.date(int(y), int(m), int(d))
        return (dd, dd)

    def NAM(y: int):
        """Ca nam duong lich y."""
        return (dt.date(int(y), 1, 1), dt.date(int(y), 12, 31))

    v.update(N_NGAY_GAN_NHAT=N_NGAY_GAN_NHAT, THANG=THANG, QUY=QUY, NGAY=NGAY, NAM=NAM)
    if lunar is not None:
        def TET(y: int):
            """Tet nam Y (§2): 23 thang Chap -> het mung 10 thang Gieng."""
            r = lunar.tet_range(int(y))
            if r is None:
                raise ValueError(f"lich am khong co Tet {y}")
            return r

        def TRUOC_TET(y: int):
            """Cung so ngay ngay truoc 23 thang Chap cua Tet nam Y."""
            a, b = TET(y)
            n = (b - a).days + 1
            return (a - dt.timedelta(days=n), a - dt.timedelta(days=1))

        v.update(TET=TET, TRUOC_TET=TRUOC_TET)
    return v
