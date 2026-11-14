"""Goi y tat dinh rut tu cau hoi, dua vao prompt truoc khi model viet code.

1. Ky thoi gian cau hoi neu (thang M/Y, quy, ngay, tuan truoc, thang truoc do, Tet...) -> bieu thuc dung
   (THANG(12, 2025), QUY(4, 2025)...) kem khoang ngay da tinh theo rulebook §2.
2. Ten trong du lieu khop voi cau hoi (mat hang, nhom hang, hoat chat, kenh...) -> gia tri loc dung chinh ta.
3. Dang so do: hoi "bao nhieu dia/ly/hop" la so luong, "loi/lai" la lai gop...

Moi goi y chi la nhan dien chuoi trong cau hoi, khong doan dap an. Khi khong chac thi khong goi y.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from .dates import month_range, month_year_for, quarter_range, quarter_year_for, shift_month, to_date, week_range
from .parsing import strip_accents

# ---------------------------------------------------------------------------- ky thoi gian


_ABBR = (
    # teen code: "ko" = khong, "ko"/"kac"/"kua" = co/cac/cua, "jum" = gium, "mik" = minh, "dc" = duoc, "nuot" = luot
    (r"\bk(?=[aou])", "c"),
    (r"\bj(?=[aeiouy])", "gi"),
    (r"\b(?:mik|mk)\b", "minh"),
    (r"\bdc\b", "duoc"),
    (r"\brevenue\b", "doanh thu"),
    (r"\borders?\b", "don"),
    (r"\bbans\b", "ban"),
    (r"\bnuot\b", "luot"),
    (r"\b(thang|hom|tuan|nam|quy|bua|hum|ngay) ni\b", r"\1 nay"),  # tieng mien Trung: "thang ni" = thang nay, "bua ni" = hom nay
    (r"\b(?:hqua|hom qa|hum qua|bua qua|bua hom qua)\b", "hom qua"),
    (r"\b(?:hnay|hum nay|bua nay|bua hom nay)\b", "hom nay"),
    (r"\b(?:trc|trk|trx)\b", "truoc"),
    (r"\b(?:bnhiu|bnhieu|bao nhiu|bnh|nhiu)\b", "bao nhieu"),
    (r"\b(duoc|dc|la|co) nhieu\b(?! nhat\b| hon\b)(?! \w+ nhat\b)", r"\1 bao nhieu"),
    (r"\bbao bao nhieu\b", "bao nhieu"),
    (r"\bdthu\b", "doanh thu"),
    (r"\bthg\b", "thang"),
    (r"\b(?:tu )?dau nam (?:toi|den|cho toi|cho den) (?:chu|gio|nay|hom nay|bay gio)\b", "tu dau nam"),
    (r"\b(thang|ngay|khach|hang|mon|nhom|tho|ai|nhan vien|ben|sp|san pham|nha cung cap) mo\b", r"\1 nao"),  # "khach mo" = khach nao
    (r"\bcat hang\b", "nhap hang"),  # "cat hang" = nhap hang
    (r"\bdh\b", "don hang"),  # "đh" = don hang
    (r"\bbq\b", "binh quan"),  # "dthu bq moi don"
    (r"\b(?:nhta|nhut)\b", "nhat"),  # go sai / mien Nam: "cao nhta", "nhứt"
    (r"\bnhan dien\b", "nhan vien"), (r"\btrung (?:bih|bin|bnh)\b", "trung binh"), (r"\bduco\b", "duoc"),
    (r"\b(?:xo|so) (?:voi|vs|vơi)\b", "so voi"),  # "xo với" = so voi
    # thang viet bang chu: "thang ba" = thang 3 (khong doi "thang nam nay", "thang gieng/chap" am lich)
    (r"\bthang muoi hai\b", "thang 12"), (r"\bthang muoi mot\b", "thang 11"), (r"\bthang muoi\b(?! \w)", "thang 10"),
    (r"\bthang muoi\b(?= (?!mot|hai)\w)", "thang 10"),
    (r"\bthang (?:mot)\b", "thang 1"), (r"\bthang hai\b", "thang 2"), (r"\bthang ba\b", "thang 3"),
    (r"\bthang bon\b", "thang 4"),  # "thang tu" khong dau co the la "thang tu" (tu = from): chi doi ban co dau
    (r"\bthang bay\b", "thang 7"), (r"\bthang tam\b", "thang 8"), (r"\bthang chin\b", "thang 9"),
)
# "sau"/"nam" khong dau nhap nhang ("thang sau" = thang toi, "nam" = nam): chi doi khi cau co dau
_ACCENT_MONTHS = ((r"\btháng sáu\b", "tháng 6"), (r"\btháng tư\b", "tháng 4"), (r"\btháng năm\b(?! nay| ngoái| trước| rồi| \d)", "tháng 5"))
_TELEX = re.compile(r"[a-z]*(?:aa|ee|oo|dd|aw|ow|uw)[a-z]*|[a-z]*[aeiouy][a-z]*[sfrxj]")
_NOT_TELEX = {"vs", "ships", "bus", "max", "size", "box", "plus", "pro", "pass", "glass", "class", "gas", "kids", "jars", "fax", "mix",
              "wax", "fix", "six", "yes", "chips", "tips", "pads", "cups", "nuts", "oats", "dogs", "cats", "phones", "cars", "airblade",
              "power", "super", "sticker", "laser", "cover", "member", "voucher", "sir", "bar", "car", "star", "jar", "for", "or", "nor",
              "mr", "ms", "dr", "gs", "lens", "news", "jeans", "shoes", "lotus", "venus", "taurus", "iris", "ios", "macos", "dos", "kas",
              "moc", "sss", "air"}


def _untelex(tok: str) -> str:
    """Bo dau go telex con sot (hoom -> hom, heets -> het, dduwowcj -> duoc, hangf -> hang).
    Tu tieng Viet khong dau khong bao gio co aa/ee/oo/dd/w hay ket thuc bang s/f/r/x/j."""
    if not _TELEX.fullmatch(tok) or tok in _NOT_TELEX or "ou" in tok or (len(tok) >= 5 and tok.endswith("er")):
        return tok  # tu tieng Anh / ten rieng hay gap ("voucher", "power", "plus") khong phai loi go telex
    t = tok
    for a, b in (("dd", "d"), ("aa", "a"), ("ee", "e"), ("oo", "o"), ("aw", "a"), ("ow", "o"), ("uw", "u"), ("w", "")):
        t = t.replace(a, b)
    return re.sub(r"(?<=[a-z])[sfrxj]$", "", t) if re.search(r"[aeiouy]", t) else t


def _norm(q: str) -> str:
    import unicodedata

    q = unicodedata.normalize("NFC", q).lower()
    q = re.sub(r"\btền\b", "tiền", q)  # go thieu: "bao nhiêu tền" (khong lan voi "tên")
    for rx, rep in _ACCENT_MONTHS:
        q = re.sub(rx, rep, q)
    q = re.sub(r"\b(?:ko|hok|hk|khg|kh0|hông)\b(?= (?:kê|phải|có|tính|biết|được|bán|mua)\b)|\b(?:ko|hok|hk|khg|kh0)\b", "không", q)  # "ko" tron = khong; "kó" = co
    q = re.sub(r"(?<=[^\W\d_])/(?=[^\W\d_])", " / ", q)  # "tawng/giamr" -> "tawng / giamr" (ngay 1/2 khong doi)
    s = strip_accents(q).lower()
    s = re.sub(r"[“”\"'?!;:()\[\]]", " ", s)
    # v4 (12/11/2026): "bao nhieu phan tram" / "5 phan tram" viet nhu "%" dinh vao tu truoc (dung dang nguoi hoi hay go: "5%")
    s = re.sub(r"\s*\bphan tram\b", "%", s)
    s = " ".join(_untelex(t) if t.isalpha() else t for t in s.split())
    for rx, rep in _ABBR:
        s = re.sub(rx, rep, s)
    return re.sub(r"\s+", " ", s).strip()


def _fmt(a: dt.date, b: dt.date) -> str:
    return a.isoformat() if a == b else f"{a.isoformat()} → {b.isoformat()}"


def _valid_day(d: int, m: int, y: int) -> dt.date | None:
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def _day_year_for(d: int, m: int, D: dt.date) -> int:
    """Ngay d/m khong ghi nam: nam nay neu khong sau hom nay, khong thi nam truoc (§2, nhu thang)."""
    x = _valid_day(d, m, D.year)
    return D.year if x is not None and x <= D else D.year - 1


_MONTH_WORD = r"(?:thang|thangs|thg|th|t)"
_YEAR_SEP = r"\s*(?:/|-|\.|\s+nam\s+|\s+)\s*"


@dataclass
class PeriodHint:
    text: str
    code: str
    a: dt.date
    b: dt.date
    note: str = ""

    def line(self) -> str:
        n = f" ({self.note})" if self.note else ""
        return f'- "{self.text}"{n} -> {self.code} = {_fmt(self.a, self.b)}'


def question_periods(question: str, as_of, lunar=None) -> list[PeriodHint]:
    D = to_date(as_of)
    s = _norm(question)
    used = [False] * len(s)
    out: list[PeriodHint] = []

    def take(m: re.Match) -> bool:
        if any(used[m.start():m.end()]):
            return False
        for i in range(m.start(), m.end()):
            used[i] = True
        return True

    explicit_months: list[tuple[int, int]] = []
    month_spans: list[tuple[int, int, int, int]] = []  # (bat dau, ket thuc, thang, nam) cua thang ghi trong cau
    explicit_quarters: list[tuple[int, int]] = []

    # "tu (ngay) 1/1 (nam nay) den het hom nay": mot ky tu ngay do den het hom nay
    for m in re.finditer(r"\btu (?:ngay )?(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{4}))?(?: nam nay)? (?:den|toi|cho den|cho toi|tinh den|tinh toi)(?: het)?(?: ngay)? hom nay\b", s):
        dd, mm = int(m[1]), int(m[2])
        x = _valid_day(dd, mm, int(m[3]) if m[3] else _day_year_for(dd, mm, D)) if 1 <= mm <= 12 else None
        if x and x <= D and take(m):
            out.append(PeriodHint(m[0], f'("{x.isoformat()}", "{D.isoformat()}")', x, D, f"từ {x:%d/%m/%Y} đến hết hôm nay"))
    # "14 ngay ke tu ngay 02/02/2026" va "14 ngay ngay truoc do"
    for m in re.finditer(r"\b(\d{1,3}) ngay (?:dau )?(?:ke tu|tinh tu|bat dau tu|tu)(?: ngay)? (\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", s):
        x = _valid_day(int(m[2]), int(m[3]), int(m[4]))
        n = int(m[1])
        if x and n > 0 and take(m):
            e = x + dt.timedelta(days=n - 1)
            out.append(PeriodHint(m[0], f'("{x.isoformat()}", "{e.isoformat()}")', x, e, f"{n} ngày, kể cả ngày {x:%d/%m/%Y}"))
            for m2 in re.finditer(r"\b(\d{1,3}) ngay (?:ngay |lien |ke |ngay lien )?truoc(?: do)?\b", s):
                n2 = int(m2[1])
                if take(m2):
                    a2, b2 = x - dt.timedelta(days=n2), x - dt.timedelta(days=1)
                    out.append(PeriodHint(m2[0], f'("{a2.isoformat()}", "{b2.isoformat()}")', a2, b2, f"{n2} ngày liền trước ngày {x:%d/%m/%Y}"))
    # ngay co nam: 15/3/2025, ngay 5 thang 3 nam 2025
    for rx in (r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", r"\bngay\s*(\d{1,2})\s*thang\s*(\d{1,2})\s*(?:nam\s*)?(\d{4})\b"):
        for m in re.finditer(rx, s):
            d = _valid_day(int(m[1]), int(m[2]), int(m[3]))
            if d and take(m):
                out.append(PeriodHint(m[0], f"NGAY({d.day}, {d.month}, {d.year})", d, d))
    # quy co nam: quy 4/2025, Q4 2025, quy 4 nam 2025
    for m in re.finditer(r"\b(?:quy|q)\s*0?([1-4])" + _YEAR_SEP + r"(\d{4})\b", s):
        if take(m):
            q, y = int(m[1]), int(m[2])
            explicit_quarters.append((q, y))
            out.append(PeriodHint(m[0], f"QUY({q}, {y})", *quarter_range(y, q)))
    # "3 thang lien nhau ket thuc o thang 12/2025" -> 10, 11, 12/2025 (tu cu den moi)
    for m in re.finditer(r"\b(\d{1,2}) thang (?:lien nhau |lien tiep |gan nhau )?(?:ket thuc|den het|toi het|cuoi cung la|tinh den|tinh toi|den|toi)(?: o| vao| la)? (?:thang )?(\d{1,2})\s*[/\-]\s*(\d{4})\b", s):
        n, mm, y = int(m[1]), int(m[2]), int(m[3])
        if 1 <= n <= 24 and 1 <= mm <= 12 and take(m):
            for k in range(n - 1, -1, -1):
                y2, m2 = shift_month(y, mm, -k)
                explicit_months.append((m2, y2))
                out.append(PeriodHint(f"tháng {m2}/{y2}", f"THANG({m2}, {y2})", *month_range(y2, m2), f"{n} tháng liền nhau kết thúc ở {mm}/{y}"))
    # nhieu thang chung mot nam: "cac thang 7, 8 va 9/2025"
    for m in re.finditer(r"\b" + _MONTH_WORD + r"\s*((?:\d{1,2}\s*(?:,|va|hoac|&)\s*(?:" + _MONTH_WORD + r"\s*)?)+)(\d{1,2})\s*[/\-]\s*(\d{4})\b", s):
        ms = [int(x) for x in re.findall(r"\d{1,2}", m[1])] + [int(m[2])]
        y = int(m[3])
        if all(1 <= x <= 12 for x in ms) and take(m):
            for x in ms:
                explicit_months.append((x, y))
                out.append(PeriodHint(f"tháng {x}/{y}", f"THANG({x}, {y})", *month_range(y, x)))
    # nam 2 chu so: "thang 6/26", "th 9/25", "3/26", "Q4/25" (du lieu cac shop thuoc 2023-2029; so sau '/' > 12 nen khong lan voi ngay/thang)
    for m in re.finditer(r"\b(?:quy|q)\s*([1-4])\s*[/\-]\s*(2[3-9])\b(?!\s*[/\-]\s*\d)", s):
        if take(m):
            q, y = int(m[1]), 2000 + int(m[2])
            explicit_quarters.append((q, y))
            out.append(PeriodHint(m[0], f"QUY({q}, {y})", *quarter_range(y, q), f"năm {y}"))
    for m in re.finditer(r"(?:\b" + _MONTH_WORD + r"\s*)?(?<![\d/])(\d{1,2})\s*[/\-]\s*(2[3-9])\b(?!\s*[/\-]\s*\d)", s):
        mm, y = int(m[1]), 2000 + int(m[2])
        if 1 <= mm <= 12 and take(m):
            explicit_months.append((mm, y))
            month_spans.append((m.start(), m.end(), mm, y))
            out.append(PeriodHint(m[0].strip(), f"THANG({mm}, {y})", *month_range(y, mm), f"tháng {mm}/{y}"))
    # thang co nam: thang 12/2025, th 3/2025, t3-2025, thang 5 nam 2025, 8/2025
    for rx in (r"\b" + _MONTH_WORD + r"\s*(\d{1,2})" + _YEAR_SEP + r"(\d{4})\b", r"(?<![\d/])(\d{1,2})\s*/\s*(\d{4})\b",
               r"(?<![\d/\-])(\d{1,2})-(20\d{2})\b(?!\s*[/\-]\s*\d)", r"\b(20\d{2})-(\d{1,2})\b(?!\s*[/\-]\s*\d)"):
        for m in re.finditer(rx, s):
            mm, y = int(m[1]), int(m[2])
            if mm > 12 and y <= 12:  # dang ISO "2026-07"
                mm, y = y, mm
            if 1 <= mm <= 12 and take(m):
                explicit_months.append((mm, y))
                month_spans.append((m.start(), m.end(), mm, y))
                out.append(PeriodHint(m[0].strip(), f"THANG({mm}, {y})", *month_range(y, mm)))
    # ngay khong nam: ngay 15/3, 15/3
    for m in re.finditer(r"(?:\bngay\s*)?(?<![\d/])(\d{1,2})\s*/\s*(\d{1,2})\b(?!\s*/)", s):
        dd, mm = int(m[1]), int(m[2])
        if 1 <= mm <= 12:
            y = _day_year_for(dd, mm, D)
            d = _valid_day(dd, mm, y)
            if d and take(m):
                out.append(PeriodHint(m[0].strip(), f"NGAY({dd}, {mm}, {y})", d, d, "không ghi năm: §2"))
    # quy khong nam
    for m in re.finditer(r"\bquy\s*0?([1-4])\b|\bq0?([1-4])\b", s):
        if take(m):
            q = int(m[1] or m[2])
            y = quarter_year_for(q, D)
            explicit_quarters.append((q, y))
            out.append(PeriodHint(m[0], f"QUY({q}, {y})", *quarter_range(y, q), "không ghi năm: §2"))
    # thang khong nam: "thang 3" (khong theo sau la nam)
    for m in re.finditer(r"\b(?:thang|thangs|thg|th)\s*(\d{1,2})\b(?!\s*[/\-]\s*\d)", s):  # khong lay 't8' tron (ten hang: 'den tuyp T8')
        mm = int(m[1])
        if 1 <= mm <= 12 and take(m):
            y = month_year_for(mm, D)
            explicit_months.append((mm, y))
            month_spans.append((m.start(), m.end(), mm, y))
            out.append(PeriodHint(m[0], f"THANG({mm}, {y})", *month_range(y, mm), "không ghi năm: §2"))
    # Tet co nam: "Tet 2026", "Tet nam 2026" (truoc buoc nam de khong bi doc thanh NAM(2026))
    tet_years: list[int] = []
    if lunar is not None:
        for m in re.finditer(r"\btet(?: am lich| nguyen dan)?(?: nam)? (20\d{2})\b", s):
            r = lunar.tet_range(int(m[1]))
            if r and take(m):
                tet_years.append(int(m[1]))
                out.append(PeriodHint(m[0], f"TET({m[1]})", *r, "23 tháng Chạp → hết mùng 10 tháng Giêng"))
        # "Tet nam nay" = Tet gan nhat da bat dau tinh den hom nay; "Tet nam truoc/ngoai" = Tet lien truoc (truoc buoc NAM_NAY / NAM_TRUOC)
        for m in re.finditer(r"\btet(?: am lich| nguyen dan)? (?:nam )?(nay|ngoai|truoc|roi)\b", s):
            y = D.year
            r0 = lunar.tet_range(y)
            if r0 is None or r0[0] > D:
                y -= 1
            if m[1] != "nay":
                y -= 1
            r = lunar.tet_range(y)
            if r and take(m):
                tet_years.append(y)
                out.append(PeriodHint(m[0], f"TET({y})", *r, "23 tháng Chạp → hết mùng 10 tháng Giêng"))
    # nam: "nam 2025", "2025" con lai
    for m in re.finditer(r"\b(?:nam\s+)?(20\d{2})\b", s):
        if take(m):
            y = int(m[1])
            out.append(PeriodHint(m[0], f"NAM({y})", dt.date(y, 1, 1), dt.date(y, 12, 31)))

    # khoang "tu thang 9/2025 den (het) thang 11/2025": ca khoang + tung thang o giua
    is_range = False
    sp = sorted(month_spans)
    if len(sp) >= 2 and re.search(r"\b(?:den|toi)(?: het)?\b|→|\s-\s", s[sp[0][1]:sp[-1][0]]):  # "den/toi" nam GIUA hai thang
        is_range = True
        (m1, y1), (m2, y2) = sp[0][2:], sp[-1][2:]
        if (y1, m1) < (y2, m2):
            a1, _ = month_range(y1, m1)
            _, b2 = month_range(y2, m2)
            k, (yy, mm) = 0, (y1, m1)
            have = {(p.a, p.b) for p in out}
            while (yy, mm) <= (y2, m2) and k < 36:
                r = month_range(yy, mm)
                if r not in have:
                    out.append(PeriodHint(f"tháng {mm}/{yy}", f"THANG({mm}, {yy})", *r, "tháng nằm giữa khoảng câu hỏi nêu"))
                yy, mm = shift_month(yy, mm, 1)
                k += 1
            out.append(PeriodHint(f"từ {m1}/{y1} đến hết {m2}/{y2}", f"(THANG({m1}, {y1})[0], THANG({m2}, {y2})[1])", a1, b2, "cả khoảng, gồm hai đầu"))
    # ---- cach noi tuong doi
    def has(rx: str) -> re.Match | None:
        for m in re.finditer(rx, s):
            if not any(used[m.start():m.end()]):
                return m
        return None

    ws, _ = week_range(D)
    pws = ws - dt.timedelta(days=7)
    py, pm = shift_month(D.year, D.month, -1)
    p_start, p_end = month_range(py, pm)
    same_end = min(p_end, dt.date(py, pm, min(D.day, p_end.day)))
    q_now = (D.month - 1) // 3 + 1

    # "tu mung 1 toi het hom nay", "tu dau thang den nay" = thang nay (ngay 1 -> het hom nay)
    mtd = has(r"\btu (?:mung 1|ngay 1|ngay mung 1|dau thang)(?: (?:thang )?nay)?(?: (?:toi|den|cho toi|cho den|tinh toi|tinh den)(?: het)? (?:hom nay|nay|gio|bay gio|chu))?\b") \
        or has(r"\b(?:thang|th) nay(?: tinh)?(?: tu dau)? (?:toi|den|cho toi|cho den|tinh toi|tinh den)(?: het)? (?:hom nay|nay|gio|bay gio|chu)\b")
    if mtd:
        take(mtd)
    if m := has(r"\bhom nay\b"):
        out.append(PeriodHint(m[0], "HOM_NAY", D, D))
    if m := has(r"\bhom qua\b"):
        y = D - dt.timedelta(days=1)
        out.append(PeriodHint(m[0], "HOM_QUA", y, y))
    if m := has(r"\bhom kia\b"):
        y = D - dt.timedelta(days=2)
        out.append(PeriodHint(m[0], f"NGAY({y.day}, {y.month}, {y.year})", y, y))
    if m := has(r"\btuan nay\b"):
        out.append(PeriodHint(m[0], "TUAN_NAY", ws, D))
    if m := has(r"\btuan (?:truoc|roi)\b"):
        out.append(PeriodHint(m[0], "TUAN_TRUOC", pws, pws + dt.timedelta(days=6), "cả tuần thứ Hai → Chủ nhật"))
    if m := has(r"\b(\d{1,4}) ngay (?:toi|sap toi|tiep theo|nua)\b"):
        n = int(m[1])
        out.append(PeriodHint(m[0], f'("{D.isoformat()}", "{(D + dt.timedelta(days=n)).isoformat()}")', D, D + dt.timedelta(days=n),
                              f"từ hôm nay đến hết hôm nay + {n} ngày"))
    if m := has(r"\b(\d{1,3}) ngay (?:gan nhat|gan day|qua|vua qua|vua roi|cuoi|tro lai day|lien gan nhat)\b"):
        n = int(m[1])
        out.append(PeriodHint(m[0], f"N_NGAY_GAN_NHAT({n})", D - dt.timedelta(days=n - 1), D, "kể cả hôm nay"))

    m_now = has(r"\b(?:thang|th) nay\b") or mtd
    m_prev = has(r"\b(?:thang|th) (?:truoc do|lien truoc|ke truoc|truoc|roi)\b")
    if m_now:
        out.append(PeriodHint(m_now[0], "THANG_NAY", D.replace(day=1), D, "từ ngày 1 đến hết hôm nay"))
    if m_prev and re.search(r"\bthang sau\b", s) and (is_range or len(explicit_months) >= 2):
        m_prev = None  # "thang sau co hon thang truoc khong": so cac thang lien nhau trong khoang, khong phai mot ky rieng
    if m_prev:
        if explicit_months and not m_now:
            mm, y = explicit_months[0]
            y2, m2 = shift_month(y, mm, -1)
            out.append(PeriodHint(m_prev[0], f"THANG({m2}, {y2})", *month_range(y2, m2), f"tháng liền trước tháng {mm}/{y}"))
        elif m_now:
            out.append(PeriodHint(m_prev[0], "THANG_TRUOC_CUNG_KY", p_start, same_end, "so tháng này với tháng trước: cùng số ngày, §2"))
        else:
            out.append(PeriodHint(m_prev[0], "THANG_TRUOC", p_start, p_end, "cả tháng"))
    if m := has(r"\bquy nay\b"):
        out.append(PeriodHint(m[0], "QUY_NAY", quarter_range(D.year, q_now)[0], D, "từ đầu quý đến hết hôm nay"))
    if m := has(r"\bquy (?:truoc do|lien truoc|truoc|roi)\b"):
        if explicit_quarters:
            q, y = explicit_quarters[0]
            q2, y2 = (q - 1, y) if q > 1 else (4, y - 1)
            out.append(PeriodHint(m[0], f"QUY({q2}, {y2})", *quarter_range(y2, q2), f"quý liền trước Q{q}/{y}"))
        else:
            pq, pqy = (q_now - 1, D.year) if q_now > 1 else (4, D.year - 1)
            out.append(PeriodHint(m[0], "QUY_TRUOC", *quarter_range(pqy, pq)))
    if (m := has(r"\bcung (?:ky|thang|thoi diem|quy|ky nay|thang do|thang nay) (?:cua )?nam (?:truoc|ngoai|roi)\b")) and (explicit_months or explicit_quarters):
        take(m)
        if explicit_months:
            mm, y = explicit_months[0]
            out.append(PeriodHint(m[0], f"THANG({mm}, {y - 1})", *month_range(y - 1, mm), f"cùng tháng {mm} năm trước"))
        else:
            q, y = explicit_quarters[0]
            out.append(PeriodHint(m[0], f"QUY({q}, {y - 1})", *quarter_range(y - 1, q), f"cùng quý {q} năm trước"))
    if m := has(r"\btu dau nam\b"):
        out.append(PeriodHint(m[0], "TU_DAU_NAM", dt.date(D.year, 1, 1), D))
    elif m := has(r"\bnam nay\b"):
        out.append(PeriodHint(m[0], "NAM_NAY", dt.date(D.year, 1, 1), dt.date(D.year, 12, 31), "cả năm; dữ liệu sau hôm nay coi như chưa có"))
    if m := has(r"\bnam (?:ngoai|truoc|roi)\b"):
        out.append(PeriodHint(m[0], "NAM_TRUOC", dt.date(D.year - 1, 1, 1), dt.date(D.year - 1, 12, 31)))

    if lunar is not None:
        if not tet_years and (tet_m := has(r"\btet\b")):  # Tet khong ghi nam: Tet gan nhat da bat dau tinh den hom nay
            y = D.year
            r = lunar.tet_range(y)
            if r is None or r[0] > D:
                y -= 1
            r = lunar.tet_range(y)
            if r:
                tet_years.append(y)
                out.append(PeriodHint(tet_m[0], f"TET({y})", *r, "Tết gần nhất tính đến hôm nay; 23 tháng Chạp → hết mùng 10 tháng Giêng"))
        if tet_years and has(r"\btruoc tet\b"):
            y = tet_years[0]
            r = lunar.tet_range(y)
            n = (r[1] - r[0]).days + 1
            out.append(PeriodHint("trước Tết", f"TRUOC_TET({y})", r[0] - dt.timedelta(days=n), r[0] - dt.timedelta(days=1), "cùng số ngày ngay trước Tết"))
    # "tu dau nam den het hom nay", "tuan nay tinh toi hom nay", "7 ngay gan nhat tinh ca hom nay": hom nay chi la diem cuoi
    # cua mot ky khac da ket thuc o hom nay, khong phai ky rieng
    if re.search(r"\b(?:toi|den|cho toi|cho den|tinh toi|tinh den|tinh luon|tinh ca|ke ca|bao gom|gom ca)(?: het| luon| ca)?(?: ngay)? hom nay\b", s) \
            and any(p.code != "HOM_NAY" and p.b == D and p.a < D for p in out):
        out = [p for p in out if p.code != "HOM_NAY"]
    if any(p.a == dt.date(D.year, 1, 1) and p.b == D and p.code != "NAM_NAY" for p in out):  # "tu 1/1 den het hom nay, doanh thu nam nay"
        out = [p for p in out if p.code != "NAM_NAY"]
    seen, uniq = set(), []
    for p in out:
        if p.code not in seen:
            seen.add(p.code)
            uniq.append(p)
    return uniq


# ---------------------------------------------------------------------------- ten trong du lieu


_HINT_COLUMNS = ("item", "item_code", "group", "dm_hoat_chat", "channel", "customer_group", "payment", "employee",
                 "employee__ho_ten", "customer", "customer_id", "supplier")
_STOP = {"va", "cua", "cho", "the", "hang", "loai", "mon", "cai", "ban", "duoc", "bao", "nhieu", "bn", "thang", "nam", "ngay"}


# cung mot mon / mat hang, mien Bac - mien Nam goi khac (de public: "Lạc rang muối" ~ "Đậu phộng rang muối",
# "Má lợn nướng" ~ "Má heo nướng", "Mực rán" ~ "Mực chiên", "Pho mát" ~ "Phô mai", "Cát xây trát" ~ "Cát xây tô").
# Doi ca hai phia ve mot cach ghi nen khong lam lech ten khac.
_SYNONYMS = (
    (r"\bdau phong\b", "lac"), (r"\blon\b", "heo"), (r"\bran\b", "chien"), (r"\bpho mat\b", "pho mai"), (r"\btrat\b", "to"),
    (r"\bngo\b", "bap"), (r"\bvung\b", "me"), (r"\bmi chinh\b", "bot ngot"), (r"\bcafe\b", "ca phe"), (r"\b(?:da ua|yaourt|yogurt)\b", "sua chua"),
    (r"\bmi goi\b", "mi tom"), (r"\bgio lua\b", "cha lua"), (r"\bxi dau\b", "nuoc tuong"), (r"\bhoa qua\b", "trai cay"),
    (r"\bdua chuot\b", "dua leo"), (r"\blop xe\b", "vo xe"), (r"\bbim\b", "ta"), (r"\bthuoc danh rang\b", "kem danh rang"),
    (r"\bxa phong\b", "xa bong"), (r"\bbanh da nem\b", "banh trang"), (r"\bdau phu\b", "dau hu"),
)


def _toks(s: str) -> list[str]:
    x = " ".join(t for t in re.split(r"[^0-9a-z]+", strip_accents(str(s)).lower()) if t)
    for rx, rep_ in _SYNONYMS:
        x = re.sub(rx, rep_, x)
    return x.split()


def _data_range(biz) -> tuple | None:
    """Khoang ngay co giao dich cua shop: hop cac bang ban (S, O), nhap (N), phieu sua (P)."""
    import pandas as pd

    ds = []
    sales = getattr(biz, "sales", None)
    for obj, col in ((getattr(sales, "S", None), "order_date"), (getattr(sales, "O", None), "order_date"),
                     (getattr(getattr(biz, "imports", None), "N", None), "date"), (getattr(getattr(biz, "repair", None), "P", None), "date")):
        if obj is not None and len(obj) and col in obj:
            ds += [obj[col].min(), obj[col].max()]
    ds = [pd.Timestamp(x) for x in ds if pd.notna(x)]
    return (min(ds).date(), max(ds).date()) if ds else None


_PAY_ALIASES = (
    (r"\bcash\b|tien mat", ("tiền mặt", "bằng tm", "tiền tươi")),
    (r"\btransfer\b|\bbank\b|chuyen khoan", ("chuyển khoản", "ck")),
    (r"e-?wallet|vi dien tu|\bmomo\b|zalo ?pay|vnpay|shopee ?pay", ("ví điện tử", "ví momo")),
    (r"\bcod\b|thu ho", ("cod", "xi ô đi", "thu hộ", "giao hàng thu tiền", "ship cod")),
    (r"\bdebt\b|ghi no", ("ghi nợ", "ghi sổ")),
)


# kenh ban ghi bang ma (store / phone / zalo_oa): "ban tai quay", "don qua dien thoai", "dat qua Zalo"
_CHANNEL_ALIASES = (
    (r"\bstore\b|tai (quay|tiem|cua hang)|truc tiep", ("tại quầy", "tại cửa hàng", "tại tiệm", "tại quán", "tại shop", "tại chỗ", "trực tiếp", "ở quầy")),
    (r"\bphone\b|dien thoai", ("qua điện thoại", "gọi điện", "đặt qua điện thoại")),
    (r"zalo", ("zalo", "zalo oa", "qua zalo")),
)


def _payment_aliases(values, col: str = "payment") -> list[tuple[str, str, str]]:
    """(cot, gia tri trong du lieu, cach goi tieng Viet) cho cac ma thanh toan / kenh ban: 'ewallet' <- 'vi dien tu', 'cod' <- 'xi o di',
    'store' <- 'tai quay'."""
    out = []
    for v in values:
        k = strip_accents(str(v)).lower()
        for rx, names in (_CHANNEL_ALIASES if col == "channel" else _PAY_ALIASES):
            if re.search(rx, k):
                out += [(col, v, a) for a in names if strip_accents(a).lower() != k]
    return out


class EntityIndex:
    """Gia tri cac cot loc chinh cua S (va N) de do ten trong cau hoi."""

    def __init__(self, columns: dict[str, list[str]], rx_col: str | None = None,
                 aliases: list[tuple[str, str, str]] | None = None, kinds: frozenset = frozenset()):
        self.rx_col = rx_col  # cot co/khong ke don cua S (ten khac nhau giua cac shop)
        self.kinds = frozenset(kinds)  # loai cua hang: 'repair', 'phone'; 'gender' khi du lieu co gioi tinh khach
        self.gender_col: str | None = None
        self.addr_cols: list | None = None  # cot dia chi khach (None: khong biet)
        self.data_range: tuple | None = None  # (ngay dau, ngay cuoi) co giao dich ban / nhap / phieu sua
        self.caps: frozenset | None = None  # phan du lieu dung san shop co (None: khong biet, khong kiem)
        self.entries: list[tuple[str, str, list[str]]] = []
        for col, vals in columns.items():
            for v in vals:
                t = _toks(v)
                if t:
                    self.entries.append((col, v, t))
        # alias: (cot, gia tri chuan, cach ghi khac) -> khop theo cach ghi, goi y gia tri chuan
        seen = {(c, v, tuple(t)) for c, v, t in self.entries}
        for col, v, alias in aliases or []:
            t = _toks(alias)
            if t and (col, v, tuple(t)) not in seen:
                seen.add((col, v, tuple(t)))
                self.entries.append((col, v, t))

    @classmethod
    def from_business(cls, biz) -> "EntityIndex":
        cols: dict[str, list[str]] = {}
        sales = getattr(biz, "sales", None)
        S = getattr(sales, "S", None)
        rx_col = None
        if S is not None and len(S):
            extra = [c for c in S.columns if c.startswith("dm_") and c not in _HINT_COLUMNS
                     and any(k in c.replace("_", "") for k in ("hoatchat", "thanhphan", "nhom", "loai", "danhmuc", "phanloai", "hang", "size", "so"))]
            extra += [c for c in ("dong_may", "hang") if c in S.columns and c not in extra]
            rx_col = next((c for c in S.columns if c.startswith("dm_") and any(k in c.replace("_", "") for k in ("kedon", "theodon"))), None)
            for c in list(_HINT_COLUMNS) + extra:
                if c in S.columns and S[c].dtype == object:
                    u = S[c].dropna().astype(str).unique().tolist()
                    if 0 < len(u) <= 3000:
                        cols[c] = u
        imp = getattr(biz, "imports", None)
        N = getattr(imp, "N", None)
        if N is not None and len(N):
            for c in ("supplier", "item"):
                if c in N.columns and N[c].dtype == object:
                    u = N[c].dropna().astype(str).unique().tolist()
                    key = c if c not in cols else f"N.{c}"
                    if c == "item":
                        u = [x for x in u if x not in set(cols.get("item", []))]
                        key = "N.item"
                    if 0 < len(u) <= 3000:
                        cols[key] = u
        kinds, aliases = set(), []
        # phuong thuc thanh toan ghi bang ma (cash / transfer / ewallet / cod / debt, README giai nghia): cau hoi viet tieng Viet
        # ("tra bang vi dien tu", "don tra xi o di" = COD). Khong them chu "vi" don (khong dau trung "vi" = boi vi).
        aliases += _payment_aliases(cols.get("payment", [])) + _payment_aliases(cols.get("channel", []), "channel")
        if S is not None and "employee" in S.columns:
            from .staff import employee_names

            emap = employee_names(getattr(biz, "tables", {}) or {}, S["employee"])
            if emap:
                aliases += [("employee_name", v, v) for v in sorted(set(emap.values())) if len(_toks(v)) >= 2]
        rep = getattr(biz, "repair", None)
        P = getattr(rep, "P", None)
        if P is not None and len(P):
            from .repair import _JOBS

            kinds.add("repair")
            for v, g in P.groupby("dong_xe")["dong_xe_raw"]:
                aliases += [("P.dong_xe", str(v), str(a)) for a in [v, *g.dropna().astype(str).unique()]]
            if "tho" in P.columns:
                aliases += [("P.tho", str(v), str(v)) for v in P["tho"].dropna().astype(str).unique() if len(_toks(v)) >= 2]
            for job, syn in _JOBS.items():
                # bo cach ghi mot tu de nham voi tu thuong ('bình' trong 'trung bình', 'rửa', 'vỏ'…)
                aliases += [("P.viec", job, a) for a in (job, *syn) if len(_toks(a)) >= 2 or _toks(a)[0] in ("bugi", "curoa", "nsd", "motul", "castrol", "bdtq")]
        if getattr(biz, "installments", None) is not None or (S is not None and "dong_may" in S.columns):
            kinds.add("phone")
        from .loader import _split_name

        gcols = [f"{t}.{c}" for t, df in (getattr(biz, "tables", {}) or {}).items() for c in df.columns
                 if " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(str(c))).split()) in ("gioi tinh", "nam nu", "phai", "gioi tinh kh", "gioi tinh khach")]
        if gcols or (S is not None and "customer_gender" in S.columns):
            kinds.add("gender")
        idx_gender = ("S.customer_gender" if S is not None and "customer_gender" in S.columns else (gcols[0] if gcols else None))
        acols = [f'T["{t}"]["{c}"]' for t, df in (getattr(biz, "tables", {}) or {}).items() for c in df.columns
                 if " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(str(c))).split()) in ("addr", "address", "dia chi", "dia chi khach", "dia chi giao", "duong", "dia chi kh")]
        out = cls(cols, rx_col, aliases, frozenset(kinds))
        out.gender_col = idx_gender
        out.addr_cols = acols
        out.data_range = _data_range(biz)
        caps = {c for c, attr in (("imports", "imports"), ("debt", "debt"), ("inventory", "inventory"), ("repair", "repair"),
                                  ("installments", "installments"), ("phone_orders", "phone_orders")) if getattr(biz, attr, None) is not None}
        if any(any(k in " ".join(re.sub(r"[^a-z0-9]+", " ", _split_name(str(c))).split()) for k in ("ly do", "li do", "nguyen nhan", "reason"))
               for df in (getattr(biz, "tables", {}) or {}).values() for c in df.columns):
            caps.add("reasons")
        inv = getattr(biz, "inventory", None)
        if inv is not None and getattr(inv, "lots", None) is not None and getattr(inv, "lot_moves", None) is not None:
            caps.add("lots")
        if S is not None and len(S):
            caps.add("sales")
            if "cogs" in S.columns and S["cogs"].notna().any():
                caps.add("profit")
            if any(c in S.columns and S[c].notna().any() for c in ("customer_group", "kh_nhom_khach", "kh_hang_khach", "kh_loai_khach")) \
                    or any(c.startswith("kh_") and any(k in c for k in ("nhom", "hang", "loai")) and S[c].notna().any() for c in S.columns):
                caps.add("customer_groups")
            if "unit" in S.columns and S["unit"].notna().any():
                caps.add("units")
            if ("group" in S.columns and S["group"].notna().any()) or any(
                    c.startswith("dm_") and any(k in c.replace("_", "") for k in ("nhom", "loai", "danhmuc", "phanloai")) for c in S.columns):
                caps.add("groups")
        O_ = getattr(sales, "O", None)
        for X in (S, O_):
            if X is not None and len(X) and "customer_key" in X.columns:
                named = X["customer_key"].notna() & ~X["is_walk_in"].fillna(False).astype(bool) if "is_walk_in" in X.columns else X["customer_key"].notna()
                if named.any():
                    caps.add("customers")
                    break
        try:
            from .delivery import build_delivery
            from .kernel import _expense_table

            if build_delivery(biz.tables, getattr(sales, "O", None)) is not None:
                caps.add("delivery")
            if _expense_table(biz.tables) is not None:
                caps.add("expenses")
        except Exception:
            pass
        out.caps = frozenset(caps)
        return out

    def match(self, question: str, k: int = 4, partial: bool = True) -> list[tuple[str, str]]:
        import unicodedata

        qt = _toks(question)
        qs = set(qt)
        ql = unicodedata.normalize("NFC", str(question)).lower()
        accented = strip_accents(ql) != ql
        qwords = set(re.findall(r"\w+", ql))
        full, part = [], []  # full: (bat dau, ket thuc, cot, gia tri) theo vi tri tu trong cau hoi
        for col, v, t in self.entries:
            n = len(t)
            pos = next((i for i in range(len(qt) - n + 1) if qt[i:i + n] == t), None)
            if pos is not None and n == 1 and accented and not col.startswith("P."):
                vw = re.findall(r"\w+", unicodedata.normalize("NFC", str(v)).lower())
                # v4 (12/11/2026): giu khop khi tu do duoc go KHONG dau trong cau (cau tron co dau / khong dau: "cho minh hoi ... dien ạ")
                if len(vw) == 1 and strip_accents(vw[0]) != vw[0] and vw[0] not in qwords and strip_accents(vw[0]) not in qwords:
                    pos = None  # "trả" (tra tien) khong phai nhom "Trà"
            if pos is not None:
                full.append((pos, pos + n, col, v))
                continue
            if col.startswith("P.") or not partial:
                continue
            sig = [x for x in t if len(x) >= 2 and x not in _STOP]
            hit = [x for x in sig if x in qs]
            if len(sig) >= 2 and len(hit) >= 2 and len(hit) / len(sig) >= 0.75:
                part.append((len(hit) / len(sig), len(hit), col, v))
        # bo ten nam tron trong mot ten dai hon da khop (vd nhom 'rau' trong 'Rau củ kho quẹt')
        full = [f for f in full if not any(g[0] <= f[0] and f[1] <= g[1] and (g[1] - g[0]) > (f[1] - f[0]) for g in full)]
        full.sort(key=lambda f: (f[0], -(f[1] - f[0])))
        res = [(col, v) for _, _, col, v in full]
        # khop mot phan chi dung khi cot do khong co ten khop tron (tranh goi y 'D10' khi cau hoi ghi 'D12')
        full_cols = {col for col, _ in res}
        part.sort(key=lambda x: (-x[0], -x[1]))
        for _, _, col, v in part:
            if col not in full_cols and (col, v) not in res:
                res.append((col, v))
        out, per = [], {}
        for col, v in res:
            per[col] = per.get(col, 0) + 1
            if per[col] <= k:
                out.append((col, v))
        return out[: 3 * k]


# ---------------------------------------------------------------------------- dang so do

_UNITS = (
    "dia|ly|coc|cai|hop|chai|goi|vi|vien|tuyp|lon|lit|kg|suat|to|chiec|doi|cuon|tui|thung|ket|bich|ong|"
    "cay|trai|qua|bao|mieng|xien|met|m3|bo|phan"
)
_QTY_RX = re.compile(r"\b(?:bao nhieu|bn|may|so)\s+(?:" + _UNITS + r"|lo(?! hang| nao))\b(?! tram)")


def _weekday_avg_hint(s: str) -> list[str]:
    wd = re.search(r"\b(?:moi|cac|1) (?:ngay )?thu (hai|ba|tu|nam|sau|bay|2|3|4|5|6|7)\b|\bmoi (?:ngay )?chu nhat\b|\bcac (?:ngay )?chu nhat\b", s)
    if not (re.search(r"\b(trung binh|binh quan|tb)\b", s) and (wd or re.search(r"\bmoi ngay\b|\b1 ngay\b|\bmot ngay\b", s))):
        return []
    if wd:
        names = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tu": 2, "4": 2, "nam": 3, "5": 3, "sau": 4, "6": 4, "bay": 5, "7": 5}
        w = names.get(wd.group(1), 6) if wd.group(1) else 6
        return [f"Trung bình mỗi ngày thứ X (rulebook): tổng của các ngày thứ đó trong kỳ / count_weekday(a, b, {w}) (số ngày thứ đó trong kỳ, "
                f"KỂ CẢ ngày không có đơn/phiếu; 0 = thứ Hai … 6 = Chủ nhật); lọc ngày: x[x.<cột ngày>.dt.weekday == {w}]."]
    return ["Trung bình mỗi ngày: tổng trong kỳ / len(days_in(a, b)) (số ngày của kỳ, kể cả ngày không có đơn)."]


def _general_hints(s: str) -> list[str]:
    """Cac dang hoi chung cua bo public (rulebook §1-§5) chua co goi y rieng."""
    out = []
    if re.search(r"\btien mat\b", s) and re.search(r"\bca (sang|chieu|toi|trua|dem)\b|\b(moi|tung|cac) ca\b|\bbao cao ca\b|\bchot ca\b|\bdung ca\b", s):
        out.append("Tiền mặt theo ca (rulebook) -> bảng báo cáo tiền mặt theo ca trong T (nếu shop có): chỉ ghi TIỀN MẶT, theo GIỜ THU TIỀN, "
                   "gồm cả tiền khách trả nợ; tiền mặt ròng = thu trừ hoàn khách trả hàng. Doanh thu, số đơn theo ca thì tính từ đơn hàng "
                   "theo giờ đặt (revenue(kỳ, ca=...)), không lấy từ báo cáo ca.")
    item_ctx = re.search(r"\b(mon|mat hang|sp|san pham|hang|thuoc|nuoc|ly|banh)\b", s) and not re.search(r"\b(khach|nha cung cap|ncc|nhan vien|tho|dong may)\b", s)
    top = re.search(r"\btop\s*(\d{1,2}|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\b|(?<![\d/\-])\b(\d{1,2}) (mon|mat hang|sp|san pham)\b", s)
    if item_ctx and re.search(r"\b(ban chay|chay nhat|ban nhieu nhat|ban duoc nhieu nhat|hot nhat|nhieu (ly|cai|hop|chai|goi|vien|phan|dia|to|suat) nhat)\b|\btop\s*\d", s):
        words = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
        g = next(g for g in top.groups()[:2] if g) if top else "1"
        k = words.get(g) or int(g)
        out.append(f"Bán chạy (§3) = theo SỐ LƯỢNG quy về đơn vị cơ bản (không theo tiền, số dòng) -> top_items(kỳ, k={k}) (lọc group=... nếu hỏi trong một nhóm); "
                   + ("answer('list', top_items(kỳ, k=%d)['item'].tolist())." % k if k > 1 else "answer('string', top_items(kỳ, k=1)['item'].iloc[0])."))
    if re.search(r"%|phan tram|\bti le\b|\bty le\b", s):
        if re.search(r"\b(ti|ty) le (don )?(bi )?huy\b|\b(phan tram|%) (don )?(bi )?huy\b|\bdon (bi )?huy\b.*(%|phan tram|ti le)", s):
            out.append("Tỉ lệ đơn huỷ -> answer('percent', cancel_rate(kỳ)) (đơn huỷ / MỌI đơn đặt trong kỳ, 2 chữ số).")
        elif re.search(r"\b(chiem|dong gop|ti trong|ty trong|chia)\b", s):
            out.append("Tỉ trọng = pct(<phần>, <toàn bộ>) -> vd answer('percent', pct(revenue(kỳ, <lọc>), revenue(kỳ))) "
                       "(§1: mẫu số là doanh thu CẢ SHOP cùng kỳ, tính cả đơn khách vãng lai).")
        elif re.search(r"\b(tang|giam|thay doi|hon hay kem|hon kem|cao hay thap|len hay xuong|so voi|chenh|tang truong)\b", s):
            out.append("Phần trăm thay đổi (§2) = pct_change(<số đo kỳ hỏi>, <số đo kỳ so>) (so với kỳ trước; giảm ra số âm) -> answer('percent', ...).")
    elif re.search(r"\b(hon|kem|chenh lech|chenh)\b.*\bbao nhieu\b|\b(tang|giam) bao nhieu\b", s) and re.search(r"\bso voi\b|\bthang truoc\b|\bky truoc\b", s):
        out.append("Chênh lệch (số, không phải %) = <số đo kỳ hỏi> − <số đo kỳ so> (ít hơn ra số âm, giữ đơn vị).")
    # cau hoi ky gop (tra gop qua han, con no ky gop) co goi y rieng (installments_as_of), khong dung so no thuong
    debt = re.search(r"\bno\b|\bcong no\b|\bmua chiu\b|\bban chiu\b|\bghi so\b|\b(con )?thieu tien\b|\bchua tra (het )?tien\b", s) \
        and not re.search(r"\b(tra gop|ky gop|gop)\b", s)
    two = debt and re.search(r"\b(no them|chiu them|ghi no them|no moi)\b", s) and re.search(r"\btra\b", s)
    if two:
        out.append("Câu hỏi HAI số: nợ thêm và trả nợ trong kỳ (§4) -> answer('list', [nợ thêm, trả nợ]) theo thứ tự câu hỏi; "
                   "một khách thì lọc theo khách: debt_added(kỳ) cột no_them và debt_paid(kỳ) cột amount, lọc cột customer / customer_id của khách đó.")
    elif debt and re.search(r"\b(tra no|thu no|thu hoi|thu ve|thu duoc|doi duoc|khach tra)\b", s) and not re.search(r"\btuoi no\b", s):
        out.append("Trả nợ / thu nợ trong kỳ (§4: các khoản trả SAU ngày mua có ngày trả trong kỳ, của mọi đơn ghi nợ; tiền trả cùng ngày mua không tính) -> "
                   "answer('number', round_money(debt_paid(kỳ).amount.sum())).")
    elif debt and re.search(r"\bno them\b|\bno moi\b|\b(mua|ban) chiu (them|moi)?\b.*\b(bao nhieu tien|tong)\b|\bphat sinh\b", s) and not re.search(r"\btuoi no\b|\bcon no\b", s):
        out.append("Nợ thêm trong kỳ (§4: phần tổng đơn chưa trả ngay trong ngày mua, của đơn ghi nợ đặt trong kỳ) -> "
                   "answer('number', round_money(debt_added(kỳ).no_them.sum())).")
    elif debt and not re.search(r"\btuoi no\b", s):
        kk = re.search(r"\btop\s*(\d+)|\b(\d+) (khach|nguoi)\b", s)
        kk = (kk[1] or kk[2]) if kk else None
        if re.search(r"\b(bao nhieu|may|so|bn) (khach|nguoi)\b", s):
            out.append("Số khách còn nợ tại ngày D (§4) -> len(debt_by_customer(D)) (khách còn nợ > 0; đơn khách lẻ không có dòng).")
        elif kk or re.search(r"\bai\b|\bkhach nao\b|\bnhieu nhat\b|\blon nhat\b", s):
            out.append(f"Khách nợ nhiều nhất tại ngày D (§4) -> debt_by_customer(D) (cột khach, con_no, so_don; đã sắp con_no giảm dần)"
                       + (f"; top {kk} kèm số nợ: answer('table', debt_by_customer(D).head({kk})[['khach', 'con_no']])." if kk else "; một khách: answer('string', debt_by_customer(D).khach.iloc[0])."))
        elif re.search(r"\btong\b|\bbao nhieu tien\b|\bcon no\b", s):
            out.append("Tổng nợ còn lại tại ngày D (§4) -> d = debt_as_of(D); d.con_no.sum() (gồm cả đơn khách lẻ).")
    if re.search(r"\bnhap (hang|ve|vao|kho)?\b", s) and re.search(r"\btien\b|\bgia tri\b", s) and not re.search(r"\bgia nhap\b|\b(binh quan|trung binh|tb)\b|\btru\b", s) \
            and not re.search(r"\b(nha cung cap|ncc) nao\b|\btu (nha cung cap|ncc)\b|\bnao\b", s):
        out.append("Tiền nhập hàng trong kỳ (§5) -> import_cost(kỳ) (tổng tiền các phiếu nhập có ngày phiếu trong kỳ, bỏ phiếu huỷ; gồm phiếu nhập đầu kỳ).")
    if re.search(r"\bnha cung cap\b|\bncc\b", s):
        if re.search(r"\b(chinh|uy tin|tin cay|quen|thuong xuyen|ruot|than thiet)\b", s):
            out.append("Nhà cung cấp chính (§5) = NHIỀU PHIẾU NHẬP nhất (bỏ phiếu huỷ), khác nhập nhiều tiền nhất (bằng nhau theo tên) -> "
                       "answer('string', top_suppliers(kỳ, by='receipts').supplier.iloc[0]).")
        elif re.search(r"\bnhieu (tien )?(hang )?nhat\b|\blon nhat\b|\bcao nhat\b|\bnhieu nhat\b|\bnao\b.*\bnhat\b", s):
            out.append("Nhà cung cấp nhập nhiều nhất (§5) = TIỀN NHẬP cao nhất -> answer('string', top_suppliers(kỳ, by='amount').supplier.iloc[0]).")
    if re.search(r"\bnhap\b.*\btru\b.*\bban\b|\bchenh\b.*\bnhap\b.*\bban\b|\bnhap (vo|vao) tru\b", s):
        out.append("Nhập trừ bán (§5) = số lượng nhập trong kỳ (N, bỏ phiếu huỷ) − số lượng bán (đơn tính doanh thu, qty_sold(kỳ, item=...)), "
                   "quy CẢ HAI về cùng một đơn vị (đơn vị câu hỏi hỏi).")
    if re.search(r"\bhet hang\b", s):
        out.append("Mặt hàng hết hàng cuối ngày D (§5) = hàng mua về để bán (không tính món pha chế) có tồn hết ngày D bằng 0 -> s = stock_at(D); s[s.ton <= 0].")
    if re.search(r"\b(het han|can date|hsd|han dung|han su dung)\b", s):
        out.append("Lô / hạn dùng (rulebook) -> expiring_lots(D, kỳ hạn dùng) = lô CÒN TỒN lúc hết ngày D có hạn dùng trong kỳ (cột lot, item, expiry, ton, "
                   "von = ton × giá nhập 1 đơn vị cơ bản của chính lô); 'N ngày tới' = từ hôm nay D đến hết D + N; expiring_items(D, kỳ) = mặt hàng theo hạn sớm nhất "
                   "(cùng hạn theo mã); nearest_expiry(D, item=...) = hạn dùng 'YYYY-MM-DD' của lô gần hết hạn nhất (còn tồn, chưa quá hạn tại D).")
    if re.search(r"\b(may|bao nhieu|so|bn) (ngay|bua|hom)\b", s) and re.search(r"\b(tu|tren|hon|it nhat|qua|duoi)\b \d", s):
        out.append("Đếm ngày theo ngưỡng doanh thu -> d = by_day(kỳ) (doanh_thu, so_don); (d.doanh_thu >= X).sum() ('từ X trở lên' gồm X; 'trên/hơn X' không gồm X). "
                   "'củ'/'tr' = triệu, 'k'/'nghìn' = nghìn.")
    elif re.search(r"\b\d+([.,]\d+)? (cu|tr)\b", s):
        out.append("'củ'/'tr' = triệu đồng (10 củ = 10.000.000), 'k' = nghìn.")
    if re.search(r"\bdong khach\b", s):
        if re.search(r"\bgio\b", s):
            out.append("Đông khách nhất theo giờ = giờ cao điểm (§3) -> by_hour(kỳ) nhiều so_don nhất; đáp án giờ bắt đầu khung, type 'number'.")
        else:
            out.append("Đông khách nhất = ngày nhiều đơn nhất (§3) -> by_day(kỳ) nhiều so_don nhất (bằng nhau lấy ngày sớm hơn).")
    if re.search(r"\b(khong|chua) (ban|ban duoc|co ai mua)\b|\bkhong ban duoc\b|\bhang e\b|\be am\b|\bkhong co don\b", s) and item_ctx:
        out.append("Mặt hàng không bán được trong kỳ (§5) -> unsold_items(kỳ) (đã từng bán trước kỳ, trong kỳ không có đơn tính doanh thu nào).")
    if re.search(r"\b(ly do|vi sao|tai sao|nguyen nhan|vi gi|chu yeu vi)\b", s) and re.search(r"\b(huy|tra)\b", s):
        out.append("Lý do huỷ / trả -> cột lý do trong O_gom_huy_tra / S_gom_huy_tra (hoặc sổ trả hàng trong T): đếm theo lý do (value_counts) trên đơn huỷ/trả trong kỳ.")
    if re.search(r"\bkhach (nao|hang nao)\b.*\bnhieu (lan|don)\b|\bnhieu (lan|don) nhat\b", s) and re.search(r"\bkhach\b", s):
        out.append("Khách mua nhiều đơn nhất -> customers(kỳ).sort_values(['so_don', <cột đầu>], ascending=[False, True]) (bỏ khách lẻ, chỉ đơn tính doanh thu).")
    return out


def _superlative_hints(s: str) -> list[str]:
    """Dinh nghia 'nhat' theo rulebook §3 (khach trung thanh, sop, hay doi y, ua chuong, ngay dat khach, gio cao diem, ban gioi)."""
    out = []
    if re.search(r"\bkhach\b", s) and re.search(r"\b(nhieu tien nhat|chi nhieu|chi tieu nhieu|mua nhieu nhat|tien nhieu nhat|mua nhieu tien|doanh thu cao nhat)\b", s) \
            and not re.search(r"\bno\b|\bcong no\b|\b(so|nhieu) don\b|\bnhieu lan\b", s):
        k = re.search(r"\b(\d{1,2}|hai|ba|bon|nam) khach\b|\btop\s*(\d{1,2})\b", s)
        words = {"hai": 2, "ba": 3, "bon": 4, "nam": 5}
        kk = (words.get(k[1]) or int(k[1])) if k and k[1] else (int(k[2]) if k and k[2] else 1)
        if kk > 1:
            out.append(f"Khách mua nhiều tiền nhất (§3: theo doanh thu, bỏ khách lẻ; customers đã sắp doanh_thu giảm dần, bằng nhau theo mã) -> "
                       f"answer('list', customers(kỳ).head({kk}).khach.tolist()).")
        else:
            out.append("Khách mua nhiều tiền nhất (§3: theo doanh thu, bỏ khách lẻ; customers đã sắp doanh_thu giảm dần, bằng nhau theo mã) -> "
                       "answer('string', customers(kỳ).khach.iloc[0]).")
    if re.search(r"\btrung thanh\b", s):
        out.append("Khách trung thành nhất (§3) = nhiều NGÀY có mua nhất -> customers(kỳ).sort_values(['so_ngay_mua', <cột đầu mã khách>], ascending=[False, True]); trả lời cột khach.")
    if re.search(r"\bkhach (hang )?sop\b|\bsop nhat\b", s):
        out.append("Khách sộp nhất (§3) = giá trị TB một đơn cao nhất trong khách có từ 3 đơn -> c = customers(kỳ); c[c.so_don >= 3].sort_values(['gia_tri_tb_don', <cột đầu>], ascending=[False, True]).")
    if re.search(r"\bdoi y\b", s):
        out.append("Khách hay đổi ý nhất (§3) = nhiều đơn HUỶ nhất -> customers(kỳ, gom_huy=True).sort_values(['so_don_huy', <cột đầu>], ascending=[False, True]).")
    if re.search(r"\bua chuong\b|\bduoc ua\b", s):
        out.append("Mặt hàng được ưa chuộng nhất (§3) = có mặt trong NHIỀU ĐƠN nhất -> top_items(kỳ, k=1, by='orders').")
    if re.search(r"\bdat khach\b", s):
        out.append("Ngày đắt khách nhất (§3) = ngày nhiều ĐƠN nhất -> by_day(kỳ) (cột so_don; bằng nhau lấy ngày sớm hơn); type 'date' YYYY-MM-DD.")
    if re.search(r"\bcao diem\b", s):
        out.append("Giờ cao điểm (§3) = khung 1 giờ tròn có nhiều đơn nhất -> by_hour(kỳ) (cột so_don); đáp án là giờ bắt đầu khung, type 'number'.")
    if re.search(r"\b(ban gioi|gioi nhat|ban hay nhat)\b", s) and re.search(r"\bnhan vien\b|\bnv\b|\bai\b", s):
        out.append("Nhân viên bán giỏi nhất (§3) = DOANH THU cao nhất (không theo số đơn) -> O[between(O.order_date, *kỳ)].groupby('employee').revenue.sum() (O đã bỏ đơn huỷ/trả; có cột employee_name thì nhóm theo họ tên đó và trả lời họ tên).")
    if re.search(r"\b(cham chi|cham nhat|sieng|nhiet tinh|cham lam)\b", s):
        out.append("Chăm chỉ nhất (§3) = nhiều NGÀY làm việc nhất (ngày có ít nhất một đơn/phiếu của người đó, kể cả phiếu bảo hành).")
    return out


def _repair_hints(s: str) -> list[str]:
    """Goi y cho tiem sua xe (bang P, rulebook muc Sua xe)."""
    out = []
    hh = re.search(r"\b(hoa|hue|huy) hong\b", s)
    if hh:
        if re.search(r"\b(moi|tung|cac|tat ca|toan bo|het|ca) (cac )?(tho|nguoi|anh em)\b|\bbang\b|\bliet ke\b|\bxep\b", s):
            code = "answer('table', commission(kỳ)[['tho', 'hoa_hong']])"
        elif re.search(r"\b(tho nao|ai)\b|\bnhat\b", s):
            code = "answer('string', commission(kỳ).tho.iloc[0])"
        else:
            code = "answer('number', commission(kỳ).set_index('tho').hoa_hong[<tho>])"
        out.append("Hoa hồng thợ -> commission(kỳ): cột tho, tien_phieu, phan_tram, hoa_hong, ĐÃ sắp hoa_hong giảm dần (bảng mọi thợ: "
                   "commission(kỳ)[['tho', 'hoa_hong']]; một thợ: commission(kỳ).set_index('tho').hoa_hong[<họ tên>]; thợ cao nhất: "
                   f"commission(kỳ).tho.iloc[0]) -> {code}.")
    rate = re.search(r"\bti le bao hanh\b|\bty le bao hanh\b|\bbao hanh\b.*(%|phan tram|ti le|ty le)", s)
    avg = re.search(r"\b(trung binh|tb|binh quan)\b", s) and re.search(r"\b(phieu|luot|xe)\b", s) and re.search(r"\btien\b|\bgia tri\b|\bdoanh thu\b", s)
    if rate:
        out.append("Tỉ lệ bảo hành -> warranty_rate(kỳ) (% = phiếu bảo hành có ngày trong kỳ / phiếu có thu tiền trong kỳ, 2 chữ số).")
    elif re.search(r"\bbao hanh\b|\bbh\b", s) and not avg:
        out.append("Phiếu bảo hành (lượt xe quay lại bảo hành) -> n_tickets(kỳ, bao_hanh=True) (số phiếu bảo hành có ngày trong kỳ, tiền 0; danh sách: tickets(kỳ, bao_hanh=True)).")
    veh = re.search(r"\bkhac nhau\b|\bbien so\b|\bchiec\b|\bbao nhieu xe\b|\bmay xe\b|\bso xe\b", s) and not re.search(r"\bluot\b|\bphieu\b", s)
    money = re.search(r"\btien\b|\bdoanh thu\b|\bdthu\b|\bthu ve\b", s)
    count = re.search(r"\b(bao nhieu|bn|bnhieu|may|mi|so|dem|nhieu|nhieu nhat)\s+(luot|phieu|xe|chiec)\b|\b(luot|phieu) (xe|sua)\b.*\b(bao nhieu|bn|may|mi)\b", s)
    if re.search(r"\bkhach quen\b|(?<!bao hanh )\bquay lai\b(?! bao hanh)|\b(toi|den|vao|vo) (sua )?(tu )?\d+ lan\b|\btu \d+ lan\b|\b\d+ lan tro len\b|\bnhieu lan\b|\bxe nao\b", s) \
            and not re.search(r"\bquay lai bao hanh\b|\bxe quay lai bh\b", s):
        out.append("Xe khách quen / tới nhiều lần -> regular_vehicles(kỳ, min_paid=2) (theo biển số, chỉ đếm phiếu CÓ THU TIỀN); hỏi 'xe nào' trả lời biển số (cột plate).")
    elif veh:
        out.append("Số chiếc xe khác nhau -> n_vehicles(kỳ, dong_xe=...) (số biển số khác nhau, kể cả xe chỉ tới bảo hành).")
    elif not (hh or rate or avg) and not re.search(r"\bbao hanh\b|\bbh\b", s) and (count or re.search(r"\bluot\b|\bphieu\b|\bbao nhieu xe\b|\bmay xe\b", s) and not money):
        out.append("Số lượt xe = số phiếu -> n_tickets(kỳ, viec=..., tho=..., dong_xe=...) (kể cả phiếu bảo hành; lọc viec= thì phiếu bảo hành không tính là lượt làm việc đó).")
    if avg:
        out.append("Tiền TB một phiếu -> avg_ticket_value(kỳ, ...) (doanh thu / số phiếu CÓ THU TIỀN, làm tròn đồng).")
    elif not (hh or rate) and (money or re.search(r"\btong tien\b|\bthu (duoc|ve)\b", s)):
        out.append("Doanh thu (tiền) = tổng Tiền các phiếu -> revenue(kỳ, viec=..., tho=..., dong_xe=...) (một thợ: tho=<họ tên>).")
    if re.search(r"\b(cham chi|cham nhat|sieng|nhiet tinh|cham lam)\b", s):
        out.append("Thợ chăm chỉ nhất (§3) = nhiều NGÀY làm việc nhất (ngày có ít nhất một phiếu, kể cả phiếu bảo hành; bằng nhau theo tên) -> "
                   "answer('string', by_tho(kỳ).sort_values(['so_ngay_lam', 'tho'], ascending=[False, True]).tho.iloc[0]).")
    elif re.search(r"\b(tho|ai)\b", s) and re.search(r"\b(nhat|nao|mo|xep)\b", s) and not hh:
        if money and not count:
            out.append("Thợ làm ra nhiều tiền nhất (doanh_thu = tổng Tiền phiếu thợ làm) -> answer('string', by_tho(kỳ).sort_values(['doanh_thu', 'tho'], ascending=[False, True]).tho.iloc[0]).")
        else:
            out.append("Thợ làm nhiều nhất (rulebook) = nhiều LƯỢT XE nhất (so_phieu KỂ CẢ phiếu bảo hành; by_tho đã sắp so_phieu giảm dần, bằng nhau theo tên) -> answer('string', by_tho(kỳ).tho.iloc[0]).")
    if re.search(r"\bdong xe\b|\bloai xe\b", s) and re.search(r"\b(nhat|nao|xep|top)\b", s):
        by = "doanh_thu" if re.search(r"\btien\b|\bdoanh thu\b", s) else ("so_xe" if re.search(r"\bchiec\b|\bbien so\b|\bkhac nhau\b", s) else "so_phieu")
        out.append(f"Theo dòng xe -> by_dong_xe(kỳ) (doanh_thu, so_phieu, so_xe = biển số khác nhau); xếp theo {by}: "
                   f"by_dong_xe(kỳ).sort_values(['{by}', 'dong_xe'], ascending=[False, True]); trả lời tên dòng xe (cột dong_xe).")
    out += _weekday_avg_hint(s)
    if re.search(r"\b(moi|tung) (thang|tuan|quy|ngay)\b", s) or re.search(r"\blan luot\b|\btheo thu tu\b", s):
        out.append("Số đo theo từng kỳ / lần lượt -> answer('list', [số 1, số 2, …]) đúng thứ tự câu hỏi nêu (§9).")
    if re.search(r"\bngay nao\b", s):
        out.append("Hỏi 'ngày nào' -> by_day(kỳ) (doanh_thu, so_phieu); type 'date' dạng YYYY-MM-DD.")
    if re.search(r"\bthang nao\b", s):
        out.append('Hỏi "tháng nào" -> type "date", format "YYYY-MM" (vd answer("date", "2025-08")).')
    return out


def measure_hints(question: str, kinds: frozenset = frozenset()) -> list[str]:
    s = _norm(question)
    out = []
    if "repair" in kinds:
        return _repair_hints(s)
    if _QTY_RX.search(s) or re.search(r"\bso luong\b(?! (khach|don|nha cung cap|ncc|phieu|lo|mat hang|ma hang|san pham))", s):
        if re.search(r"\b(con|ton)\b", s) and not re.search(r"\bban\b", s):
            out.append("Câu hỏi đếm theo đơn vị hàng -> SỐ LƯỢNG tồn (stock_at / lot_stock_at), không phải tiền.")
        elif not re.search(r"\b(nhap|nha cung cap|ncc)\b", s):
            out.append("Câu hỏi đếm theo đơn vị hàng (đĩa, ly, hộp…) -> SỐ LƯỢNG bán: qty_sold(kỳ, <cột>=<giá trị>) (vd item=..., group=..., dong_may=...), không phải doanh thu.")
    low = question.lower()
    if re.search(r"lãi|lời|lợi nhuận", low) or re.search(r"\bloi (nhat|nhieu|nhuan|bao|gop)\b|(?<!con )(?<!ban )\blai (gop|nhat|nhieu|bao nhieu)\b|\btien lai\b", s):
        out.append("Lời/lãi = lãi gộp = doanh thu − giá vốn (§5): gross_profit(kỳ, ...) hoặc top_items(kỳ, by='profit').")
    if re.search(r"\b(bao nhieu|bn|may|so|tong|tong cong|tong so)\s+(don|bill|hoa don)\b(?! gia| vi)", s) \
            and not re.search(r"\bban chiu\b|\bmua chiu\b|\bghi no\b|\bcong no\b|\bcon no\b|\bngay (phai )?tra\b|\bhan (thanh toan|tra)\b", s):
        if re.search(r"\bhuy\b", s):
            out.append("Đơn huỷ -> n_cancelled(kỳ) / cancel_rate(kỳ) (tính trên mọi đơn đặt trong kỳ).")
        elif not re.search(r"\btra hang\b|\bbi tra\b|\bhoan tra\b", s):
            out.append("Đếm đơn (§1: chỉ đơn KHÔNG huỷ và KHÔNG trả hàng) -> n_orders(kỳ, ...) đã đúng luật; điều kiện khác thì lọc O. Không đếm trên O_gom_huy_tra.")
    if m := re.search(r"\bca (sang|chieu)\b", s):
        ca = "sáng" if m[1] == "sang" else "chiều"
        out.append(f"Ca {ca} (§2: theo giờ tạo đơn) -> bộ lọc ca='{ca}', vd n_orders(kỳ, ca='{ca}'), revenue(kỳ, ca='{ca}').")
    if re.search(r"\bkhach\b", s) and re.search(r"\b\d+\s*don\b|\bso don\b|\bnhieu don\b", s):
        out.append("Đếm đơn theo khách -> customers(kỳ) (cột so_don; đã bỏ khách lẻ và đơn huỷ/trả).")
    if _price_q(s) and re.search(r"\bngay\b", s):
        out.append("Đổi giá -> price_effect(ngày đổi giá, N, item=...); ngày đổi giá = ngày câu hỏi ghi, không ghi thì price_change_date(item, giá mới).")
    if re.search(r"\btuoi no\b", s) and len(re.findall(r"\d+\s*(?:-|–|den|toi)\s*\d+", s)) >= 1:
        e = [int(x) for x in re.findall(r"\d+\s*(?:-|–|den|toi)\s*(\d+)", s)]
        out.append(f"Nợ chia nhóm tuổi nợ -> answer('list', debt_aging(ngày, edges={tuple(e) if e else (7, 15)})) = [tiền nhóm 1, nhóm 2, …, nhóm cuối] theo thứ tự câu hỏi.")
    elif (m := re.search(r"\b(?:hon|qua|tren|tu) (\d+) ngay( tro len)?", s)) and re.search(r"\bno\b|chua tra|tuoi no", s):
        op = ">=" if m[2] else ">"
        out.append(f"Tuổi nợ -> d = debt_as_of(ngày); d[d.tuoi_no {op} {m[1]}] (cột con_no; §2: 'quá/hơn N' không gồm N, 'từ N trở lên' gồm N).")
    if re.search(r"\b(trung binh|binh quan|tb)\b[^.]*\b(moi|1|mot)\s+(don|bill|hoa don)\b|\bgia tri (trung binh|tb|binh quan) (cua )?(moi |mot |1 )?don\b", s):
        out.append("Giá trị TB một đơn -> avg_order_value(kỳ, ...) (doanh thu / số đơn tính doanh thu, làm tròn đồng; lọc vd customer_group=...).")
    if re.search(r"\b(chan|nguyen)\b", s) and re.search(r"\ble\b", s) and re.search(r"\b(ton|con|kho)\b", s):
        out.append("Tồn quy ra đơn vị lớn + phần lẻ -> answer('list', stock_units(ngày, item=..., unit='<đơn vị lớn>')) = [số nguyên, số lẻ].")
    if re.search(r"\b(moi|tung) (thang|tuan|quy|ngay|nam)\b", s) and re.search(r"\bbao nhieu\b", s) and not re.search(r"\b(trung binh|binh quan)\b", s):
        out.append("Số đo theo từng kỳ -> answer('list', [số kỳ 1, số kỳ 2, ...]) theo thứ tự thời gian câu hỏi nêu; không dùng table, không kèm tên tháng.")
    if re.search(r"\b(lai|loi) (rong|that)\b|\btru (het )?chi phi\b|\bloi nhuan rong\b", s):
        out = [x for x in out if not x.startswith("Lời/lãi")]
        out.append("Lãi ròng (rulebook) = lãi gộp − mọi khoản trong sổ chi phí cùng tháng -> net_profit(kỳ) (= gross_profit(kỳ) − expenses(kỳ)).")
    if re.search(r"\bvoucher\b|\bvau cho\b|\bdung (\d+ )?diem\b|\bdiem (thanh vien|tich luy)\b|\bphi ship\b|\bphi giao\b|\btien ship\b", s):
        kinds_q = [k for k, rx in (("voucher", r"\bvoucher\b|\bvau cho\b"), ("diem", r"\bdiem\b"), ("ship", r"\bship\b|\bphi giao\b")) if re.search(rx, s)]
        out.append("Tiền giảm voucher / dùng điểm / phí ship -> adjustments(kỳ, kind='voucher'|'diem'|'ship') (số dương, theo ngày hoá đơn, chỉ hoá đơn không huỷ, không trả).")
        if len(kinds_q) == 1:
            out.append(f"Khoản {kinds_q[0]} -> answer('number', adjustments(kỳ, kind='{kinds_q[0]}')).")
    if re.search(r"\bthanh vien\b.*\b(moi|dang ky)\b|\bdang ky moi\b", s):
        out.append("Khách thành viên mới trong tháng = số khách có Ngày đăng ký trong tháng đó (bảng khách hàng trong T), kể cả khách chưa mua.")
    if re.search(r"\bgia nhap\b|\bnhap (vao )?(voi )?gia\b", s) and re.search(r"\b(binh quan|trung binh|tb)\b", s):
        out.append("Giá nhập bình quân (§5: theo số lượng nhập, theo đơn vị nhập, bỏ phiếu huỷ) -> avg_import_price(kỳ, item=..., unit='<đơn vị>').")
    if re.search(r"\b(trung binh|binh quan|tb)\b", s) and re.search(r"\b(tien|gia|doanh thu|dong|vnd|thu ve|ban duoc)\b", s) \
            and not re.search(r"%|phan tram|\bti le\b|\bty le\b|\bti trong\b", s):
        out.append("Trung bình của một khoản tiền cũng là tiền (§9): làm tròn về đồng -> answer('number', round_money(x)).")
    out += _weekday_avg_hint(s)
    out += _superlative_hints(s)
    out += [x for x in _general_hints(s) if not any(x[:25] == y[:25] for y in out)]
    if not out and re.search(r"\bdoanh thu\b|\btien hang\b|\btien ban\b|\bban (duoc|ra)?\s*(tong cong |tong )?(bao nhieu|bn) tien\b|\bban duoc\b.*\btien\b", s) \
            and not re.search(r"%|phan tram|\bnhat\b|\bnao\b|\bmoi\b|\btung\b|\bso voi\b|\bhon\b|\bkem\b", s):
        out.append("Doanh thu (§1: đơn không huỷ, không trả; tiền dòng hàng, không gồm phí ship/voucher) -> revenue(kỳ, <lọc>).")
    if re.search(r"\b(tra gop|gop|ky gop)\b", s) and re.search(r"\b(qua han|toi han|chua gop|con lai|con thieu|cham gop|chua tra|chua dong)\b", s):
        out.append("Trả góp -> d = installments_as_of(ngày): so_ky_toi_han (kể cả kỳ hạn đúng ngày đó), con_lai (gồm kỳ chưa tới hạn), tien_qua_han / qua_han (kỳ hạn TRƯỚC ngày đó chưa góp đủ); số khách = d.loc[d.qua_han, 'customer_key'].nunique().")
        if re.search(r"\b(bao nhieu|may|so|bn) (khach|nguoi)\b", s) and re.search(r"\bqua han\b|\bcham\b", s):
            out.append("Số khách quá hạn góp -> answer('number', installments_as_of(ngày).loc[lambda d: d.qua_han, 'customer_key'].nunique()).")
    if re.search(r"\bmay cu\b", s) and re.search(r"\b(thu|tru|tien)\b", s) and not re.search(r"\bkhach (phai )?tra\b", s) and not re.search(r"\bban (duoc|ra)\b.*\bmay cu\b|\bmay cu\b.*\bban (duoc|ra)\b", s):
        out.append("Thu máy cũ -> trade_in_total(kỳ) (tổng giá thu trừ vào đơn không huỷ, không trả); đơn có thu máy cũ: PO.thu_may_cu > 0.")
    if re.search(r"\bkhach (phai )?tra\b", s) and re.search(r"\b(bot|may cu|sau khi tru|tru tien)\b", s):
        out.append("Khách phải trả (rulebook điện thoại) = tiền đơn sau bớt và sau trừ máy cũ, gồm cả phần còn nợ -> customer_pays(kỳ); trung bình một đơn = customer_pays(kỳ) / n_orders(kỳ).")
    top = re.search(r"\btop\s*(\d+)|\b(\d+) (dong may|hang)\b|\b(nhieu nhat|chay nhat|cao nhat)\b", s)
    k = next((int(g) for g in (top.groups()[:2] if top else ()) if g), 1)
    if re.search(r"\bdong may\b", s):
        if top:
            out.append(f"Xếp hạng dòng máy theo số máy bán -> answer('list', top_items(kỳ, k={k}, key='dong_may')['dong_may'].tolist()) "
                       "(cộng số lượng các dòng là máy, phụ kiện không có dong_may).")
        else:
            out.append("Dòng máy -> cột S.dong_may (phần tên trước dung lượng …GB; chỉ dòng là máy); số máy = qty_sold(kỳ, dong_may=...).")
    if re.search(r"\bhang (nao|apple|samsung|xiaomi|oppo|vivo|realme|nokia)\b|\bmay (samsung|apple|xiaomi|oppo|iphone)\b", s) and re.search(r"\bmay\b", s):
        out.append("Hãng máy -> cột S.hang (chữ đầu tên hàng; iPhone là Apple; chỉ dòng là máy)"
                   + (f"; xếp hạng: top_items(kỳ, k={k}, key='hang')." if top else "."))
    if re.search(r"\bdot\b", s) and re.search(r"\bgiao\b|\bxe cho\b|\bcho (duoc|dc)\b", s) or re.search(r"\bda giao\b|\bxe (da )?giao\b|\bchua giao\b", s):
        if re.search(r"\b(bao nhieu|may|so|tong|bn)\s+(dot|chuyen|luot)\b", s) and not re.search(r"\b(don|ma don|ord)\s*[a-z]*\d", s):
            out.append("Đếm đợt giao trong kỳ -> answer('number', delivery_rounds(kỳ)) (số cặp (đơn, đợt) có ngày giao trong kỳ, tính đến hết hôm nay, kể cả đơn sau đó trả hàng; KHÔNG đếm số dòng sổ giao).")
        else:
            out.append("Giao hàng nhiều đợt -> bảng D_giao (order_id, dot, ngay_giao, qty_giao, order_date, status; mỗi dòng một mặt hàng của một đợt); "
                       "delivery_rounds(kỳ) = số đợt (đơn, đợt) có ngày giao trong kỳ, tính đến hết hôm nay, kể cả đơn sau đó trả hàng.")
    if re.search(r"\bbao hanh\b", s):
        out.append("Bảo hành -> tiệm sửa xe: bảng P (bao_hanh), warranty_rate(kỳ) = phiếu bảo hành / phiếu có thu tiền; cửa hàng điện thoại: bảng W_bh, "
                   "phone_warranty_rate(kỳ) = lượt bảo hành / số máy bán (theo hồ sơ cửa hàng có phần nào).")
    if re.search(r"\blan luot\b|\btheo thu tu\b", s):
        out.append("Câu hỏi 'lần lượt / theo thứ tự' -> answer('list', [số 1, số 2, …]) đúng thứ tự câu hỏi nêu (§9), không dùng table.")
    if re.search(r"\bthang nao\b", s):
        out.append('Hỏi "tháng nào" -> type "date", format "YYYY-MM" (vd answer("date", "2025-08")).')
    return out


# cum thanh toan ("tra bang ...", "tra tien mat", "vi dien tu"): khong phai tra hang / ly do ("vi" khong dau trung "vì")
_PAY_PHRASE = re.compile(r"\btra (bang|qua) |\btra (tien mat|tien tuoi|tm|chuyen khoan|ck|cod|the|vi)\b|\bvi (dien tu|momo|zalo ?pay|vnpay|shopee ?pay)\b")
_PRICE_RX = re.compile(r"\b(tang|len|doi|giam|ha|xuong)\s+(gia\s+)?(len\s+|xuong\s+|thanh\s+|con\s+)?\d[\d.,]*\s*(k|d|dong|vnd|nghin|ngan)?\b")


def _price_q(s: str) -> bool:
    """Cau hoi hieu ung doi gia: 'tang/len gia ...' hoac 'len 26k tu 13/11, so 30 ngay ...'."""
    return bool(re.search(r"\b(tang|len|doi|giam|ha) gia\b", s) or (_PRICE_RX.search(s) and re.search(r"\b\d{1,3} ngay\b", s)))


def _money(num: str, unit: str | None) -> int | None:
    """'8.900.000' -> 8900000; '26' + 'k' -> 26000; '10' + 'cu' -> 10000000; '8,5' + 'tr' -> 8500000."""
    unit = unit or ""
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", num):
        v = float(re.sub(r"[.,]", "", num))
    else:
        try:
            v = float(num.replace(",", "."))
        except ValueError:
            return None
    if unit in ("k", "nghin", "ngan"):
        v *= 1000
    elif unit in ("tr", "trieu", "cu"):
        v *= 1_000_000
    return int(round(v))


def _new_price(s: str) -> int | None:
    m = re.search(r"\b(?:len|thanh|con|xuong)\s+(?:gia\s+)?(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(k|d|dong|vnd|nghin|ngan)?\b", s)
    return _money(m[1], m[2]) if m else None


def _expected_type(s: str) -> str | None:
    """Kieu dap an cau hoi doi (nhan dien chac chan) -> dung de bo code goi y khong khop (vd hoi 'ngay nao' ma code ra mot so)."""
    # '%' khong dung ngay sau chu so ('kem 5% tuyp', 'Otrivin 0,1%' la ten hang): 'bao nhieu %', 'nhieu % mua', 'may %'
    if re.search(r"\bphan tram\b|\b(ti|ty) (le|trong)\b|(?:^|[^\d.,\s])\s*%", s):
        return "percent"
    if re.search(r"\b(ngay|hom|bua) nao\b|\bngay may\b|\bthang nao\b|\bkhi nao\b|\bluc nao\b|\bhet han ngay\b", s):
        return "date"
    if re.search(r"\b(mat hang|mon|sp|san pham|thuoc|dong xe|dong may|loai xe|tho|nhan vien|nv|khach|khach hang|nha cung cap|ncc|kenh|ben|hoat chat|xe|lo|size|vi|mau|hang)\s+nao\b"
                 r"|(?<!khong )(?<!chang )(?<!chua )(?<!ko )\bai\b|\bla gi\b|\bten (gi|la gi)\b", s):
        return "name"
    return None


_TOTAL_FNS = ("revenue", "n_tickets", "qty_sold", "n_orders", "customer_pays", "trade_in_total", "import_cost", "n_vehicles", "gross_profit",
              "net_profit", "debt_paid", "debt_added", "adjustments", "delivery_rounds", "n_cancelled")


def _guard_code(code: str | None, s: str, sf: list) -> str | None:
    """Bo (hoac sua) code goi y khi khong khop dang cau hoi: kieu dap an, trung binh, theo thu, 'nhat', nhom hang, ti trong theo so luong.
    Do tren bo public: cac code goi y sai deu la tong (revenue/n_tickets/...) cho cau hoi 'ngay nao', '%', 'trung binh moi thu', 'nhom ...'."""
    if not code or code.startswith("answer('unanswerable'"):
        return code
    ct = re.match(r"answer\('(\w+)'", code)
    ct = ct[1] if ct else None
    exp = _expected_type(s)
    if exp == "percent" and ct != "percent" or exp == "date" and ct != "date" or exp == "name" and ct not in ("string", "list", "table"):
        return None
    total = re.match(r"answer\('number', (?:round_money\()?(" + "|".join(_TOTAL_FNS) + r")\(", code)
    if re.search(r"\b(trung binh|binh quan|tb)\b", s) and ct == "number" and not re.search(r"avg_|/|mean\(|count_weekday|gia_tri_tb_don", code):
        return None
    if re.search(r"\b(moi|cac|1) (ngay )?thu (hai|ba|tu|nam|sau|bay|[2-7])\b|\b(moi|cac) (ngay )?chu nhat\b", s) and "count_weekday" not in code:
        return None
    if total and re.search(r"\bnhat\b", re.sub(r"\bit nhat\s+\d+|\b(lien |moi )?gan (day )?nhat\b|\bmoi nhat\b", "", s)):
        return None
    if exp == "percent" and "revenue(" in code and re.search(r"\b(chiem|ti trong|ty trong|dong gop)\b|\btong so\b", s) \
            and (_QTY_RX.search(s) or re.search(r"\btong so (" + _UNITS + r")\b", s)):
        return None  # ti trong theo so luong (so bich, so hop) khong phai theo doanh thu
    item_fn = re.search(r"\b(top_items|qty_sold|revenue|n_orders|avg_order_value|gross_profit)\(", code)
    if item_fn and re.search(r"\bnhom\b(?! (tuoi|no|khach))", s) and not re.search(r"\b(group|customer_group|dm_\w+)=", code):
        groups = [v for c, v in sf if c == "group"]
        m = re.search(r"\b(top_items|qty_sold|revenue|n_orders|avg_order_value|gross_profit)\(", code)
        seg = _balanced(code, m.end() - 1) if m and len(groups) == 1 else None
        if not seg:
            return None
        end = m.end() - 1 + len(seg) - 1
        code = code[:end] + f", group={groups[0]!r}" + code[end:]
    return code


_FILLER = set("""doanh thu tong cong bao nhieu bn la cua quan shop tiem hang ban duoc ra ve tien co trong ky thang nam ngay hom tuan quy nay qua truoc roi
z vay a nhi nhe ha oi cho minh em anh chi hoi xiu nhanh ak ah nha the tinh den toi het tu va voi trung binh binh moi mot don bill hoa gia tri
gi gom tat ca duoc dc thu dat kiem may tat hon kem so sanh lieu biet muon mun bit xem giup ban rieng dip sop luon nhieu hi ne nghe lai gop loi nhuan
mang tren gium dum check coi ta total da dem do e hen nhen""".split())


_REV_WORDS = set("mat hang san pham sp mon thu ve nhom cac hoat chat thuoc tu co loai ke don khong nha k".split())


def _name_tokens(sf: list) -> set:
    """Cac tu (khong dau) cua ten da khop lam bo loc."""
    return {t for _, v in sf for t in re.findall(r"[a-z]+", _norm(str(v)))}


def _plain(s: str, ps: list, extra: set = frozenset()) -> bool:
    """Cau hoi chi co so do + ky (khong co dieu kien loc nao ngoai ky): bo phan ky va tu thong dung thi khong con tu nao.
    extra: tu duoc phep them (vd cac tu cua ten mat hang da khop lam bo loc)."""
    x = re.sub(r"\bkhach (hang )?mua\b", " ", s)  # "khach mua trung binh moi don" khong phai bo loc khach
    for p in ps:
        x = x.replace(_norm(p.text), " ")
    x = re.sub(r"\b\d+([/\-.]\d+)*\b", " ", x)
    return all(t in _FILLER or t in extra for t in re.findall(r"[a-z]+", x))


_WEEKDAY = re.compile(r"(?<!theo )(?<!doanh )\bthu (hai|ba|tu|nam|sau|bay|[2-7])\b(?! (thoi gian|cau hoi|do|tren|giam|tang|uu tien|hang|nhu))"
                      r"|\bchu nhat\b|\bcuoi tuan\b|\bngay thuong\b|\bngay le\b")


def _unexplained(s: str, ps: list, code: str, names=()) -> str | None:
    """Dieu kien co trong cau hoi ma dong code khong the hien (thu trong tuan, ngay le trong thang, dau / cuoi ky, phu dinh,
    gio trong ngay, gioi tinh, kenh ban, loai khach, ca, qua han) -> ten dieu kien (code goi y bi bo, de model tu lam)."""
    if code.startswith("answer('unanswerable'"):
        return None
    x = s
    for p in ps:
        x = x.replace(_norm(p.text), " ")
    for v in names:  # ten trong du lieu (nhom "Dung ngoai") khong phai dieu kien
        x = x.replace(_norm(str(v)), " ")
    if _WEEKDAY.search(x) and not re.search(r"count_weekday|weekday|dayofweek", code):
        return "thu trong tuan"
    if re.search(r"\b(ngay|hom|mung|mong|bua) \d{1,2}\b(?!\s*[/\-.]\d)", x):
        return "ngay trong thang chua thanh ky"
    if "price_effect" not in code and re.search(r"\b(dau|giua|cuoi|nua dau|nua cuoi|nua) (thang|tuan|nam|quy)\b|\b\d+ (ngay|tuan|thang) (dau|cuoi)\b|\btuan (dau|cuoi|thu \d)\b", x):
        return "mot phan cua ky"
    if re.search(r"\b(ngoai|loai tru|loai bo|bo qua)\b|\btru\b(?! (so )?ban)|\bkhong (tinh|ke|gom|bao gom|lay|dem|co)\b|\bchua tinh\b", x) \
            and re.search(r"\w+='", code) and "net_import" not in code:
        return "phu dinh"
    if re.search(r"\bbuoi (sang|trua|chieu|toi|dem)\b|\b\d{1,2} ?(h|gio)\b|\b(truoc|sau) \d{1,2}\b|\bkhung gio\b", x) \
            and not re.search(r"\bca=|by_hour", code):
        return "gio trong ngay"
    if re.search(r"\bkhach (hang )?(nam|nu)\b|\b(nam|nu) gioi\b|\bgioi tinh\b", x) and "gender" not in code:
        return "gioi tinh"
    if re.search(r"\b(online|offline|shopee|lazada|tiki|grab|tiktok|facebook|fb|zalo|giao hang|ship|tai cho|tai quay|qua dien thoai)\b", x) \
            and not re.search(r"\bchannel=|\bpayment=|adjustments\(|delivery_rounds", code):
        return "kenh ban"
    if re.search(r"\bkhach (le|si|quen|moi|vip|vang lai)\b", x) and "customer" not in code and "customers(" not in code:
        return "loai khach"
    if re.search(r"\bca (sang|chieu|toi|trua|dem)\b", x) and "ca=" not in code:
        return "ca"
    if re.search(r"\b(qua han|den han|tre han|han tra|han thanh toan|phai tra)\b", x) and "debt_as_of" in code:
        return "qua han"
    return None


def _future(p: "PeriodHint") -> bool:
    """Ky 'N ngay toi' (tu hom nay den het hom nay + N)."""
    return "hôm nay +" in (p.note or "")


def _lot_code(s: str, ps: list, items: list) -> str | None:
    """Cau hoi lo / han dung (rulebook nha thuoc) -> mot dong code tren expiring_lots / expiring_items / nearest_expiry."""
    days = [p for p in ps if p.code.startswith("NGAY(")]
    D = "HOM_NAY"
    m = re.search(r"\b(?:tinh tai|tinh den(?: het)?|theo ton kho|tai|tinh den het) ngay (\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", s)
    if m:
        D = f"NGAY({int(m[1])}, {int(m[2])}, {int(m[3])})"
        days = [p for p in days if p.code != D] or days
    nxt = next((p for p in ps if _future(p)), None)
    if nxt is not None:
        W = nxt.code
    elif len(days) >= 2:
        W = f"({days[0].code}[0], {days[-1].code}[1])"
    elif len(days) == 1 and re.search(r"\btu (nay|hom nay|bay gio|gio)\b|\bden het\b|\btoi het\b", s):
        W = f"(HOM_NAY[0], {days[0].code}[1])" if D == "HOM_NAY" else f"({D}[0], {days[0].code}[1])"
    else:
        W = None
    if W is None:
        if re.search(r"\b(gan|sap|can|som)\b", s) and re.search(r"\bnhat\b", s) and len(items) == 1:
            return f"answer('date', nearest_expiry({D}, item={items[0]!r}))"
        return None
    if re.search(r"\bvon\b", s):
        return f"answer('number', round_money(expiring_lots({D}, {W}).von.sum()))"
    if re.search(r"\b(bao nhieu|may|so|bn|mi)\s+lo\b", s):
        return f"answer('number', len(expiring_lots({D}, {W})))"
    if re.search(r"\b(bao nhieu|may|so|bn|mi)\s+(mat hang|sp|san pham|loai|thuoc|mon)\b", s):
        return f"answer('number', len(expiring_items({D}, {W})))"
    if re.search(r"\b(nhung|cac|hang|sp|san pham|mat hang|thuoc)\s+nao\b|\b(ke|liet ke|danh sach)\b", s):
        return f"answer('list', expiring_items({D}, {W}))"
    return None


def _pair_code(s: str, ps: list, sf: list) -> str | None:
    """Cau hoi hai so do cung loai: 'doanh thu don tra tien mat voi don tra vi dien tu, tung ben / lan luot' (hai gia tri cung mot cot,
    mot ky) -> answer('list', [doanh thu gia tri 1, doanh thu gia tri 2]) theo thu tu cau hoi (§9)."""
    if len(ps) != 1 or len(sf) != 2 or sf[0][0] != sf[1][0] or sf[0][1] == sf[1][1]:
        return None
    if not re.search(r"\bdoanh thu\b|\btien ban\b|\bban duoc\b.*\btien\b", s) or not re.search(r",|\bva\b|\bvoi\b", s):
        return None
    if re.search(r"\b(so|bao nhieu|may|bn) don\b|%|phan tram|\bso luong\b|\bchenh\b|\bhon\b|\bkem\b|\bnhat\b", s):
        return None
    each = re.search(r"\b(lan luot|moi (loai|ben|cai|kenh|hinh thuc|nhom)|tung (loai|ben|cai|kenh|hinh thuc|nhom)|rieng (tung|moi)|theo thu tu)\b", s) \
        or len(re.findall(r"\bbao nhieu\b", s)) >= 2 or len(re.findall(r"\bdoanh thu\b", s)) >= 2
    if not each or re.search(r"\btong\b|\bcong (ca|lai|chung)\b|\bgop\b", s):
        return None  # "tong doanh thu tien mat va chuyen khoan": mot so, khong phai list
    K, c = ps[0].code, sf[0][0]
    return f"answer('list', [revenue({K}, {c}={sf[0][1]!r}), revenue({K}, {c}={sf[1][1]!r})])"


_PACK = "thung|loc|ket|hop|vi|bao|xe|cuon|bich|tui|day|can|chuc|kien"
_PACK_RX = re.compile(r"\b(?:bao nhieu|may|bn|nhieu|so)\s+(" + _PACK + r")\b")


def _diaper_code(s: str, K: str, sf: list, index) -> str | None:
    """Rulebook me va be: size / hang bim lay theo cot So / Size, cot Hang cua danh muc; phan tram cua mot size hay mot hang
    chia cho tong so BICH moi hang bim ta cung ky. Cung dang: so bich ban ra cua mot size / hang."""
    G = next((v for c, v, _ in index.entries if c == "group" and re.search(r"\bbim\b", _norm(str(v)))), None)
    sub = [(c, v) for c, v in sf if c.startswith("dm_") and (re.search(r"size|(^|_)so(_|$)", c) or c.endswith("_hang"))]
    if G is None or len(sub) != 1 or len(sf) > 2 or any(c not in ("group",) and (c, v) not in sub for c, v in sf):
        return None
    c, v = sub[0]
    num = f"qty_unit({K}, unit='bịch', group={G!r}, {c}={v!r})"
    den = f"qty_unit({K}, unit='bịch', group={G!r})"
    if re.search(r"%|phan tram|\b(ti|ty) (le|trong)\b|\bchiem\b", s) and not re.search(r"\b(tang|giam|so voi|chenh|doanh thu|tien)\b", s):
        return f"answer('percent', pct({num}, {den}))"
    if re.search(r"\b(bao nhieu|may|bn|so) bich\b", s) and not re.search(r"%|phan tram|\b(doanh thu|tien|trung binh|moi)\b", s):
        return f"(lambda n: answer('number', n) if {den} else None)({num})"
    return None


def _accented(orig: str, word: str) -> str:
    """Tu trong cau hoi goc (con dau) ung voi tu khong dau 'word'; khong thay thi tra lai 'word'."""
    for w in re.findall(r"\w+", orig.lower()):
        if strip_accents(w).lower() == word:
            return w
    return word


def _special_code(s: str, ps: list, sf: list, mh: list, kinds: frozenset, orig: str = "") -> str | None:
    """Code goi y cho cac dang can ham rieng (doi gia, lo, ngay/thang nhieu tien nhat, tang lien tuc, so mat hang / khach tu N don,
    so ngay dat nguong doanh thu, gia tri TB mot don). Chi tra code khi moi thanh phan deu nhan dien chac chan."""
    items = [v for c, v in sf if c == "item"]
    filt = [c for c, v in sf]
    if any(x.startswith("Đổi giá") for x in mh):
        n = re.search(r"\b(\d{1,3}) ngay\b", s)
        if not (re.search(r"%|phan tram", s) and n and len(items) == 1):
            return None
        day = next((f"'{p.a.isoformat()}'" for p in ps if "kể cả ngày" in (p.note or "")), None) or next((p.code for p in ps if p.code.startswith("NGAY(")), None)
        if day is None:
            price = _new_price(s)
            if price is None:
                return None
            day = f"price_change_date({items[0]!r}, {price})"
        return f"answer('percent', price_effect({day}, {int(n[1])}, item={items[0]!r}))"
    if re.search(r"\b(het han|hsd|han dung|han su dung|can date)\b", s):
        return _lot_code(s, ps, items)
    one = len(ps) == 1
    K = ps[0].code if one else None
    s_nm = s  # cau hoi bo ten da khop (don vi trong ten hang "hop 20 goi" khong phai don vi cau hoi)
    for _, v in sf:
        s_nm = s_nm.replace(_norm(str(v)), " ")
    # ton kho tinh theo don vi lon chan + phan le (don vi co ban): "con may hop chan, may vien le", "may xe 4m3 chan, may m3 le"
    cl = re.search(r"\b(?:may|bao nhieu|bn|nhieu|duoc)\s+([a-z]+(?: [a-z0-9]*\d[a-z0-9]*)?)\s+chan\b", s)
    if one and len(items) == 1 and (K == "HOM_NAY" or K.startswith("NGAY(")) and cl and re.search(r"\ble\b", s[cl.end():]) \
            and not re.search(r"\b(ban|nhap|doanh thu|tien|gia)\b", s):
        unit = " ".join(_accented(orig, w) for w in cl[1].split())
        return f"answer('list', stock_units({K}, item={items[0]!r}, unit={unit!r}))"
    # ton kho cua MOT mat hang luc het mot ngay (don vi co ban): "ton kho X cuoi ngay hom nay", "trong kho con may chai X"
    if one and len(items) == 1 and (K == "HOM_NAY" or K.startswith("NGAY(")) \
            and (re.search(r"\bton\b|\btrong kho\b", s) or re.search(r"\b(cuoi|het) ngay\b|\bhet hom nay\b", s) and re.search(r"\bcon (lai )?(bao nhieu|may|bn)\b", s)) \
            and not re.search(r"\b(het han|doanh thu|ban ra|ban duoc|gia von|von|tien|gia tri|chan|le|nguyen|nhap)\b"
                              r"|\blo (hang|so|nao)\b|\b(tung|cac|moi) lo\b|\btheo lo\b", s):
        su = _PACK_RX.search(s_nm)
        code_ = f"answer('number', stock_at({K}).loc[lambda t: t.item == {items[0]!r}, 'ton'].sum())"
        return f"{code_} if unit_is_base({items[0]!r}, {_accented(orig, su[1])!r}) is not False else None" if su else code_
    # loc theo ca (§2: theo gio tao don): "ca chieu ban duoc bao nhieu tien", "so don ca chieu chenh ca sang bao nhieu %"
    shifts = list(dict.fromkeys(re.findall(r"\bca (sang|chieu|toi|trua|dem)\b", s)))
    if one and shifts and not filt and not re.search(r"\b(trung binh|binh quan|moi|nhat|nao|mat hang|mon|khach)\b", s):
        cnt = re.search(r"\b(so|bao nhieu|may|bn|mi) (don|hoa don|bill)\b|\bso don\b|\bdon hang\b", s)
        cash = re.search(r"\bdoanh thu\b|\btien\b", s)
        fn = "n_orders" if cnt and not cash else "revenue" if cash and not cnt else None
        val = {"sang": "sáng", "chieu": "chiều", "toi": "tối", "trua": "trưa", "dem": "đêm"}
        if fn and len(shifts) == 1 and not re.search(r"%|phan tram|\b(chenh|hon|kem|so voi)\b", s):
            return f"answer('number', {fn}({K}, ca={val[shifts[0]]!r}))"
        if fn and len(shifts) == 2 and re.search(r"%|phan tram", s):
            a, b = (val[x] for x in shifts)
            base = re.search(r"\b(?:so voi|so vs|voi) ca (sang|chieu|toi|trua|dem)\b", s)
            if base and val[base[1]] == a:  # "so voi ca sang, ca chieu chenh bao nhieu %": ca sau "so voi" la ky so
                a, b = b, a
            return f"answer('percent', pct_change({fn}({K}, ca={a!r}), {fn}({K}, ca={b!r})))"
    best = re.search(r"\b(cao nhat|nhieu nhat|lon nhat|nhieu tien nhat|dat nhat)\b", s)
    money = re.search(r"\b(doanh thu|nhieu tien|ban duoc nhieu|thu nhieu|tien ban|tien)\b", s)
    if one and not filt and best and money and re.search(r"\b(ngay|hom|bua) nao\b", s) and not re.search(r"\b(don|khach|ly|hop|chai|luot|phieu|xe)\b", s):
        dc = "date" if "repair" in kinds else "order_date"  # so sua xe: by_day theo cot date
        return f"answer('date', by_day({K}).sort_values(['doanh_thu', '{dc}'], ascending=[False, True]).{dc}.iloc[0])"
    if "repair" in kinds:
        return None
    if one and not filt and re.search(r"%|phan tram|\b(ti|ty) le\b", s) and re.search(r"\bdon\b.*\bhuy\b", s) \
            and not re.search(r"\b(tra|doi|khach|nhat|nao|mat hang|mon|doanh thu|tien|ca)\b", s):
        return f"answer('percent', cancel_rate({K}))"
    # gia ban mot don vi cua MOT mat hang vao mot ngay (§: don gia dong ban, truoc chiet khau)
    if one and K.startswith("NGAY(") and len(items) == 1 and len(filt) == 1 and re.search(r"\bgia( ban)?\b|\bdon gia\b", s) \
            and not re.search(r"\b(nhap|von|doi|tang|giam|trung binh|binh quan|doanh thu|tong)\b|%|phan tram", s_nm):
        pu = re.search(r"\b(?:mot|1|moi)\s+(" + _PACK + r")\b", s_nm)
        if pu:
            u_ = _accented(orig, pu[1])
            return f"answer('number', price_on({K}, item={items[0]!r})) if unit_is_base({items[0]!r}, {u_!r}) is not False else None"
        return f"answer('number', price_on({K}, item={items[0]!r}))"
    # doanh thu cua MOT mat hang / nhom / hoat chat trong mot ky: "thang 12 ban X duoc bao nhieu tien", "nhom lau thu duoc bao nhieu"
    if one and len(sf) == 1 and (sf[0][0] in ("item", "group", "dm_hoat_chat") or isinstance(sf[0][1], bool)) and re.search(r"\b(doanh thu|tien)\b", s) \
            and not re.search(r"\b(phi|ship|van chuyen|giao hang|voucher|giam|diem|thu may cu)\b", s) \
            and _plain(s, ps, extra=_name_tokens(sf) | _REV_WORDS):
        return f"answer('number', revenue({K}, {sf[0][0]}={sf[0][1]!r}))"
    # top khach theo tien mua (customers() da xep doanh_thu giam dan, bang nhau theo ma)
    if one and not filt and re.search(r"\b(khach|ai)\b", s) and re.search(r"\b(tien|chi tieu|chi)\b", s) \
            and re.search(r"\b(cao nhat|nhieu nhat|nhieu tien nhat|lon nhat)\b", s) and not re.search(r"\b(don|lan|no|nhom|%|phan tram|trung binh)\b", s):
        k = re.search(r"\b(\d{1,2}) (khach|nguoi)\b", s)
        if k:
            return f"answer('list', customers({K}).khach.head({int(k[1])}).tolist())"
        if not re.search(r"\b(nhung|cac|liet ke|top)\b", s):
            return f"answer('string', customers({K}).khach.iloc[0])"
    # mat hang e / khong ban duoc trong ky (§5): dem hoac liet ke
    unsold = any(x.startswith("Mặt hàng không bán được") for x in mh) or re.search(
        r"\b(mon|mat hang|sp|san pham|hang|thuoc) (bi )?e\b(?=\s*(?:$|,|nhat|nao|khong|trong|cua|thang|tuan|quy|nam|hom|ngay|tu|giai|ca|vay|z|ha|nhi))", s)
    days2 = [p for p in ps if p.code.startswith("NGAY(")]
    W = K if one else (f"({min(days2, key=lambda p: p.a).code}[0], {max(days2, key=lambda p: p.a).code}[1])"
                       if len(ps) == 2 and len(days2) == 2 and re.search(r"\b(tu|giai doan)\b", s) else None)
    if unsold and W and not filt:
        if re.search(r"\b(bao nhieu|may|so|bn|mi|dem)\s+(mat hang|mon|sp|san pham|loai|thuoc)\b|\bmay mon\b", s):
            return f"answer('number', len(unsold_items({W})))"
        if re.search(r"\b(nhung|cac|liet ke|ke ten|danh sach)\b|\bnao\b", s):
            return f"answer('list', unsold_items({W}))"
    # doanh thu nhieu thang, liet ke / lan luot (khong loc): list theo thu tu cau hoi (hoac thoi gian)
    mlist = [p for p in ps if p.code.startswith("THANG(")]
    if len(mlist) >= 2 and len(mlist) == len([p for p in ps if not p.code.startswith("(")]) and not filt and re.search(r"\bdoanh thu\b", s) \
            and re.search(r"\b(lan luot|theo thu tu|liet ke|tung thang|moi thang|tu thang cu)\b", s) and not re.search(r"%|phan tram|\b(chenh|hon|kem|tang|giam|nhat|nao|trung binh)\b", s):
        pos = [s.find(_norm(p.text)) for p in mlist]
        order = sorted(mlist, key=lambda p: p.a)
        if not re.search(r"\bthoi gian\b", s) and all(x >= 0 for x in pos):
            order = [p for _, p in sorted(zip(pos, mlist), key=lambda t: t[0])]
        return f"answer('list', [{', '.join(f'revenue({p.code})' for p in order)}])"
    # ti trong doanh thu cua MOT bo loc tren doanh thu ca shop cung ky (§1)
    if one and len(sf) == 1 and re.search(r"%|phan tram", s) and re.search(r"\b(doanh thu|tien)\b", s) \
            and re.search(r"\b(chiem|phan cua|ti trong|ty trong|dong gop|trong tong|tren tong)\b|\bbang\b.*(%|phan tram).*\btong\b", s) and not _QTY_RX.search(s) \
            and not re.search(r"\b(so don|don hang|so luong|tang|giam|chenh|so voi|trung binh)\b", s):
        c, v = sf[0]
        return f"answer('percent', pct(revenue({K}, {c}={v!r}), revenue({K})))"
    # gia tri hang ton theo gia von luc het mot ngay
    if one and (K == "HOM_NAY" or K.startswith("NGAY(")) and not items and re.search(r"\bgia von\b", s) and re.search(r"\b(kho|ton)\b", s) \
            and not re.search(r"\b(lo|het han|nhom|nao)\b", s):
        return f"answer('number', round_money(stock_value_at({K})))"
    # so luong ban cua MOT mat hang (don vi co ban): "thang 11 ban ra bao nhieu noi Lau ga la e"
    uq = re.search(r"\b(?:bao nhieu|may|bn|nhieu)\s+([a-z0-9]+)\b", s)
    if one and len(sf) == 1 and sf[0][0] == "item" and uq and uq[1] not in ("tien", "don", "khach", "lan", "ngay", "mat", "loai", "mon", "sp", "san") \
            and re.search(r"\bban\b", s) and not re.search(r"%|phan tram|\b(tien|doanh thu|trung binh|binh quan|moi|nhat|ton|kho|nhap|chan|le|gia|don|khach|lan)\b",
                                                         re.sub(r"\bgan (day )?nhat\b", " ", s_nm)):
        code_ = f"answer('number', qty_sold({K}, item={sf[0][1]!r}))"
        pk = _PACK_RX.search(s_nm)  # hoi theo don vi dong goi (thung, loc, hop...): chi tra so luong co ban khi do la don vi co ban
        return f"{code_} if unit_is_base({sf[0][1]!r}, {_accented(orig, pk[1])!r}) is not False else None" if pk else code_
    if one and not filt and re.search(r"\b(ngay|hom|bua) nao\b|\bngay\b.*\b(dat|dong) khach nhat\b", s) and re.search(r"\b(dat|dong) khach\b", s) and not re.search(r"\bgio\b", s):
        return f"answer('date', by_day({K}).sort_values(['so_don', 'order_date'], ascending=[False, True]).order_date.iloc[0])"
    months = [p for p in ps if p.code.startswith("THANG(")]
    if len(months) >= 2 and not filt and money and re.search(r"\bthang nao\b", s):
        pairs = ", ".join(f"('{p.a:%Y-%m}', revenue({p.code}))" for p in sorted(months, key=lambda p: p.a))
        if best:
            return f"answer('date', max([{pairs}], key=lambda t: t[1])[0])"
        if re.search(r"\b(thap nhat|it nhat)\b(?!\s*\d)", s):
            return f"answer('date', min([{pairs}], key=lambda t: t[1])[0])"
    up = re.search(r"\btang (lien tuc|deu|dan|lien tiep)\b|\bthang sau\b.*\b(cao|nhieu|lon) hon\b.*\bthang truoc\b", s)
    down = re.search(r"\bgiam (lien tuc|deu|dan|lien tiep)\b|\bthang sau\b.*\b(thap|it|nho) hon\b.*\bthang truoc\b", s)
    if len(months) >= 3 and not filt and re.search(r"\bdoanh thu\b", s) and bool(up) != bool(down):
        op = " < " if up else " > "
        return f"answer('boolean', bool({op.join(f'revenue({p.code})' for p in sorted(months, key=lambda p: p.a))}))"
    # cong no tai mot ngay (§4)
    day1 = one and (K == "HOM_NAY" or K.startswith("NGAY("))
    if day1 and any(x.startswith("Số khách còn nợ tại ngày") for x in mh):
        return f"answer('number', len(debt_by_customer({K})))"
    tn = re.search(r"\b(qua|hon|tren|tu|it nhat)\s+(\d+)\s+ngay\b", s)
    if day1 and not filt and not tn and not re.search(r"\btuoi\b", s) and any(x.startswith("Tổng nợ còn lại tại ngày") for x in mh):
        return f"answer('number', round_money(debt_as_of({K}).con_no.sum()))"
    if day1 and any(x.startswith("Khách nợ nhiều nhất") for x in mh):
        kk = re.search(r"\btop\s*(\d+)|\b(\d+) (khach|nguoi)\b", s)
        if kk:
            return f"answer('table', debt_by_customer({K}).head({int(kk[1] or kk[2])})[['khach', 'con_no']])"
        if not re.search(r"\b(nhung|cac|danh sach|liet ke)\b", s):
            return f"answer('string', debt_by_customer({K}).khach.iloc[0])"
    if day1 and tn and any(x.startswith("Tuổi nợ") for x in mh) and re.search(r"\b(tong|tien|con no)\b", s) \
            and not re.search(r"\b(may|bao nhieu|so) (don|khach)\b|%|phan tram", s):
        op = ">=" if tn[1] in ("tu", "it nhat") or re.search(r"\btro len\b", s) else ">"
        return f"answer('number', round_money(debt_as_of({K}).pipe(lambda d: d[d.tuoi_no {op} {tn[2]}]).con_no.sum()))"
    # hai so: no them va tra no trong ky (§4), ca shop hoac mot khach; thu tu theo cau hoi
    if one and any(x.startswith("Câu hỏi HAI số: nợ thêm") for x in mh):
        cus = [v for c, v in sf if c == "customer"]
        pa_ = re.search(r"\b(no them|chiu them|ghi no them|no moi)\b", s)
        pp_ = re.search(r"\btra\b", s)
        if len(cus) <= 1 and pa_ and pp_:
            f = f".pipe(lambda d: d[d.customer == {cus[0]!r}])" if cus else ""
            add, paid = f"round_money(debt_added({K}){f}.no_them.sum())", f"round_money(debt_paid({K}){f}.amount.sum())"
            return f"answer('list', [{add}, {paid}])" if pa_.start() < pp_.start() else f"answer('list', [{paid}, {add}])"
    # nhap tru ban cua MOT mat hang trong ky (§5), quy ve don vi cau hoi
    nu = re.search(r"\b(?:bao nhieu|may|bn|nhieu)\s+([a-z]+)\b", s_nm)
    if one and len(items) == 1 and any(x.startswith("Nhập trừ bán") for x in mh) and nu and nu[1] not in ("tien", "dong", "don", "lan", "ngay"):
        return f"answer('number', net_import({K}, item={items[0]!r}, unit={_accented(orig, nu[1])!r}))"
    # gia nhap binh quan mot <don vi> cua MOT mat hang trong ky (§5): don vi lay tu cau hoi ("mot hop", "1 thung")
    if one and len(items) == 1 and re.search(r"\bnhap\b", s) and re.search(r"\bbinh quan\b", s):
        u = re.search(r"\b(?:mot|1|moi)\s+(" + _UNITS + r"|lo|tuyp|hop|thung|bao|cuon|cay|m)\b", s)
        if u:
            v = f"avg_import_price({K}, item={items[0]!r}, unit={_accented(orig, u[1])!r})"
            return f"(lambda v: answer('number', v) if v is not None else answer('unanswerable', reason='không có dòng nhập có giá của mặt hàng trong kỳ'))({v})"
    # so ly size M / L ban ra, cong moi mon (ten mon ghi size trong ngoac)
    sz = re.search(r"\bsize ([sml])\b", s)
    if one and not filt and sz and re.search(r"\b(ly|coc)\b", s) and not re.search(r"%|phan tram|\b(tien|doanh thu|nhat|nao|trung binh)\b", s):
        return f"answer('number', int(top_items({K}, k=10**9).pipe(lambda t: t[t.item.str.contains(r'\\({sz[1].upper()}\\)')].value.sum())))"
    # tong so luong theo DON VI BAN, cong moi mat hang: "hom qua ban duoc bao nhieu ly", "tinh theo binh ... may binh"
    uu = re.search(r"\b(?:bao nhieu|may|bn|nhieu|so)\s+(" + _UNITS + r"|noi|binh)\b", s)
    if one and not filt and uu and re.search(r"\bban\b", s) and not sz \
            and _plain(s, ps, extra={uu[1], "cong", "moi", "mat", "hang", "tinh", "theo", "cac", "tat", "loai", "mon", "san", "pham"}) \
            and not re.search(r"%|phan tram|\b(tien|doanh thu|nhat|nao|trung binh|binh quan|ton|kho|nhap|don|khach)\b"
                              r"|\bmoi (ngay|don|thang|tuan|nguoi|khach|ca|hoa don|bill)\b", re.sub(r"\bgan (day )?nhat\b", " ", s)):
        return f"answer('number', qty_unit({K}, unit={_accented(orig, uu[1])!r}))"
    # mat hang lai gop cao nhat (§5: lai gop = doanh thu - gia von dong)
    if one and not filt and re.search(r"\b(hang|mat hang|mon|sp|san pham) nao\b|\b(hang|mat hang|mon|sp|san pham)\b.*\bla (gi|chi)\b", s) \
            and re.search(r"\b(loi|lai|loi nhuan)( gop)?( ca thang)? (nhat|cao nhat|nhieu nhat|lon nhat)\b"
                          r"|\b(loi|lai|loi nhuan) gop\b.*\b(cao nhat|nhieu nhat|lon nhat)\b", s) \
            and not re.search(r"%|phan tram|\b(it nhat|thap nhat|lo)\b", s):
        return f"answer('string', top_items({K}, k=1, by='profit')['item'].iloc[0])"
    # nhan vien / nguoi ban co doanh thu cao nhat (§3: theo doanh thu, bang nhau theo ma)
    if one and not filt and re.search(r"\b(nhan vien|nguoi ban|nv)\b", s) \
            and (re.search(r"\bdoanh thu\b", s) and re.search(r"\b(cao nhat|nhieu nhat|lon nhat)\b", s) or re.search(r"\bban (gioi|tot) nhat\b", s)) and not re.search(r"\b(don|khach|mat hang|mon|%|phan tram|ngay)\b", s):
        return f"answer('string', top_items({K}, k=1, by='amount', key='employee').iloc[0, 0])"
    # % khach co tu N don tro len tren so khach co mua (customers() da bo khach le)
    thr2 = re.search(r"(\bit nhat|\btu|\btren|\bhon|≥|>=)\s*(\d+)\s+(?:don|hoa don|bill|lan)\b", s)
    if one and not filt and thr2 and re.search(r"%|phan tram|\b(ti|ty) le\b", s) and re.search(r"\bkhach\b", s):
        op = ">=" if thr2[1] in ("it nhat", "tu", "≥", ">=") or re.search(r"\btro len\b", s) else ">"
        return f"answer('percent', pct(int((customers({K}).so_don {op} {thr2[2]}).sum()), len(customers({K}))))"
    # so don co tien hang tu X tro len
    othr = re.search(r"(\btu|\btren|\bhon|\bit nhat|\bqua|\bvuot|\bdat|≥|>=)\s*(\d[\d.,]*)\s*(trieu|tr|cu|k|nghin|ngan|dong|d|vnd)?\b", s)
    if one and not filt and othr and re.search(r"\b(bao nhieu|may|so|bn|mi|dem)\s+(don|hoa don|bill)\b|\bso don\b", s) \
            and re.search(r"\b(tien hang|doanh thu|gia tri|tien)\b", s) and not re.search(r"\b(may|bao nhieu|so|bn) (ngay|khach|mat hang|mon)\b|%|phan tram", s):
        x = _money(othr[2], othr[3])
        if x and x >= 1000:
            op = ">=" if othr[1] in ("tu", "it nhat", "dat", "≥", ">=") or re.search(r"\btro len\b", s) else ">"
            return f"answer('number', int((order_values({K}) {op} {x}).sum()))"
    thr = re.search(r"(\bit nhat|\btu|\btren|\bhon|\bqua|≥|>=)\s*(\d+)\s+(?:don|hoa don|bill)\b", s) \
        or re.search(r"\b(o)\s+(\d+)\s+(?:don|hoa don|bill)(?: hang)?(?: khac nhau)? tro len\b", s)
    if one and not filt and thr:
        op = ">=" if thr[1] in ("it nhat", "tu", "≥", ">=") or re.search(r"\btro len\b", s) else ">"
        if re.search(r"\b(bao nhieu|may|so|bn|mi)\s+(mat hang|mon|sp|san pham|loai hang|loai)\b", s):
            return f"answer('number', int((top_items({K}, k=10**9, by='orders').value {op} {thr[2]}).sum()))"
        if re.search(r"\b(bao nhieu|may|so|bn|mi)\s+(khach|nguoi)\b", s):
            return f"answer('number', int((customers({K}).so_don {op} {thr[2]}).sum()))"
    day_thr = re.search(r"(\btu|\btren|\bhon|\bit nhat|\bqua|\bvuot|\bdat|≥|>=)\s*(\d[\d.,]*)\s*(trieu|tr|cu|k|nghin|ngan|dong|d|vnd)?\b", s)
    if one and not filt and day_thr and re.search(r"\b(may|bao nhieu|so|bn|mi) (ngay|bua|hom)\b", s) and re.search(r"\bdoanh thu\b|\btien\b|\bban duoc\b", s):
        x = _money(day_thr[2], day_thr[3])
        if x and x >= 1000:
            op = ">=" if day_thr[1] in ("tu", "it nhat", "dat", "≥", ">=") or re.search(r"\btro len\b", s) else ">"
            return f"answer('number', int((by_day({K}).doanh_thu {op} {x}).sum()))"
    if one and any(x.startswith("Giá trị TB một đơn") for x in mh) and _plain(s, ps):
        return f"answer('number', avg_order_value({K}))"
    if one and any(x.startswith("Đếm đơn (§1") for x in mh) and _plain(s, ps) and not re.search(r"\b(trung binh|binh quan|moi)\b", s):
        return f"answer('number', n_orders({K}))"
    if one and any(x.startswith("Lời/lãi") for x in mh) and _plain(s, ps) and re.search(r"\b(lai|loi) gop\b", s) \
            and not re.search(r"\b(nhat|nao|mat hang|mon|sp|san pham)\b", s):
        return f"answer('number', gross_profit({K}))"
    if one and not filt and re.search(r"\bkhach\b", s) and not re.search(r"\b(%|phan tram|bao nhieu|may|so) khach\b|\bno\b", s):
        crit = None
        if re.search(r"\bnhieu (lan|don|don hang|hoa don|bill) nhat\b", s):
            crit, extra = "so_don", ""
        elif re.search(r"\btrung thanh\b", s):
            crit, extra = "so_ngay_mua", ""
        elif re.search(r"\bdoi y\b", s):
            crit, extra = "so_don_huy", "gom_huy=True"
        if crit:
            args = f"{K}, {extra}" if extra else K
            return (f"answer('string', customers({args}).pipe(lambda c: c.sort_values(['{crit}', c.columns[0]], "
                    "ascending=[False, True])).khach.iloc[0])")
        if re.search(r"\bkhach (hang )?sop\b|\bsop nhat\b", s):
            return (f"answer('string', customers({K}).pipe(lambda c: c[c.so_don >= 3].sort_values(['gia_tri_tb_don', c.columns[0]], "
                    "ascending=[False, True])).khach.iloc[0])")
    if one and _plain(s, ps) and re.search(r"\bdoanh thu\b|\btien ban\b|\btien hang\b|\bban (duoc|dc|ra)\b.*\btien\b", s) \
            and not re.search(r"%|\bmoi\b|\b(trung binh|binh quan)\b|\b(lai|loi)\b", s):
        return f"answer('number', revenue({K}))"
    return None


_CODE_FNS = ("n_tickets", "n_vehicles", "avg_ticket_value", "warranty_rate", "commission", "revenue", "qty_sold", "n_cancelled",
             "cancel_rate", "import_cost", "trade_in_total", "customer_pays", "delivery_rounds", "debt_paid", "debt_added",
             "adjustments", "net_profit", "gross_profit", "avg_order_value", "installments_as_of", "debt_aging", "stock_units")
_PCT_FNS = ("warranty_rate", "cancel_rate")


def _balanced(x: str, i: int) -> str | None:
    """Doan '( ... )' bat dau o x[i] == '(' (bo qua ngoac trong chuoi)."""
    if i >= len(x) or x[i] != "(":
        return None
    depth, q = 0, None
    for j in range(i, len(x)):
        ch = x[j]
        if q:
            q = None if ch == q else q
        elif ch in "'\"":
            q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return x[i:j + 1]
    return None


def _concrete(expr: str) -> bool:
    """Bieu thuc la code that: khong con cho trong (ky, <...>, ..., x) va ngoai chuoi trong nhay chi co ky tu ASCII."""
    bare = re.sub(r"'[^']*'|\"[^\"]*\"", "''", expr)
    return not re.search(r"<|\.\.\.|…|\bD\b|\bX\b|\(x\)|\|", bare) and bare.isascii()


# ham dung san -> phan du lieu shop phai co (khong co thi ham khong ton tai trong sandbox; train: 5/5 cau nhu vay gold unanswerable)
_FN_CAP = {"import_cost": "imports", "avg_import_price": "imports", "top_suppliers": "imports",
           "debt_paid": "debt", "debt_added": "debt", "debt_by_customer": "debt", "debt_as_of": "debt", "debt_aging": "debt",
           "stock_at": "inventory", "lot_stock_at": "inventory", "stock_units": "inventory", "stock_value_at": "inventory",
           "delivery_rounds": "delivery", "installments_as_of": "installments", "trade_in_total": "phone_orders",
           "customer_pays": "phone_orders", "net_profit": "expenses", "n_tickets": "repair", "n_vehicles": "repair",
           "commission": "repair", "by_tho": "repair", "avg_ticket_value": "repair", "warranty_rate": "repair",
           "expiring_lots": "lots", "expiring_items": "lots", "nearest_expiry": "lots", "price_effect": "sales", "price_change_date": "sales",
           "customers": "customers", "gross_profit": "profit", "net_import": "imports"}
_CAP_TEXT = {"imports": "sổ nhập hàng", "debt": "sổ công nợ", "inventory": "sổ tồn kho", "delivery": "sổ giao hàng nhiều đợt",
             "installments": "sổ trả góp", "phone_orders": "sổ thu máy cũ / khách trả", "expenses": "sổ chi phí", "repair": "sổ sửa xe",
             "lots": "sổ lô / hạn dùng", "sales": "chi tiết dòng bán", "customers": "tên / mã khách", "profit": "giá vốn trên dòng bán"}


def _code_hint(lines: list[str]) -> str | None:
    """Mot dong code nop dap an day du (ky va bo loc da dien) lay tu goi y so do, neu co."""
    for x in lines:
        for m in re.finditer(r"answer\('(\w+)', ", x):
            expr = _balanced(x, m.start() + len("answer"))
            if expr and _concrete(expr):
                return "answer" + expr
    for x in lines:
        m = re.search(r"->[^()]*?\b(" + "|".join(_CODE_FNS) + r")\(", x)
        if m and m[1] not in ("commission", "installments_as_of", "debt_aging", "stock_units", "adjustments"):
            expr = _balanced(x, m.end() - 1)
            if expr and _concrete(expr):
                return f"answer('{'percent' if m[1] in _PCT_FNS else 'number'}', {m[1]}{expr})"
    return None


def question_hints(question: str, as_of, lunar=None, index: EntityIndex | None = None) -> str:
    lines = []
    ps = question_periods(question, as_of, lunar)
    if ps:
        lines.append("Kỳ thời gian nêu trong câu hỏi (đã tính theo §2; dùng ĐÚNG biểu thức này làm kỳ, không dùng biến kỳ khác):")
        lines += [p.line() for p in ps]
        r = getattr(index, "data_range", None) if index is not None else None
        qn = _norm(question)
        # §7: ky nam HOAN TOAN ngoai khoang ngay co du lieu -> khong tra loi duoc (khong phai 0). Bo qua cau hoi ve han dung,
        # ky gop, ton kho, cong no (ngay hoi co the sau ngay giao dich cuoi ma van hop le).
        outside = [p for p in ps if p.b < r[0] or p.a > r[1]] if r else []
        compare = re.search(r"%|phan tram|\b(so voi|so vs|tang|giam|thay doi|hon hay kem|cao hay thap|len hay xuong|chenh)\b", qn)
        if r and outside and (len(outside) == len(ps) or compare) and not any(_future(p) for p in ps) \
                and not re.search(r"\b(het han|hsd|han dung|han su dung|can date|gop|ky gop|ton|kho|no|cong no|tuoi no|bao hanh)\b", qn):
            which = "mọi kỳ trên" if len(outside) == len(ps) else "kỳ so " + ", ".join(p.code for p in outside)
            lines.append(f"- CHÚ Ý: {which} nằm HOÀN TOÀN ngoài khoảng ngày có dữ liệu của shop ({r[0]:%Y-%m-%d} → {r[1]:%Y-%m-%d}) "
                         "-> không trả lời được: answer('unanswerable', reason='kỳ hỏi nằm ngoài khoảng dữ liệu') (§7: không có dữ liệu khác với 0).")
        elif r and any(p.a < r[0] <= p.b or p.a <= r[1] < p.b for p in ps) and not any(_future(p) for p in ps):
            lines.append(f"- Dữ liệu của shop từ {r[0]:%Y-%m-%d} đến {r[1]:%Y-%m-%d}: kỳ chỉ có một phần trong khoảng này thì tính trên phần có dữ liệu (§7).")
    else:
        lines.append("Câu hỏi không nêu kỳ cụ thể nào nhận diện được; chọn kỳ trong bảng Mốc thời gian theo §2.")
    if index is not None:
        ms = index.match(question)
        if ms:
            lines.append("Tên trong dữ liệu khớp với câu hỏi (dùng đúng chính tả này làm giá trị lọc; S/N/P là bảng chứa):")
            for col, v in ms:
                tbl, c = col.split(".", 1) if "." in col else ("S", col)
                lines.append(f"- {tbl}.{c} = {v!r}" + (f" (lọc {c}={v!r})" if tbl == "P" else ""))
    mh = measure_hints(question, index.kinds if index is not None else frozenset())
    qs = _norm(question)
    no_gender = False
    # khach o / nha o / tai duong X (khong bat "duong" la duong an: "ban duoc bao nhieu kg duong")
    addr_q = re.search(r"\bkhach( hang)?( nha)?( o| tai)? duong\b|\b(o|tai) duong\b|\bdia chi\b", qs) and re.search(r"\bkhach\b", qs)
    if index is not None and addr_q and getattr(index, "addr_cols", None) is not None:
        if index.addr_cols:
            mh.append(f"Khách theo địa chỉ / tên đường -> lọc cột địa chỉ {', '.join(index.addr_cols[:2])} chứa tên đường (so không dấu, không phân biệt hoa thường).")
        else:
            mh.append("Dữ liệu không ghi địa chỉ khách (không có cột địa chỉ; không suy từ tên) -> answer('unanswerable', reason='shop không ghi địa chỉ khách') (§7).")
    if index is not None and re.search(r"\bkhach (hang )?(nam|nu|gioi tinh)\b|\b(nam|nu) gioi\b|\bgioi tinh\b", qs):
        if index.gender_col:
            mh.append(f"Khách nam / nữ -> cột giới tính {index.gender_col} (giá trị xem trong mô tả bảng); tỉ trọng chia doanh thu CẢ SHOP cùng kỳ (§1).")
        else:
            mh.append("Dữ liệu không ghi giới tính khách (không có cột giới tính; không đoán theo tên) -> answer('unanswerable') (§7).")
            no_gender = True
    if re.search(r"\b(tra|thanh toan) (bang|qua) ", qs) and re.search(r"\bdon\b", qs):
        mh.append("'Đơn trả bằng X' / 'thanh toán qua X' = PHƯƠNG THỨC THANH TOÁN X (cột payment), không phải đơn trả hàng; lọc payment=<giá trị trong dữ liệu>.")
    if index is not None and re.search(r"\b(ke don|theo don|ban theo don|don bac si|toa bac si|don thuoc)\b", qs):
        neg = re.search(r"\b(khong|ko|k|hong|hok|khg|kg|chua|tru) (phai |can |thuoc |hang |co )?(ke don|theo don|ban theo don|don bac si|toa bac si|don thuoc)\b|\bngoai ke don\b|\botc\b", qs)
        if index.rx_col:
            mh.append(f"Thuốc {'KHÔNG ' if neg else ''}kê đơn -> bộ lọc {index.rx_col}={'False' if neg else 'True'} (cột của S), vd revenue(kỳ, {index.rx_col}={'False' if neg else 'True'}).")
        else:
            mh.append("Shop không có cột thuốc kê đơn trong danh mục -> kiểm tra README; không có thì unanswerable (§7).")
    if len(ps) >= 2 and any(x.startswith(("Phần trăm thay đổi", "Chênh lệch")) for x in mh):
        meas = "revenue" if re.search(r"\bdoanh thu\b|\bban duoc\b.*\btien\b|\btien\b", qs) else None
        if meas:
            fn = "pct_change" if any(x.startswith("Phần trăm") for x in mh) else None
            expr = f"{fn}({meas}({ps[0].code}), {meas}({ps[1].code}))" if fn else f"{meas}({ps[0].code}) - {meas}({ps[1].code})"
            mh.append(f"Kỳ hỏi = {ps[0].code}, kỳ so = {ps[1].code} -> answer('{'percent' if fn else 'number'}', {expr}) (thêm bộ lọc nếu câu hỏi có).")
    # dien san bo loc da nhan ra: bang P (viec, tho, dong_xe); bang S khi chi khop dung mot ten
    msf = index.match(question, partial=False) if index is not None else []  # dien bo loc chi theo ten khop tron
    pf = [(c.split(".", 1)[1], v) for c, v in msf if c.startswith("P.")]
    sf = [(c, v) for c, v in msf if "." not in c]
    sf = list(dict.fromkeys(sf))  # cung mot ten khop hai lan ("Zalo OA")
    gv = {str(v).casefold() for c, v in sf if c == "group"}  # cot danh muc dm_* trung gia tri voi group: cung mot bo loc
    nv = {str(v).casefold() for c, v in sf if not c.startswith("dm_")}  # cot danh muc trung gia tri voi item / group
    sf = [(c, v) for c, v in sf if not (c.startswith("dm_") and (str(v).casefold() in gv or str(v).casefold() in nv))]
    q_pay = _PAY_PHRASE.sub(" ", qs)
    q_pay = re.sub(r"\btra (het|no|tien|lai|gop|du|xong|ngay|dan|cham|tre|truoc|sau|them|duoc|dc)\b", " ", q_pay)  # "tra" la dong tu
    # ten mot tu chi xuat hien trong cum thanh toan ("tra bang tm") khong phai bo loc (nhom 'Trà')
    if not re.search(r"\bkhach (hang )?le\b", qs):  # "ti le", "chan, le" khong phai khach le
        sf = [(c, v) for c, v in sf if not (c == "customer" and _norm(str(v)) == "le")]
    sf = [(c, v) for c, v in sf if c in ("payment", "channel") or len(_norm(str(v)).split()) > 1
          or re.search(r"\b" + re.escape(_norm(str(v))) + r"\b", q_pay)]
    two_vals = len(sf) == 2 and sf[0][0] == sf[1][0] and sf[0][1] != sf[1][1]
    multi = (len(re.findall(r"\b(doanh thu|so don|so luong)\b", qs)) >= 2 or len(re.findall(r"\bbao nhieu\b", qs)) >= 2 or two_vals) \
        and re.search(r",|\bva\b|\bvoi\b", qs) \
        and not re.search(r"%|phan tram|\bso voi\b|\bhon\b|\bkem\b|\bchenh\b", qs)
    kinds_ = index.kinds if index is not None else frozenset()
    rx_on = [m[1] for x in mh if x.startswith("Thuốc") and (m := re.search(r"bộ lọc (\w+=(?:True|False)) \(cột của S\)", x))]
    sf_x = sf if sf or len(rx_on) != 1 else [(rx_on[0].split("=")[0], rx_on[0].split("=")[1] == "True")]  # loc thuoc ke don
    if multi:
        special = _pair_code(qs, ps, sf)
        if special is None:  # cau hai so (no them + tra no, chan + le...): chi nhan dang rieng tra list
            special = _special_code(qs, ps, sf, mh, kinds_, question)
            special = special if special and special.startswith("answer('list'") else None
    else:
        special = _special_code(qs, ps, sf_x, mh, kinds_, question)
    if special is None and index is not None and len(ps) == 1 and re.search(r"\b(bim|ta)\b", qs):
        special = _diaper_code(qs, ps[0].code, sf, index)
    caps0 = getattr(index, "caps", None) if index is not None else None
    # §7: hoi theo nhom hang ma shop khong ghi nhom hang (khong co cot nhom / loai trong danh muc) -> khong tra loi duoc
    no_group = caps0 is not None and "sales" in caps0 and "groups" not in caps0 and re.search(r"\bnhom\b(?! (tuoi|no|khach|tho|nhan vien))", qs)
    # §7: hoi ly do huy / tra ma shop khong ghi ly do (khong co cot ly do / nguyen nhan) -> khong tra loi duoc (train 2/2)
    q_noreason = _PAY_PHRASE.sub(" ", qs)
    no_reason = caps0 is not None and "sales" in caps0 and "reasons" not in caps0 and re.search(r"\b(huy|tra)\b", q_noreason) \
        and re.search(r"\b(ly do|li do|vi sao|tai sao|nguyen nhan|vi gi|chu yeu vi)\b|\b(huy|tra|tra lai|doi)\b.*\bvi\b", q_noreason)
    subj = re.search(r"\b(le phep|chu dao|tan tam|tan tinh|lanh nghe|tay nghe|quy men|de thuong|than thien|vui ve|lich su|hien lanh|"
                     r"dang tin|uy tin nhat|khen|dep trai|xinh|thai do|niem no|cau tha)\b", qs) and not re.search(r"\b(cham chi|so don|doanh thu)\b", qs)
    if subj:
        mh.append("Câu hỏi đánh giá con người / thái độ (lễ phép, tay nghề, được khen...): dữ liệu bán hàng không ghi -> answer('unanswerable', reason='dữ liệu không ghi đánh giá này') (§7).")
    if no_reason:
        mh.append("Shop không ghi lý do huỷ / trả hàng (không có cột lý do / nguyên nhân) -> answer('unanswerable', reason='shop không ghi lý do huỷ / trả') (§7).")
    no_cgroup = caps0 is not None and "sales" in caps0 and "customer_groups" not in caps0 and not any(c == "customer_group" for c, _ in sf) \
        and re.search(r"\bkhach (hang )?(nhom|thuoc nhom|than thiet|vip)\b|\bnhom khach\b", qs)
    if no_cgroup:
        mh.append("Shop không ghi nhóm khách (không có cột nhóm / hạng khách) -> answer('unanswerable', reason='shop không ghi nhóm khách') (§7).")
    if no_group:
        mh.append("Shop không ghi nhóm hàng (danh mục không có cột nhóm / loại; không đoán nhóm từ tên hàng) -> answer('unanswerable', reason='shop không ghi nhóm hàng') (§7).")
    if mh or special or no_reason or any("nằm HOÀN TOÀN ngoài khoảng" in x for x in lines):
        if len(ps) == 1:  # mot ky duy nhat: dien san bieu thuc ky vao loi goi goi y
            mh = [re.sub(r"\((kỳ|ngày|D)(?=[,)])", "(" + ps[0].code, x) for x in mh]
        if pf or any("viec=..." in x for x in mh):
            arg = "".join(f", {k}={v!r}" for k, v in pf)
            mh = [re.sub(r"(?:, (?:viec|tho|dong_xe)=\.\.\.)+", arg, x) for x in mh]
        tho = [v for k, v in pf if k == "tho"]
        if len(tho) == 1:
            mh = [x.replace("<tho>", repr(tho[0])) for x in mh]
        if len(sf) == 1 and not any(x.startswith(("Ca ", "Thuốc")) for x in mh):
            col, v = sf[0]
            mh = [x.replace("<cột>=<giá trị>", f"{col}={v!r}").replace(", <lọc>", f", {col}={v!r}") for x in mh]
        rxf = [m[1] for x in mh if x.startswith("Thuốc") and (m := re.search(r"bộ lọc (\w+=(?:True|False)) \(cột của S\)", x))]
        if not sf and len(rxf) == 1:  # chi loc thuoc ke don / khong ke don
            mh = [x.replace("<cột>=<giá trị>", rxf[0]).replace(", <lọc>", f", {rxf[0]}") for x in mh]
        if mh:
            lines.append("Dạng số đo:")
            lines += ["- " + x for x in mh]
        reason_q = re.search(r"\b(huy|tra)\b.*\b(vi|do|ly do|li do)\b", _PAY_PHRASE.sub(" ", qs))  # "tra bang vi" la thanh toan
        skip = multi or any(x.startswith("Đổi giá") for x in mh) or reason_q
        code = None if skip else (_code_hint(mh) if len(ps) <= 1 else _code_hint([x for x in mh if x.startswith("Kỳ hỏi =")]))
        caps = getattr(index, "caps", None) if index is not None else None
        # code dang rieng truoc, code tu dong so do sau; bi loai (kieu dap an, shop khong ghi don vi ban) thi thu ung vien sau
        chosen = None
        for c in (special, code):
            c = _guard_code(c, qs, sf) if c else None
            if c and "qty_unit(" in c and caps is not None and "units" not in caps:
                c = None  # shop khong ghi don vi ban tren dong ban: de model tu tinh theo don vi co ban
            if c:
                chosen = c
                break
        code = chosen if chosen is None or not _unexplained(qs, ps, chosen, [v for _, v in msf]) else None
        miss = [f for f in re.findall(r"\b(\w+)\(", code or "") if _FN_CAP.get(f) and caps is not None and _FN_CAP[f] not in caps]
        if code and miss:
            need = _CAP_TEXT[_FN_CAP[miss[0]]]
            lines.append(f"- CHÚ Ý: shop không có {need} (hàm {miss[0]} không có trong sandbox) -> không trả lời được (§7).")
            code = f"answer('unanswerable', reason='shop không ghi {need}')"
        if no_group:
            code = "answer('unanswerable', reason='shop không ghi nhóm hàng')"
        if no_gender:
            code = "answer('unanswerable', reason='shop không ghi giới tính khách')"
        if subj:
            code = "answer('unanswerable', reason='dữ liệu không ghi đánh giá này')"
        if no_cgroup:
            code = "answer('unanswerable', reason='shop không ghi nhóm khách')"
        if no_reason:
            code = "answer('unanswerable', reason='shop không ghi lý do huỷ / trả')"
        if any("nằm HOÀN TOÀN ngoài khoảng" in x for x in lines):
            code = "answer('unanswerable', reason='kỳ hỏi nằm ngoài khoảng dữ liệu')"
        if addr_q and getattr(index, "addr_cols", None) is not None:
            code = "answer('unanswerable', reason='shop không ghi địa chỉ khách')" if not index.addr_cols else None
        if code:
            lines.append(f"Code gợi ý (đã điền kỳ và bộ lọc theo luật; chạy đúng dòng này, chỉ sửa khi kết quả cho thấy sai): {code}")
    return "\n".join(lines)
