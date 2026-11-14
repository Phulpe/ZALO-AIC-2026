"""Test don vi cho bo doc so / ngay / dinh dang dap an (chay: PYTHONPATH=src python tests/test_parsing.py)."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.answers import normalize, same_answer, score
from datamate.dates import LunarCalendar, month_year_for, week_range
from datamate.helpers import status_of
from datamate.parsing import parse_datetime, parse_number

NUM = {
    "90k": 90000, "12tr460": 12460000, "1tr780": 1780000, "7,59tr": 7590000, "1,3125tr": 1312500,
    "17,224666tr": 17224666, "1198,032tr": 1198032000, "105.000đ": 105000, "84,000": 84000,
    "1.234.567": 1234567, "2,08": 2.08, "18.6": 18.6, "(2)": -2, "-4tr300": -4300000, "0": 0,
    "1,5k": 1500, "2tỷ300": 2300000000, "20.3m": 20.3,
}
UNIT = {"2 cái": (2, "cái"), "1 xe 4m3": (1, "xe 4m3"), "3 thùng": (3, "thùng"), "2,5m": (2.5, None)}
DATES = {
    ("1 thg 3 2025", None): dt.datetime(2025, 3, 1), ("23/07/2025", None): dt.datetime(2025, 7, 23),
    ("1/1", 2026): dt.datetime(2026, 1, 1), (46023, None): dt.datetime(2026, 1, 1),
    ("2026-01-01 06:02:43", None): dt.datetime(2026, 1, 1, 6, 2, 43), ("ngày 3 tháng 4 năm 2025", None): dt.datetime(2025, 4, 3),
}


def test_numbers():
    for raw, want in NUM.items():
        got = parse_number(raw)[0]
        assert got is not None and abs(got - want) < 1e-6, (raw, got, want)
    for raw, (v, u) in UNIT.items():
        got = parse_number(raw)
        assert abs(got[0] - v) < 1e-9 and (u is None or got[1] == u), (raw, got)
    assert parse_number("abc") == (None, None)


def test_dates():
    for (raw, yh), want in DATES.items():
        assert parse_datetime(raw, yh) == want, (raw, parse_datetime(raw, yh))
    assert week_range(dt.date(2026, 6, 10)) == (dt.date(2026, 6, 8), dt.date(2026, 6, 14))
    assert month_year_for(7, dt.date(2026, 6, 10)) == 2025 and month_year_for(6, dt.date(2026, 6, 10)) == 2026
    cal = LunarCalendar.load(os.path.join(os.path.dirname(__file__), "..", "data", "train", "calendar.csv"))
    assert cal.tet_range(2026) == (dt.date(2026, 2, 10), dt.date(2026, 2, 26))


def test_answers():
    assert normalize("percent", 2.0849) == {"type": "percent", "answer": 2.08}
    assert normalize("percent", 2.005) == {"type": "percent", "answer": 2.01}  # nua len
    assert normalize("number", 89526400.0) == {"type": "number", "answer": 89526400}
    assert normalize("date", "2025-9") == {"type": "date", "answer": "2025-09"}
    assert same_answer({"type": "string", "answer": "cà phê SỮA (m)"}, {"type": "string", "answer": "Cà phê sữa (M)"})
    assert not same_answer({"type": "string", "answer": "Sting dầu 330ml"}, {"type": "string", "answer": "Sting dâu 330ml"})
    assert same_answer({"type": "list", "answer": ["SP0003"]}, {"type": "list", "answer": ["Cà phê sữa (M)"], "alias": [["SP0003"]]})
    t = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["A", 1], ["B", 2]]}}
    p = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["B", 2], ["A", 1]]}}
    assert same_answer(p, t)
    assert score({"type": "abstain"}, t) == 0 and score({"type": "number", "answer": 1}, t) == -1


def test_status():
    cases = {"Đã hoàn thành": "hoàn thành", "đã xong": "hoàn thành", "Huỷ đơn": "huỷ", "cancel": "huỷ",
             "Không giao được": "huỷ", "Đổi trả": "trả hàng", "hàng trả về": "trả hàng", "Đang vận chuyển": "đang giao",
             "đang ship": "đang giao", "Completed": "hoàn thành", "đã lấy": "hoàn thành", "Hoàn trả": "trả hàng",
             "ghi sổ, khách trả hết rồi": None, "nợ 1tr242, khách ck đủ rồi": None, "Khách trả": "trả hàng",
             "khách mang trả": "trả hàng", "trả lại hàng": "trả hàng", "Đang xử lý": "đang giao", "chưa giao": "đang giao",
             "ok": "hoàn thành", "Thành công": "hoàn thành", "Đã hoàn hàng": "trả hàng", "Đang giao dịch": "đang giao"}
    for raw, want in cases.items():
        assert status_of(raw) == want, (raw, status_of(raw))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
