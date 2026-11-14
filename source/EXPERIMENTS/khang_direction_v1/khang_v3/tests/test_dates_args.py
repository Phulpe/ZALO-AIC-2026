import os, sys, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.dates import date_args, to_date
D = dt.date(2025, 8, 24)
assert date_args(("2025-08-24",), 1) == [D]
assert date_args((2025, 8, 24), 1) == [D]
assert date_args(((2025, 8, 24),), 1) == [D]
assert date_args((2025, 8, 1, 2025, 8, 31), 2) == [dt.date(2025, 8, 1), dt.date(2025, 8, 31)]
assert date_args(("2025-08-01", "2025-08-31"), 2) == [dt.date(2025, 8, 1), dt.date(2025, 8, 31)]
assert to_date((2025, 8, 24)) == D
try:
    date_args((), 1, "debt_as_of"); raise SystemExit("phai bao loi")
except TypeError as e:
    assert "debt_as_of" in str(e)
print("OK")

# ky (dau, cuoi) cho ham "tinh den het ngay": lay ngay cuoi ky
from datamate.dates import period_vars as _pv
_v = _pv("2026-07-17")
assert date_args((_v["NGAY"](28, 6, 2025),), 1)[0].isoformat() == "2025-06-28"
assert date_args((_v["THANG"](6, 2025),), 1)[0].isoformat() == "2025-06-30"
print("OK period-as-of")
