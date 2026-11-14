import os, sys, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.dates import period_vars, LunarCalendar
from datamate.kernel import load_business_cached
from datamate.metrics import Metrics
from datamate.helpers import pct_change
v = period_vars("2025-09-27", LunarCalendar.load("data/train/calendar.csv"))
assert v["THANG_TRUOC_CUNG_KY"] == (dt.date(2025, 8, 1), dt.date(2025, 8, 27))
assert v["TUAN_TRUOC"] == (dt.date(2025, 9, 15), dt.date(2025, 9, 21))
assert v["THANG"](12) == (dt.date(2024, 12, 1), dt.date(2024, 12, 31))
assert v["TET"](2026) == (dt.date(2026, 2, 10), dt.date(2026, 2, 26))
b = load_business_cached("data/train/data/biz_103", "/tmp/datamate_cache_dev")
m = Metrics(b.sales.S, b.sales.O)
r = pct_change(m.revenue(v["THANG_NAY"]), m.revenue(v["THANG_TRUOC_CUNG_KY"]))
assert r == 0.92, r  # q_6500a9bd7c72
b = load_business_cached("data/train/data/biz_109", "/tmp/datamate_cache_dev")
m = Metrics(b.sales.S, b.sales.O)
n = m.n_orders(period_vars("2026-08-05")["THANG"](10, 2025), ca="chiều")
assert n == 920, n  # q_745a5b81c65b
print("OK", r, n)
