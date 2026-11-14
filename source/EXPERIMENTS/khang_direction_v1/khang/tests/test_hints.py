"""Bo nhan dien ky / ten / dang so do trong cau hoi (src/datamate/hints.py).

Cac cau lay tu nhung lan model chon sai ky tren bo train (eval10_rand40)."""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.dates import LunarCalendar, period_vars  # noqa: E402
from datamate.hints import EntityIndex, measure_hints, question_periods  # noqa: E402

LUN = LunarCalendar.load(os.path.join(os.path.dirname(__file__), "..", "data", "train", "calendar.csv"))


def codes(q, as_of):
    return [p.code for p in question_periods(q, as_of, LUN)]


D = dt.date
cases = [
    ("<REDACTED_COMPETITION_QUESTION>", "2026-05-25", ["THANG(5, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-09-20", ["THANG(7, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2025-01-21", ["TUAN_TRUOC"]),
    ("e so dia Canh ga chien nuoc mam ban dc trong thang 1/2025 la bao nhieu", "2026-05-11", ["THANG(1, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-06-02", ["THANG(12, 2025)", "THANG(11, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-29", ["THANG(12, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-09-20", ["THANG(3, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-05", ["THANG(10, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-09-03", ["QUY(4, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-05",
     ["THANG(7, 2025)", "THANG(8, 2025)", "THANG(9, 2025)"]),
    ("trong các tháng 7, 8 và 9/2025 tháng nào doanh thu cao nhất", "2026-08-05", ["THANG(7, 2025)", "THANG(8, 2025)", "THANG(9, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-06-16", ["THANG(1, 2025)"]),
    ("doanh thu tháng này so với tháng trước tăng mấy %", "2025-09-27", ["THANG_NAY", "THANG_TRUOC_CUNG_KY"]),
    ("doanh thu tháng trước là bao nhiêu", "2025-09-27", ["THANG_TRUOC"]),
    ("doanh thu tháng 3 là bao nhiêu", "2025-09-27", ["THANG(3, 2025)"]),
    ("doanh thu tháng 12 là bao nhiêu", "2025-09-27", ["THANG(12, 2024)"]),
    ("doanh thu ngày 15/3/2025", "2025-09-27", ["NGAY(15, 3, 2025)"]),
    ("doanh thu ngày 15/12", "2025-09-27", ["NGAY(15, 12, 2024)"]),
    ("7 ngày gần nhất bán được bao nhiêu", "2025-09-27", ["N_NGAY_GAN_NHAT(7)"]),
    ("doanh thu Tết 2026 so với trước Tết", "2026-03-01", ["TET(2026)", "TRUOC_TET(2026)"]),
    ("năm 2025 doanh thu bao nhiêu", "2026-03-01", ["NAM(2025)"]),
    ("Q3/2025 có bao nhiêu đơn", "2026-03-01", ["QUY(3, 2025)"]),
    ("quý trước có bao nhiêu đơn", "2026-03-01", ["QUY_TRUOC"]),
    ("tính đến hôm nay còn nợ bao nhiêu", "2026-03-01", ["HOM_NAY"]),
    ("bán được bao nhiêu hộp Panadol", "2026-03-01", []),
]
for q, as_of, want in cases:
    got = codes(q, as_of)
    assert got == want, (q, got, want)

# bieu thuc goi y phai trung voi bien ky trong sandbox
v = period_vars("2026-06-02", LUN)
p = question_periods("<REDACTED_COMPETITION_QUESTION>", "2026-06-02", LUN)
assert (p[1].a, p[1].b) == v["THANG"](11, 2025)
p = question_periods("doanh thu tháng này so với tháng trước", "2025-09-27", LUN)
assert (p[1].a, p[1].b) == period_vars("2025-09-27", LUN)["THANG_TRUOC_CUNG_KY"]
p = question_periods("doanh thu ngày 15/12", "2025-09-27", LUN)
assert (p[0].a, p[0].b) == period_vars("2025-09-27", LUN)["NGAY"](15, 12)

# dang so do
assert any("qty_sold" in h for h in measure_hints("<REDACTED_COMPETITION_QUESTION>"))
assert any("qty_sold" in h for h in measure_hints("e so dia Canh ga chien nuoc mam ban dc trong thang 1/2025 la bao nhieu"))
assert not any("qty_sold" in h for h in measure_hints("nhóm dị ứng chiếm bao nhiêu phần trăm doanh thu"))
assert any("profit" in h for h in measure_hints("<REDACTED_COMPETITION_QUESTION>"))
assert not any("profit" in h for h in measure_hints("Paracetamol còn lại bao nhiêu hộp"))
assert any("n_orders" in h for h in measure_hints("<REDACTED_COMPETITION_QUESTION>"))
assert any("customers" in h for h in measure_hints("<REDACTED_COMPETITION_QUESTION>"))
assert any("YYYY-MM" in h for h in measure_hints("tháng nào doanh thu cao nhất"))

# ten trong du lieu
idx = EntityIndex({"item": ["Rau củ kho quẹt", "Cá kèo kho tộ", "Cánh gà chiên nước mắm", "Cánh gà chiên"], "group": ["dị ứng", "tiêu hoá"]})
assert idx.match("<REDACTED_COMPETITION_QUESTION>") == [("item", "Rau củ kho quẹt")]
assert idx.match("e so dia Canh ga chien nuoc mam ban dc") == [("item", "Cánh gà chiên nước mắm")]
assert ("group", "dị ứng") in idx.match("dthu nhom hang di ung chiem bao nhiu phan tram")
print("OK", len(cases))

# public test: nam 2 chu so, tieng mien Trung, "N ngay toi"
more = [
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-01", ["THANG(3, 2026)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-01", ["THANG(9, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-01", ["THANG(5, 2026)", "THANG(5, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-01", ["THANG(6, 2025)", "THANG(7, 2025)", "THANG(8, 2025)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-10", ["THANG_NAY"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-10", ["HOM_NAY"]),
    ("Q4/25 có bao nhiêu đơn", "2026-08-10", ["QUY(4, 2025)"]),
    ("doanh thu ngày 25/12", "2026-08-10", ["NGAY(25, 12, 2025)"]),
    ("hàng nào hết hạn trong 120 ngày tới mà hiện vẫn còn tồn kho", "2026-06-28", ['("2026-06-28", "2026-10-26")']),
]
for q, as_of, want in more:
    got = codes(q, as_of)
    assert got == want, (q, got, want)
print("OK public", len(more))
for q, as_of, want in [
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-10", ["THANG(3, 2026)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-10", ["HOM_NAY"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-10", ["TU_DAU_NAM"]),
    ("doanh thu tháng năm nay", "2026-08-10", ["THANG_NAY", "NAM_NAY"]) if False else ("doanh thu tháng mười hai", "2026-08-10", ["THANG(12, 2025)"]),
]:
    got = codes(q, as_of)
    assert got == want, (q, got, want)
print("OK words")
for q, as_of, want in [
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-21",
     ["THANG(5, 2025)", "THANG(7, 2025)", "THANG(6, 2025)", "(THANG(5, 2025)[0], THANG(7, 2025)[1])"]),
    ("Th9/2025 đến hết th11/2025 tiệm có bnhiu xe khách quen", "2026-08-21",
     ["THANG(9, 2025)", "THANG(11, 2025)", "THANG(10, 2025)", "(THANG(9, 2025)[0], THANG(11, 2025)[1])"]),
]:
    got = codes(q, as_of)
    assert got == want, (q, got, want)
from datamate.hints import _norm  # noqa: E402
assert "voucher" in _norm("<REDACTED_COMPETITION_QUESTION>")
assert any("adjustments" in h for h in measure_hints("<REDACTED_COMPETITION_QUESTION>"))
assert any("net_profit" in h for h in measure_hints("<REDACTED_COMPETITION_QUESTION>"))
print("OK ranges")
for q, as_of, want in [
    ("<REDACTED_COMPETITION_QUESTION>", "2026-04-20", ["THANG(1, 2026)", "THANG(2, 2026)", "THANG(3, 2026)"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-04-20",
     ["THANG(9, 2025)", "THANG(11, 2025)", "THANG(10, 2025)", "(THANG(9, 2025)[0], THANG(11, 2025)[1])"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-04-20", ["THANG(4, 2025)"]),
    ("doanh thu tháng tư", "2026-04-20", ["THANG(4, 2026)"]),
    ("T8/2025 doanh thu", "2026-04-20", ["THANG(8, 2025)"]),
]:
    got = codes(q, as_of)
    assert got == want, (q, got, want)
print("OK edge")
# tiem sua xe (dev_012): ten dong xe theo moi cach ghi -> ten chuan; viec chuan; goi y ham co dien san ky
from datamate.hints import question_hints  # noqa: E402

ridx = EntityIndex({}, aliases=[("P.dong_xe", "Air Blade", a) for a in ("AB", "airblade", "Air Blade")]
                   + [("P.viec", "sửa điện", a) for a in ("sửa điện", "làm điện")] + [("P.viec", "thay ắc quy", "thay bình")],
                   kinds=frozenset({"repair"}))
h = question_hints("Trong tháng 1/2026, bao nhiêu xe AB khác nhau vào tiệm?", D(2026, 9, 30), LUN, ridx)
assert "P.dong_xe = 'Air Blade'" in h and "n_vehicles(THANG(1, 2026)" in h, h
h = question_hints("Lượt xe làm điện tháng 8/25 bnhiu?", D(2026, 9, 30), LUN, ridx)
assert "P.viec = 'sửa điện'" in h and "n_tickets(THANG(8, 2025)" in h and "n_vehicles" not in h, h
h = question_hints("Tiền trung bình một phiếu tháng 3/2026 là bao nhiêu?", D(2026, 9, 30), LUN, ridx)
assert "avg_ticket_value(THANG(3, 2026)" in h and "ắc quy" not in h, h
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 9, 30), LUN, ridx)
assert "commission(THANG(7, 2026))[['tho', 'hoa_hong']]" in h, h
h = question_hints("Th9/2025 đến hết th11/2025 tiệm có bnhiu xe khách quen", D(2026, 9, 30), LUN, ridx)
assert "regular_vehicles(kỳ" in h, h  # nhieu ky: khong dien san
# cua hang dien thoai / giao hang: goi y loi goi cu the
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 9, 30), LUN, EntityIndex({}))
assert "top_items(THANG(1, 2026), k=3, key='dong_may')" in h, h
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 10), LUN, EntityIndex({}))
assert "answer('number', delivery_rounds(THANG(11, 2025)))" in h, h
print("OK repair/phone/delivery hints")
# public: dang "m-yyyy", thang den nay, cung thang nam truoc, N thang tinh toi (pub, 2026-11-07)
for q, as_of, want in [
    ("<REDACTED_COMPETITION_QUESTION>", "2026-08-31", ["THANG(7, 2026)"]),
    ("doanh thu 2026-07", "2026-08-31", ["THANG(7, 2026)"]),
    ("Doanh thu 3 thang lien nhau tinh toi 3-2026, tung thang bao nhieu", "2026-08-31", ["THANG(1, 2026)", "THANG(2, 2026)", "THANG(3, 2026)"]),
    ("dthu shop mk tu mung 1 toi het hom nay so vs cung ky thang trc cao hay thap hon bnhiu %", "2026-06-15", ["THANG_NAY", "THANG_TRUOC_CUNG_KY"]),
    ("<REDACTED_COMPETITION_QUESTION>", "2026-06-15", ["THANG(2, 2026)", "THANG(2, 2025)"]),
    ("doanh thu quy 1/2026 so voi cung ky nam truoc", "2026-06-15", ["QUY(1, 2026)", "QUY(1, 2025)"]),
]:
    got = codes(q, as_of)
    assert got == want, (q, got, want)
E0 = EntityIndex({})
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 15), LUN, E0)
assert "pct_change(revenue(THANG(8, 2025)), revenue(THANG(7, 2025)))" in h, h
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 15), LUN, E0)
assert "n_cancelled(THANG(4, 2026))" in h, h
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 15), LUN, E0)
assert "by='receipts'" in h, h
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 15), LUN, E0)
assert "debt_by_customer(NGAY(11, 5, 2026)).head(3)" in h, h
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 15), LUN, E0)
assert "top_items(THANG(5, 2025), k=3)" in h, h
print("OK public general")
# §7: ky nam hoan toan ngoai khoang du lieu -> unanswerable (train: 16/16 cau gold unanswerable, khong bao nham cau nao)
R = EntityIndex({})
R.data_range = (D(2025, 2, 1), D(2026, 9, 14))  # biz_106
assert "nằm HOÀN TOÀN ngoài khoảng" in question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 9, 14), LUN, R)
assert "kỳ so THANG(1, 2025)" in question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 9, 14), LUN, R)
assert "HOÀN TOÀN" not in question_hints("doanh thu tháng 3/2025", D(2026, 9, 14), LUN, R)
assert "HOÀN TOÀN" not in question_hints("Hàng nào hết hạn trong 120 ngày tới mà vẫn còn tồn kho?", D(2026, 9, 14), LUN, R)
print("OK out of range")
# Code goi y: mot dong answer day du (ky + bo loc da dien) - pub15 2026-11-07: model bo qua goi y, tu loc so goc ra 3 thay vi 25
h = question_hints("Lượt xe làm điện tháng 8/25 bnhiu?", D(2026, 9, 30), LUN, ridx)
assert "Code gợi ý" in h and h.rstrip().endswith("answer('number', n_tickets(THANG(8, 2025), viec='sửa điện'))"), h
h = question_hints("Trong tháng 1/2026, bao nhiêu xe AB khác nhau vào tiệm?", D(2026, 9, 30), LUN, ridx)
assert h.rstrip().endswith("answer('number', n_vehicles(THANG(1, 2026), dong_xe='Air Blade'))"), h
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 15), LUN, E0)
assert h.rstrip().endswith("answer('percent', pct_change(revenue(THANG(8, 2025)), revenue(THANG(7, 2025))))"), h
assert "Code gợi ý" not in question_hints("doanh thu tháng 3/2025 của món nào cao nhất", D(2026, 6, 15), LUN, E0)
print("OK code hints")
# hoi theo dia chi khach: shop khong co cot dia chi -> unanswerable (train: 6/6 cau gold unanswerable); khong bat "duong" an
A = EntityIndex({})
A.addr_cols = []
h = question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 1), LUN, A)
assert h.rstrip().endswith("answer('unanswerable', reason='shop không ghi địa chỉ khách')"), h
assert "địa chỉ" not in question_hints("Tháng 3/2026 bán được bao nhiêu kg đường cát trắng?", D(2026, 6, 1), LUN, A)
A.addr_cols = ['T["ChiTietHoaDon"]["Addr"]']
assert "lọc cột địa chỉ" in question_hints("<REDACTED_COMPETITION_QUESTION>", D(2026, 6, 1), LUN, A)
print("OK address")


def test_public_failures():
    """Cac loi do duoc tren bo public (Kaggle 08/11): ky 'thang nay toi hom nay', ten vung mien, cau doi gia khong co chu 'gia',
    kieu dap an cua code goi y."""
    from datamate.hints import _expected_type, _guard_code, _money, _new_price, _norm, _price_q, _toks

    for q in ("<REDACTED_COMPETITION_QUESTION>",
              "<REDACTED_COMPETITION_QUESTION>",
              "<REDACTED_COMPETITION_QUESTION>"):
        assert codes(q, "2026-06-10") == ["THANG_NAY", "THANG_TRUOC_CUNG_KY"], (q, codes(q, "2026-06-10"))
    assert _toks("Lạc rang muối 200g") == _toks("Đậu phộng rang muối 200g")
    assert _toks("Má lợn nướng nước mắm") == _toks("Má heo nướng nước mắm")
    assert _toks("Mực rán nước mắm") == _toks("Mực chiên nước mắm")
    assert _toks("Cát xây trát") == _toks("Cát xây tô") and _toks("Pho mát Con Bò Cười") == _toks("Phô mai Con Bò Cười")
    assert _toks("Bò lúc lắc khoai tây") != _toks("Đậu phộng rang muối")
    s = _norm("<REDACTED_COMPETITION_QUESTION>")
    assert _price_q(s) and _new_price(s) == 26000
    assert _new_price(_norm("Từ khi Bạc xỉu (M) lên 35.000 đồng/ly, 14 ngày đầu bán nhiều hay ít ly hơn")) == 35000
    assert not _price_q(_norm("Doanh thu tháng 3/2025 tăng bao nhiêu so với tháng 2?"))
    assert _money("8.900.000", "dong") == 8900000 and _money("10", "cu") == 10_000_000 and _money("8,5", "tr") == 8_500_000
    assert _expected_type(_norm("<REDACTED_COMPETITION_QUESTION>")) == "date"
    assert _expected_type(_norm("<REDACTED_COMPETITION_QUESTION>")) == "percent"
    assert _expected_type(_norm("<REDACTED_COMPETITION_QUESTION>")) == "percent"
    assert _expected_type(_norm("Acyclovir Stella kem 5% tuýp 5g bán được bao nhiêu tuýp tháng 3?")) is None
    assert _expected_type(_norm("10/2025 dòng xe nào đem lại nhiều tiền sửa nhất?")) == "name"
    g = lambda code, q: _guard_code(code, _norm(q), [])
    assert g("answer('number', revenue(THANG(10, 2025)))", "<REDACTED_COMPETITION_QUESTION>") is None
    assert g("answer('number', n_tickets(THANG(3, 2026)))", "<REDACTED_COMPETITION_QUESTION>") is None
    assert g("answer('number', customer_pays(THANG(8, 2025)))", "08/2025 tien khach phai tra sau khi tru bot va tien may cu, binh quan mot don la may dong?") is None
    assert g("answer('number', revenue(THANG(10, 2025)))", "10/2025 dong xe mo dem lai nhieu tien sua nhat?") is None
    assert g("answer('string', top_items(THANG(11, 2025), k=1)['item'].iloc[0])", "<REDACTED_COMPETITION_QUESTION>") is None
    assert _guard_code("answer('string', top_items(THANG(11, 2025), k=1)['item'].iloc[0])", _norm("<REDACTED_COMPETITION_QUESTION>"),
                       [("group", "Trà")]) == "answer('string', top_items(THANG(11, 2025), k=1, group='Trà')['item'].iloc[0])"
    assert g("answer('number', revenue(THANG(10, 2025)))", "Doanh thu tháng 10/2025 bao nhiêu?") == "answer('number', revenue(THANG(10, 2025)))"
    print("OK public failures")


if __name__ == "__main__":
    test_public_failures()


def test_payment_aliases():
    """Ma thanh toan tieng Anh (de public dev_006, dev_008: cash/transfer/ewallet/cod/debt) khop cach goi trong cau hoi."""
    from datamate.hints import _payment_aliases

    vals = ["cash", "transfer", "ewallet", "cod", "debt"]
    E = EntityIndex({"payment": vals}, aliases=_payment_aliases(vals))
    assert ("payment", "ewallet") in E.match("<REDACTED_COMPETITION_QUESTION>")
    assert ("payment", "cod") in E.match("<REDACTED_COMPETITION_QUESTION>")
    assert ("payment", "transfer") in E.match("<REDACTED_COMPETITION_QUESTION>")
    assert not [m for m in E.match("<REDACTED_COMPETITION_QUESTION>") if m[0] == "payment"]
    G = EntityIndex({"group": ["Trà", "Cà phê"], "payment": ["Chuyển khoản"]})
    assert G.match("<REDACTED_COMPETITION_QUESTION>") == [("payment", "Chuyển khoản")]
    assert ("group", "Trà") in G.match("<REDACTED_COMPETITION_QUESTION>")
    print("OK payment aliases")


if __name__ == "__main__":
    test_payment_aliases()


def test_reason_payment():
    """'tra tien mat', 'tra vi dien tu' la thanh toan, khong phai cau hoi ly do tra hang (vi khong dau trung vì)."""
    import re
    from datamate.hints import _PAY_PHRASE, _norm

    rq = lambda q: bool(re.search(r"\b(huy|tra)\b", _PAY_PHRASE.sub(" ", _norm(q))) and re.search(
        r"\b(ly do|li do|vi sao|tai sao|nguyen nhan|vi gi|chu yeu vi)\b|\b(huy|tra|tra lai|doi)\b.*\bvi\b", _PAY_PHRASE.sub(" ", _norm(q))))
    assert not rq("<REDACTED_COMPETITION_QUESTION>")
    assert not rq("<REDACTED_COMPETITION_QUESTION>")
    assert rq("<REDACTED_COMPETITION_QUESTION>") and rq("<REDACTED_COMPETITION_QUESTION>")
    assert rq("<REDACTED_COMPETITION_QUESTION>")
    print("OK reason / payment")


if __name__ == "__main__":
    test_reason_payment()


def test_channel_pair():
    """Kenh ban ghi ma (store/phone/zalo_oa) khop 'tai quay', 'qua dien thoai'; cau hai so do cung cot -> list theo thu tu cau hoi."""
    from datamate.hints import _pair_code, _payment_aliases, _norm

    E = EntityIndex({"channel": ["store", "phone", "zalo_oa"]}, aliases=_payment_aliases(["store", "phone", "zalo_oa"], "channel"))
    q = "<REDACTED_COMPETITION_QUESTION>"
    sf = E.match(q, partial=False)
    assert sf == [("channel", "store"), ("channel", "phone")], sf
    ps = question_periods(q, "2026-07-22", LUN)
    assert _pair_code(_norm(q), ps, sf) == "answer('list', [revenue(THANG(6, 2025), channel='store'), revenue(THANG(6, 2025), channel='phone')])"
    assert _pair_code(_norm("6/2025 so don tai quay va qua dien thoai bao nhieu?"), ps, sf) is None  # dem don, khong phai doanh thu
    print("OK channel / pair")


if __name__ == "__main__":
    test_channel_pair()


def test_no_false_code_hints():
    """Cau co dieu kien ma dong code khong the hien (thu, ngay trong thang, dau ky, phu dinh, kenh, gioi tinh, ca, qua han,
    tong hai gia tri) -> khong sinh code goi y; hoi theo don vi dong goi -> code co kiem unit_is_base (sai don vi thi khong nop)."""
    import re
    from datamate.kernel import load_business_cached

    lun = LunarCalendar.load("data/train/calendar.csv")
    idx = EntityIndex.from_business(load_business_cached("data/train/data/biz_106", "/tmp/datamate_cache_dev"))
    from datamate.hints import question_hints

    no_code = ["doanh thu thứ 7 tuần trước là bao nhiêu", "doanh thu ngày 15 tháng 5 là bao nhiêu",
               "10 ngày đầu tháng 11/2025 doanh thu bao nhiêu", "tuần đầu tháng 11/2025 bán được bao nhiêu tiền",
               "3 tháng đầu năm 2025 doanh thu bao nhiêu", "doanh thu ngoài nhóm trà tháng 11/2025 là bao nhiêu",
               "doanh thu tháng 11/2025 không tính nhóm trà là bao nhiêu", "tỉ lệ đơn huỷ online tháng 11/2025 là bao nhiêu phần trăm",
               "tháng 11/2025 nhân viên nào có doanh thu cao nhất ca sáng",
               "tổng doanh thu tiền mặt và chuyển khoản tháng 11/2025 là bao nhiêu", "số đơn chủ nhật tuần trước là bao nhiêu"]
    for q in no_code:
        h = question_hints(q, "2026-01-10", lun, idx)
        m = re.search(r"^Code gợi ý \([^)]*\): (.*)$", h, re.M)
        assert not m, (q, m.group(1))
    # shop khong ghi gioi tinh khach: §7 khong tra loi duoc (train: gold unanswerable)
    h = question_hints("tháng 11/2025 khách nam nào chi nhiều tiền nhất", "2026-01-10", lun, idx)
    assert "answer('unanswerable', reason='shop không ghi giới tính khách')" in h, h
    h = question_hints("tháng 11/2025 bán được bao nhiêu thùng Matcha latte (L)", "2026-01-10", lun, idx)
    assert "unit_is_base('Matcha latte (L)', 'thùng')" in h, h
    print("OK no false code hints")


if __name__ == "__main__":
    test_no_false_code_hints()
