"""fewshot: truy hoi tat dinh, bo chinh cau dang hoi, gioi han moi ham, uu tien cung cua hang; van ban vi du dung dinh dang.
Du lieu gia lap (khong dung cau hoi that)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_fewshot():
    from datamate.fewshot import FewShotBank, fewshot_text

    rows = [
        {"id": "a1", "business_id": "s1", "question": "Doanh thu tháng 3/2025 của shop là bao nhiêu?", "code": "answer('number', revenue(M_2025_03))", "fn": "revenue"},
        {"id": "a2", "business_id": "s2", "question": "Doanh thu tháng 4/2025 là bao nhiêu tiền?", "code": "answer('number', revenue(M_2025_04))", "fn": "revenue"},
        {"id": "a3", "business_id": "s1", "question": "Doanh thu quý 1/2025 bao nhiêu?", "code": "answer('number', revenue(Q_2025_1))", "fn": "revenue"},
        {"id": "b1", "business_id": "s3", "question": "Có bao nhiêu đơn bị huỷ trong tháng 5/2025?", "code": "answer('number', n_cancelled(M_2025_05))", "fn": "n_cancelled"},
        {"id": "c1", "business_id": "s3", "question": "Tỉ lệ huỷ đơn tháng 5/2025 là bao nhiêu phần trăm?", "code": "answer('percent', cancel_rate(M_2025_05))", "fn": "cancel_rate"},
    ]
    bank = FewShotBank(rows)
    q = "Doanh thu tháng 6/2025 của cửa hàng là bao nhiêu?"
    top = bank.top(q, "s9", k=3)
    assert [r["fn"] for r in top][:2] == ["revenue", "revenue"], top  # toi da 2 vi du moi ham
    assert all(r["fn"] != "revenue" for r in top[2:])
    assert bank.top(q, "s9", k=3) == top  # tat dinh
    assert all(r["id"] != "a1" for r in bank.top(rows[0]["question"], "s1", k=3, exclude_id="a1"))
    # uu tien cung cua hang khi diem gan bang
    t2 = bank.top("Doanh thu tháng 4/2025 là bao nhiêu?", "s1", k=1, max_per_fn=1, same_biz_bonus=1.0)
    assert t2[0]["business_id"] == "s1", t2
    q3 = bank.top("Tỉ lệ huỷ đơn tháng 7/2025 bao nhiêu %?", None, k=1)
    assert q3[0]["fn"] == "cancel_rate", q3
    txt = fewshot_text(top)
    assert txt.startswith("Ví dụ câu hỏi tương tự") and txt.count("\n- Hỏi: ") == len(top) and "Code: answer(" in txt
    assert fewshot_text([]) == ""
    assert bank.top("xyz qqq", None, k=3) == []
    print("OK fewshot")


if __name__ == "__main__":
    test_fewshot()
