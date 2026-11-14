"""v4 (12/11/2026): do ben hint voi loi dem va 'phan tram'. Du lieu gia lap (khong dung cau hoi that).
- ten nhom mot tu co dau ('Điện') van khop khi cau go khong dau nhung co loi dem co dau ('Cho mình hỏi ... dien ... ạ')
- van KHONG khop 'Trà' khi cau ghi 'trả' (co dau, khac nghia)
- 'phan tram' chuan hoa thanh '%' dinh vao tu truoc.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_hints_v4_robust():
    from datamate.hints import EntityIndex, _norm

    idx = EntityIndex({"group": ["Điện", "Trà"]})
    assert ("group", "Điện") in idx.match("doanh thu nhom dien thang 3/2025 bao nhieu")
    assert ("group", "Điện") in idx.match("Cho mình hỏi doanh thu nhom dien thang 3/2025 bao nhieu ạ")
    assert ("group", "Điện") in idx.match("Doanh thu nhóm điện tháng 3/2025 bao nhiêu?")
    assert ("group", "Trà") not in idx.match("Khách trả bao nhiêu tiền tháng 3/2025?")
    assert _norm("Tăng bao nhiêu phần trăm?") == "tang bao nhieu%"
    assert _norm("giảm 5 phần trăm") == "giam 5%"
    assert _norm("Tăng bao nhiêu %?") == "tang bao nhieu %"
    print("OK hints v4 robust")


if __name__ == "__main__":
    test_hints_v4_robust()
