"""Ham chi so (metrics.py) phai cho dung dap an train da biet."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.helpers import pct
from datamate.kernel import load_business_cached
from datamate.metrics import Metrics

CACHE = "/tmp/datamate_cache_dev"
G = {g["id"]: g for g in map(json.loads, open("data/train/answers.jsonl"))}


def biz(b):
    x = load_business_cached(f"data/train/data/{b}", CACHE)
    return Metrics(x.sales.S, x.sales.O), x


def main():
    m, b = biz("biz_112")
    share = pct(m.revenue("2025-08-01", "2025-08-31", group="cà phê"), m.revenue("2025-08-01", "2025-08-31"))
    assert share == G["q_3400f2bffb00"]["answer"] == 35.17, share
    m, _ = biz("biz_106")
    assert m.revenue("2025-05-01", "2025-05-31") == 271866000
    m, _ = biz("biz_128")
    assert m.top_items("2026-06-01", "2026-06-30", 1).shape[0] == 1
    print("OK", share)


if __name__ == "__main__":
    main()


def test_answer_table_forms():
    """answer('table', ...) nhan DataFrame, list dict, rows la dict / DataFrame / tuple (pub15: rows dict bi doi thanh ten cot)."""
    import pandas as pd
    from datamate.answers import normalize

    df = pd.DataFrame({"tho": ["Đàm Lan An", "Hà Loan Đạt"], "hoa_hong": [9296595, 7548960.0]})
    want = {"type": "table", "answer": {"columns": ["tho", "hoa_hong"], "rows": [["Đàm Lan An", 9296595], ["Hà Loan Đạt", 7548960]]}}
    for v in (df, df.to_dict("records"), {"columns": ["tho", "hoa_hong"], "rows": df.to_dict("records")},
              {"columns": ["tho", "hoa_hong"], "rows": df}, {"columns": ["tho", "hoa_hong"], "rows": list(df.itertuples(index=False))}):
        assert normalize("table", v) == want, v
    print("OK answer table")


if __name__ == "__main__":
    test_answer_table_forms()
