"""Doi chieu lop lam sach du lieu: tu tinh 'doanh thu thang M/Y' theo rulebook §1 tren bang dong hang
(chon cot theo mo ta README) va so voi dap an train. Khong dung model."""
import json, os, re, sys, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from datamate.kernel import load_business_cached
from datamate.helpers import status_of
from datamate.parsing import strip_accents

DATA = "data/train"
Q = [json.loads(l) for l in open(f"{DATA}/questions.jsonl")]
G = {g["id"]: g for g in map(json.loads, open(f"{DATA}/answers.jsonl"))}
pat = re.compile(r"^(?:tong |tổng )?(?:doanh thu|dthu)\s*(?:cua (?:cua hang|shop) )?(?:trong )?thang (\d{1,2})[/ ](?:nam )?(\d{4})")


def plain(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")


def find(df, info, *descs):
    for c in df.columns:
        ci = info.columns.get(c)
        if ci and ci.description and any(ci.description.lower().startswith(d) for d in descs):
            return c
    return None


rows = []
for q in Q:
    s = plain(q["question"])
    m = re.search(r"(doanh thu|dthu)[^0-9]{0,40}thang (\d{1,2})[/ ](?:nam )?(\d{4})", s)
    if not m or G[q["id"]]["type"] != "number" or any(k in s for k in ("nhom", "mat hang", "khach", "ca ", "kenh", "nhan vien", "ngay", "tuan", "%", "trung binh", "so voi", "chenh", "tang", "giam", "moi")):
        continue
    mm, yy = int(m.group(2)), int(m.group(3))
    b = load_business_cached(os.path.join(DATA, "data", q["business_id"]), "/tmp/datamate_cache_dev")
    got, why = None, ""
    for t, df in b.tables.items():
        inf = b.info[t]
        amt = find(df, inf, "thành tiền sau giảm", "thành tiền của dòng")
        date = find(df, inf, "ngày bán", "thời điểm đặt đơn")
        st = find(df, inf, "trạng thái đơn")
        oid = find(df, inf, "mã đơn", "số hoá đơn", "số thứ tự đơn")
        qty = find(df, inf, "số lượng")
        if amt and date and st:
            d = df.copy()
            d["_st"] = d[st].map(status_of)
            if oid:
                bad = d.groupby(oid)["_st"].transform(lambda x: x.isin(["huỷ", "trả hàng"]).any())
                if qty:
                    bad |= d.groupby(oid)[qty].transform(lambda x: (x < 0).any())
                odate = d.groupby(oid)[date].transform("min")
            else:
                bad = d["_st"].isin(["huỷ", "trả hàng"])
                odate = d[date]
            sel = (~bad) & (odate.dt.year == yy) & (odate.dt.month == mm)
            got = int(round(d.loc[sel, amt].sum()))
            unk = d["_st"].isna().sum()
            why = f"{t} amt={amt} date={date} st={st} oid={oid} unknown_status={unk}"
            break
    if got is None:
        continue
    gold = G[q["id"]]["answer"]
    rows.append((q["business_id"], got == gold, got, gold, q["question"][:60], why))
ok = sum(r[1] for r in rows)
print(f"{ok}/{len(rows)} khop")
for r in rows:
    if not r[1]:
        print("SAI", r)
