"""Quet toan bo o goc trong du lieu, dem cac mau ghi so co k/tr/ty kem dau phan cach (de chot cach doc)."""
import sys, os, re, glob, collections, csv, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.loader import _read_xlsx, _read_csv
pat = re.compile(r"^-?\d+[.,]\d+\s*(k|tr|triệu|tỷ|tỉ)\s*\d*$", re.I)
pat2 = re.compile(r"^-?\d+\s*(k|tr|triệu|tỷ)\s*\d+$", re.I)
c = collections.Counter(); ex = collections.defaultdict(list)
for f in glob.glob("data/*/data/*/*"):
    if f.endswith(".xlsx"):
        sheets = _read_xlsx(f)
    elif f.endswith(".csv"):
        sheets = [(None, _read_csv(f))]
    else:
        continue
    for _, rows in sheets:
        for r in rows:
            for v in r:
                if isinstance(v, str):
                    s = v.strip().lower()
                    for name, p in (("sep+unit", pat), ("unit+rest", pat2)):
                        if p.match(s):
                            key = (name, re.sub(r"\d", "9", s))
                            c[key] += 1
                            if len(ex[key]) < 3: ex[key].append((os.path.basename(os.path.dirname(f)), v))
for k, n in c.most_common(40):
    print(n, k, ex[k])
