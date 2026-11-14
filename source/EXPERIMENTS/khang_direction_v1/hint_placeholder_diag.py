"""Dem loai cho trong con trong dong 'Dang so do' cua cau no-hint (chi tong hop)."""
import sys, os, json, re, datetime, collections, warnings
warnings.filterwarnings("ignore")
khang, data_root, reasons = sys.argv[1:4]
sys.path.insert(0, os.path.join(khang, "src")); os.chdir(khang)
from datamate import hints as H
from datamate.kernel import load_business_cached
from datamate.dates import LunarCalendar
ids = {json.loads(l)["id"] for l in open(reasons) if "measure_but_no_code" in l}
lun = LunarCalendar.load(f"{data_root}/calendar.csv")
qs = [q for q in map(json.loads, open(f"{data_root}/questions.jsonl", encoding="utf-8")) if q["id"] in ids]
c = collections.Counter(); idxs = {}
for q in qs:
    b = q["business_id"]
    if b not in idxs:
        idxs[b] = H.EntityIndex.from_business(load_business_cached(f"{data_root}/data/{b}", "/tmp/datamate_cache_khang_direction_v1"))
    t = H.question_hints(q["question"], datetime.date.fromisoformat(q["as_of_date"]), lun, idxs[b])
    mh = t.split("Dạng số đo:", 1)[1] if "Dạng số đo:" in t else ""
    tags = set()
    if "<cột>" in mh or "<lọc>" in mh: tags.add("filter_placeholder")
    if "..." in mh: tags.add("ellipsis_arg")
    if "<tho>" in mh: tags.add("worker_placeholder")
    if re.search(r"\((kỳ|ngày|D)[,)]", mh): tags.add("period_placeholder")
    n = mh.count("\n- ")
    tags.add(f"n_measures={min(n,3)}")
    for x in tags: c[x] += 1
print(len(qs), dict(c.most_common()))
