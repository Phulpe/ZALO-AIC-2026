"""In tom tat ket qua nap du lieu cua mot hoac nhieu cua hang (de kiem tra loader)."""
import sys, time, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.loader import load_business

for path in sys.argv[1:]:
    t = time.time()
    b = load_business(path)
    print(f"=== {b.id} ({time.time()-t:.1f}s) end={b.data_end} warn={b.warnings}")
    for name, df in b.tables.items():
        inf = b.info[name]
        kinds = {c: ci.kind for c, ci in inf.columns.items()}
        low = {c: round(ci.parse_rate, 3) for c, ci in inf.columns.items() if ci.parse_rate < 1}
        print(f"  {name}: {len(df)} rows, src={len(inf.sources)} hdr={sorted(set(inf.header_rows))} key={inf.key} dup={inf.duplicates_dropped} ffill={inf.ffilled}")
        print("     ", kinds)
        if low:
            print("      LOW PARSE:", low, {c: ci.failed[:3] for c, ci in inf.columns.items() if ci.failed})
