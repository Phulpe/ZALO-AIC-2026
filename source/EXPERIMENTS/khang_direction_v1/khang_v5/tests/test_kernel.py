"""Kiem tra sandbox: chay code, giu bien trong phien, cach ly giua cac phien, timeout, answer()."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datamate.kernel import BusinessKernel


def main():
    k = BusinessKernel("data/train/data/biz_128", "/tmp/datamate_cache_dev", "data/train/calendar.csv")
    t = time.time()
    r = k.execute("s1", "names = list(T)\nlen(names)", "2026-09-07")
    print("start+exec", round(time.time() - t, 1), "s", r)
    assert r["error"] is None
    r = k.execute("s1", "x = 1\nwhile True:\n    x += 1", "2026-09-07", soft_timeout=2)
    assert "TimeoutError" in r["error"], r
    r = k.execute("s1", "T[names[0]]['moi'] = 1\nanswer('number', 12.0)", "2026-09-07")
    assert r["answer"] == {"type": "number", "answer": 12}, r
    r = k.execute("s2", "print('moi' in T[list(T)[0]].columns)\nprint(tet_range(2026))\nanswer('table', {'columns':['a','b'],'rows':[['x',1.0]]})", "2026-09-07")
    print(r)
    assert "False" in r["stdout"], "phien s2 bi anh huong boi s1"
    r = k.execute("s3", "answer('percent', 'abc')", "2026-09-07")
    assert r["answer"] is None and "sai format" in r["error"], r
    k.close()
    print("OK")


if __name__ == "__main__":
    main()
