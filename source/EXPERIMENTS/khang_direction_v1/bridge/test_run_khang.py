import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_khang as R

class T(unittest.TestCase):
    def test_strip_only_code_line(self):
        h = "Kỳ: 2025-01-01..2025-01-31\nCode gợi ý (doanh thu): answer('number', revenue(S, a, b))\nTên hàng: X"
        out = R.strip_code_hint(h)
        self.assertNotIn("Code gợi ý", out)
        self.assertIn("Kỳ:", out); self.assertIn("Tên hàng", out)

    def test_noop_without_code(self):
        self.assertEqual(R.strip_code_hint("Kỳ: a"), "Kỳ: a")

    def test_prerun_regex_no_longer_matches(self):
        import re
        h = R.strip_code_hint("x\nCode gợi ý (t): answer('n', 1)\ny")
        self.assertIsNone(re.search(r"^Code gợi ý \([^)]*\): (.*)$", h, re.M))

    def test_logging_client(self):
        import json, tempfile
        class Rep:
            content, reasoning, finish_reason, completion_tokens = "c", None, "stop", 3
        class Base:
            def __init__(self, *a, **k): pass
            def chat(self, messages, **kw): return Rep()
        p = os.path.join(tempfile.mkdtemp(), "t.jsonl")
        C = R.make_logging_client(Base, p)
        r = C().chat([{"role": "user", "content": "q"}], seed=7, temperature=0.6)
        self.assertEqual(r.content, "c")
        rec = json.loads(open(p, encoding="utf-8").read())
        self.assertEqual(rec["seed"], 7); self.assertEqual(rec["messages"][0]["content"], "q")
        self.assertEqual(rec["reply"]["content"], "c")


if __name__ == "__main__":
    unittest.main()
