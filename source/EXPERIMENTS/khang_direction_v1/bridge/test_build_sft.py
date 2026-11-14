import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sft as B
import train_lora as TL


def rec(qid, k, step, n_asst, content, finish="stop"):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": f"P\n- id: {qid}\n- x"}]
    for i in range(n_asst):
        msgs += [{"role": "assistant", "content": f"a{i}"}, {"role": "user", "content": f"o{i}"}]
    return {"seed": B.qseed(qid, 1234) + 1000 * k + step, "messages": msgs,
            "reply": {"content": content, "finish_reason": finish}}


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, "run", "logs"))
        recs = [rec("Q1", 0, 1, 0, "c1"), rec("Q1", 0, 2, 1, "c2"),  # 2 buoc, dung
                rec("Q1", 1, 1, 0, "w"),                             # sample 1 sai
                rec("Q2", 0, 1, 0, "t", "length"), rec("Q2", 0, 2, 1, "t2"),  # bi cat -> loai
                rec("Q3", 0, 1, 0, "z")]                             # khong co gold
        self.traj = os.path.join(self.d, "t.jsonl")
        open(self.traj, "w").write("".join(json.dumps(r) + "\n" for r in recs))
        logs = {"Q1": [{"answer": {"v": 1}, "steps": [{"code": "x"}, {"code": "y"}]},
                       {"answer": {"v": 2}, "steps": [{"code": "z"}]}],
                "Q2": [{"answer": {"v": 1}, "steps": [{"code": "x"}]}],
                "Q3": [{"answer": {"v": 1}, "steps": [{"code": "x"}]}]}
        for q, s in logs.items():
            json.dump({"samples": s}, open(os.path.join(self.d, "run", "logs", q + ".json"), "w"))
        self.gold = os.path.join(self.d, "g.jsonl")
        open(self.gold, "w").write(json.dumps({"id": "Q1", "v": 1}) + "\n" + json.dumps({"id": "Q2", "v": 1}) + "\n")

    def test_build(self):
        out = os.path.join(self.d, "sft.jsonl")
        st = B.build(self.traj, os.path.join(self.d, "run"), self.gold, out,
                     score_fn=lambda p, g: 1 if p["v"] == g["v"] else -1)
        rows = [json.loads(l) for l in open(out)]
        self.assertEqual([(r["qid"], r["k"]) for r in rows], [("Q1", 0)])
        self.assertEqual(rows[0]["messages"][-1], {"role": "assistant", "content": "c2"})
        self.assertEqual(len(rows[0]["messages"]), 5)
        self.assertEqual(st["truncated"], 1); self.assertEqual(st["wrong"], 1); self.assertEqual(st["no_gold_or_log"], 1)

    def test_items_mask(self):
        class Tok:
            eos_token = "<e>"
            def apply_chat_template(self, m, **k): return "|".join(x["content"] for x in m) + "|A:"
            def __call__(self, s, add_special_tokens=False): return {"input_ids": [ord(c) for c in s]}
        rows = [{"qid": "Q1", "k": 0, "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "ok"}]},
                {"qid": "Q1", "k": 1, "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "ok2"}]},
                {"qid": "Q1", "k": 2, "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "ok3"}]}]
        items, sk = TL.build_items(rows, Tok(), 100, 2, 1)
        self.assertEqual(len(items), 2)
        it = [i for i in items if len(i["labels"]) == len("u|A:") + len("ok<e>")][0]
        self.assertEqual(it["labels"][:4], [-100] * 4)
        self.assertEqual(it["labels"][4:], [ord(c) for c in "ok<e>"])
        items, sk = TL.build_items(rows, Tok(), 5, 3, 1)
        self.assertEqual((len(items), sk), (0, 3))


if __name__ == "__main__":
    unittest.main()
