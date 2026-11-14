"""khang_v3: moi cau mot dong PROG (stdout) + PROGRESS.tsv; --fixed-samples/--parallel-first chay du so lan dau dong thoi;
cau co code goi y van nop thang (khong goi model); loi model -> abstain, khong lam hong cau khac."""

import contextlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class _Models(http.server.BaseHTTPRequestHandler):
    """/v1/models tra model; moi POST tra 400 (runner bao loi ngay, khong thu lai)."""
    posts = 0

    def do_GET(self):
        body = json.dumps({"data": [{"id": "qwen3.5-9b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).posts += 1
        self.send_response(400)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_progress_and_parallel_first():
    from datamate.runner import main

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Models)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    hint_ids = ("q_15fd3e0d0363", "q_3379b46319a8", "q_8848e665f8d1")  # cau co code goi y (test_runner_direct)
    nohint_id = "q_238225519d3a"  # cau train khong co code goi y
    ids = hint_ids + (nohint_id,)
    with tempfile.TemporaryDirectory() as out:
        idf = os.path.join(out, "ids.txt")
        open(idf, "w").write("\n".join(ids) + "\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["--data", "data/train", "--out", os.path.join(out, "r"), "--cache", "/tmp/datamate_cache_dev",
                       "--base-urls", f"http://127.0.0.1:{srv.server_port}", "--ids", idf, "--workers", "4",
                       "--samples", "3", "--min-votes", "2", "--fixed-samples", "--parallel-first", "2",
                       "--question-timeout", "600", "--progress-every", "3600"])
        assert rc in (0, None), rc
        prog = [l for l in buf.getvalue().splitlines() if l.startswith("PROG | ")]
        assert len(prog) == len(ids), prog
        assert {l.split(" | ")[2] for l in prog} == set(ids)
        assert sorted(int(l.split(" | ")[1].split("/")[0]) for l in prog) == [1, 2, 3, 4]
        by_id = {l.split(" | ")[2]: l for l in prog}
        for i in hint_ids:
            assert "src=hint" in by_id[i] and "| answered |" in by_id[i], by_id[i]
        assert "src=llm" in by_id[nohint_id] and "| abstain |" in by_id[nohint_id] and "samples=2" in by_id[nohint_id], by_id[nohint_id]
        rows = open(os.path.join(out, "r", "PROGRESS.tsv"), encoding="utf-8").read().splitlines()
        assert rows[0].split("\t")[:3] == ["idx", "total", "id"] and len(rows) == 1 + len(ids), rows
        sub = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(out, "r", "submission.jsonl"), encoding="utf-8")}
        assert sub[nohint_id]["type"] == "abstain" and all(sub[i]["type"] != "abstain" for i in hint_ids), sub
        # 2 lan dau chay dong thoi, ca hai loi -> dung (khong the dat 2 phieu): it nhat 2 POST, khong qua 3
        assert 2 <= _Models.posts <= 3, _Models.posts
        # khong in noi dung cau hoi ra dong tien do
        q = {json.loads(l)["id"]: json.loads(l)["question"] for l in open("data/train/questions.jsonl", encoding="utf-8")}
        assert not any(q[i][:30] in by_id[i] for i in ids)
    srv.shutdown()
    print("OK runner v3 progress + parallel-first")


if __name__ == "__main__":
    test_progress_and_parallel_first()
