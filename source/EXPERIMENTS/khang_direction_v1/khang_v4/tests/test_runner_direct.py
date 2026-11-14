"""Runner: cau co Code goi y chay ra dap an thi nop thang, khong goi model; log co du truong BTC yeu cau (muc 3.3)."""

import http.server
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class _Models(http.server.BaseHTTPRequestHandler):
    """May chu gia: chi tra /v1/models (buoc cho san sang). Moi loi goi model tra 500, nen cau nao goi model se bi lo ra."""

    def do_GET(self):
        body = json.dumps({"data": [{"id": "qwen3-14b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.send_response(500)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_runner_direct_code_hint():
    from datamate.runner import main

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Models)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ids = ("q_15fd3e0d0363", "q_3379b46319a8", "q_8848e665f8d1")  # doanh thu mot mat hang, ti le huy, so ly theo don vi
    gold = {json.loads(l)["id"]: json.loads(l) for l in open("data/train/answers.jsonl", encoding="utf-8")}
    with tempfile.TemporaryDirectory() as out:
        idf = os.path.join(out, "ids.txt")
        open(idf, "w").write("\n".join(ids) + "\n")
        rc = main(["--data", "data/train", "--out", os.path.join(out, "r"), "--cache", "/tmp/datamate_cache_dev",
                   "--base-urls", f"http://127.0.0.1:{srv.server_port}", "--ids", idf, "--workers", "2"])
        assert rc in (0, None), rc
        sub = [json.loads(l) for l in open(os.path.join(out, "r", "submission.jsonl"), encoding="utf-8")]
        assert {r["id"] for r in sub} == set(ids), sub
        from datamate.answers import score

        for r in sub:
            assert score(r, gold[r["id"]]) == 1, (r, gold[r["id"]])
            log = json.load(open(os.path.join(out, "r", "logs", f"{r['id']}.json"), encoding="utf-8"))
            assert log["source"] == "code_hint" and log["thinking"] is None and log["code"] and log["output"], log
            assert isinstance(log["elapsed_ms"], int) and log["elapsed_ms"] >= 0, log
    srv.shutdown()
    print("OK runner direct code hint")


if __name__ == "__main__":
    test_runner_direct_code_hint()
