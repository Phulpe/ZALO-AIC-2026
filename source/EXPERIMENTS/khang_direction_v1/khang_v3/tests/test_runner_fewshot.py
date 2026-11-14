"""Runner --fewshot: cau khong co code goi y duoc them vi du vao goi y (log fewshot_n, prompt gui model co 'Ví dụ câu hỏi tương tự');
cau co code goi y van nop thang, khong goi model. Ngan hang gia lap."""
import http.server
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class _Srv(http.server.BaseHTTPRequestHandler):
    bodies = []

    def do_GET(self):
        b = json.dumps({"data": [{"id": "qwen3.5-9b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        type(self).bodies.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0))))
        self.send_response(400)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_runner_fewshot():
    from datamate.runner import main

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Srv)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    hint_id, nohint_id = "q_15fd3e0d0363", "q_238225519d3a"
    with tempfile.TemporaryDirectory() as out:
        bank = os.path.join(out, "bank.jsonl")
        rows = [{"id": f"x{i}", "business_id": "s", "question": f"Doanh thu bán hàng tháng {i}/2025 là bao nhiêu tiền?",
                 "code": f"answer('number', revenue(M_2025_{i:02d}))", "fn": f"f{i}"} for i in range(1, 6)]
        open(bank, "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        idf = os.path.join(out, "ids.txt")
        open(idf, "w").write(hint_id + "\n" + nohint_id + "\n")
        rc = main(["--data", "data/train", "--out", os.path.join(out, "r"), "--cache", "/tmp/datamate_cache_dev",
                   "--base-urls", f"http://127.0.0.1:{srv.server_port}", "--ids", idf, "--workers", "2", "--samples", "1",
                   "--min-votes", "1", "--fewshot", bank, "--fewshot-k", "3", "--progress-every", "3600"])
        assert rc in (0, None), rc
        lh = json.load(open(os.path.join(out, "r", "logs", f"{hint_id}.json"), encoding="utf-8"))
        assert lh["source"] == "code_hint", lh.get("source")
        ln = json.load(open(os.path.join(out, "r", "logs", f"{nohint_id}.json"), encoding="utf-8"))
        assert ln["fewshot_n"] >= 1, ln.get("fewshot_n")
        assert _Srv.bodies, "model khong duoc goi cho cau khong co code goi y"
        user = [m["content"] for m in _Srv.bodies[0]["messages"] if m["role"] == "user"][0]
        assert "Ví dụ câu hỏi tương tự" in user and "Code: answer(" in user
    srv.shutdown()
    print("OK runner fewshot")


if __name__ == "__main__":
    test_runner_fewshot()
