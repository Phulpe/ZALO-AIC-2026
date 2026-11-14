"""khang_v5 --all-samples: may gia luon tra cung mot dap an. Khong co co -> dung som sau 2 mau (2 phieu da chot);
co co -> chay du 3 mau. Dap an cuoi giong nhau o ca hai che do. Thu muc tam dat theo TEST_TMP (de giu du lieu trong ZAIC)."""
import http.server
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPLY = "```python\nanswer('number', 12345)\n```"


class _Srv(http.server.BaseHTTPRequestHandler):
    calls = 0

    def do_GET(self):
        self._send({"data": [{"id": "qwen3.5-9b"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).calls += 1
        self._send({"id": "x", "object": "chat.completion", "model": "qwen3.5-9b",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": REPLY}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


def _run(extra, tmp, srv):
    from datamate.runner import main

    nohint_id = "q_238225519d3a"
    idf = os.path.join(tmp, "ids.txt")
    open(idf, "w").write(nohint_id + "\n")
    out = os.path.join(tmp, "r" + str(len(extra)))
    rc = main(["--data", "data/train", "--out", out, "--cache", os.path.join(tmp, "cache"),
               "--base-urls", f"http://127.0.0.1:{srv.server_port}", "--ids", idf, "--workers", "1", "--samples", "3",
               "--min-votes", "2", "--progress-every", "3600"] + extra)
    assert rc in (0, None), rc
    return json.load(open(os.path.join(out, "logs", f"{nohint_id}.json"), encoding="utf-8"))


def test_all_samples():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Srv)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMP")) as tmp:
        a = _run([], tmp, srv)
        b = _run(["--all-samples"], tmp, srv)
    srv.shutdown()
    assert len(a["samples"]) == 2, len(a["samples"])
    assert len(b["samples"]) == 3, len(b["samples"])
    assert a["answer"] == b["answer"] and b["answer"].get("answer") == 12345, (a["answer"], b["answer"])
    print("OK runner all-samples")


if __name__ == "__main__":
    test_all_samples()
