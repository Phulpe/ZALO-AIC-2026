"""May chu gia cho buoc chi-hint: /v1/models tra model; moi POST tra 400 (runner nem loi ngay, khong retry) -> cau khong co code hint thanh abstain."""
import http.server, json, sys, threading


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps({"data": [{"id": "qwen3.5-9b"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"hint_stage_no_llm"}')

    def log_message(self, *a):
        pass


def start(port):
    s = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


if __name__ == "__main__":
    start(int(sys.argv[1]))
    threading.Event().wait()
