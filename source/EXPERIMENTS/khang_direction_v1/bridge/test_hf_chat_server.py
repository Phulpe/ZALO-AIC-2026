"""Test CPU cho hf_chat_server: khong can GPU/model. Dung LLMClient that cua Khang de goi qua HTTP."""
import os
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "khang", "src"))
import hf_chat_server as S  # noqa: E402
from datamate.llm import LLMClient  # noqa: E402


class Pure(unittest.TestCase):
    def test_split_thinking(self):
        self.assertEqual(S.split_thinking("<think>a b</think>\n```python\nx\n```"), ("a b", "```python\nx\n```"))
        self.assertEqual(S.split_thinking("tu mo san</think>kq"), ("tu mo san", "kq"))
        self.assertEqual(S.split_thinking("<think>bi cat"), ("bi cat", ""))
        self.assertEqual(S.split_thinking("  khong suy nghi "), (None, "khong suy nghi"))

    def test_truncate_stop(self):
        self.assertEqual(S.truncate_stop("abc<END>def", ["<END>"]), ("abc", True))
        self.assertEqual(S.truncate_stop("abXcYd", ["Y", "X"]), ("ab", True))
        self.assertEqual(S.truncate_stop("abc", None), ("abc", False))

    def test_format(self):
        r = S.format_response("m", "c", None, "stop", 3, 2)
        self.assertNotIn("reasoning_content", r["choices"][0]["message"])
        self.assertEqual(r["usage"]["total_tokens"], 5)


class FakeGen:
    def __init__(self, delay=0.05):
        self.batches = []
        self.delay = delay

    def __call__(self, batch):
        time.sleep(self.delay)
        self.batches.append([j.body["messages"][-1]["content"] for j in batch])
        for j in batch:
            last = j.body["messages"][-1]["content"]
            if last == "boom":
                raise RuntimeError("cuda oom gia")
            text = f"<think>nghi {last}</think>tra loi {last}"
            reasoning, content = S.split_thinking(text)
            j.result = S.format_response("fake", content, reasoning, "stop", 10, 4)


class Http(unittest.TestCase):
    def setUp(self):
        self.gen = FakeGen()
        self.b = S.Batcher(self.gen, max_batch=4, wait_s=0.1)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(self.b, "fake", 30))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()

    def test_khang_client_roundtrip_and_batching(self):
        c = LLMClient([self.url], "fake", timeout=30, backend="vllm")
        c.wait_ready(10)
        out = {}

        def call(i):
            out[i] = c.chat([{"role": "user", "content": f"q{i}"}], temperature=0.6, top_p=0.95, top_k=20,
                            max_tokens=64, seed=i)
        ts = [threading.Thread(target=call, args=(i,)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        for i in range(8):
            self.assertEqual(out[i].content, f"tra loi q{i}")
            self.assertEqual(out[i].reasoning, f"nghi q{i}")
            self.assertEqual(out[i].finish_reason, "stop")
        self.assertLessEqual(max(len(x) for x in self.gen.batches), 4)
        self.assertGreater(max(len(x) for x in self.gen.batches), 1, "khong gom duoc batch")

    def test_different_sampling_params_not_mixed(self):
        c = LLMClient([self.url], "fake", timeout=30, backend="vllm")
        res = {}
        t1 = threading.Thread(target=lambda: res.setdefault("a", c.chat([{"role": "user", "content": "a"}], temperature=0.0)))
        t2 = threading.Thread(target=lambda: res.setdefault("b", c.chat([{"role": "user", "content": "b"}], temperature=0.6)))
        t1.start(); t2.start(); t1.join(); t2.join()
        for batch in self.gen.batches:
            self.assertFalse("a" in batch and "b" in batch)

    def test_batch_error_reported_server_survives(self):
        c = LLMClient([self.url], "fake", timeout=5, backend="vllm")
        with self.assertRaises(RuntimeError):
            c.chat([{"role": "user", "content": "boom"}])
        self.assertEqual(c.chat([{"role": "user", "content": "ok"}]).content, "tra loi ok")

    def test_bad_request(self):
        import urllib.request, urllib.error
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=b"{}", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 400)

    def test_binds_localhost_only(self):
        self.assertEqual(self.srv.server_address[0], "127.0.0.1")

class OOMSplit(unittest.TestCase):
    def test_split_until_fits(self):
        import hf_chat_server as S
        class OOM(Exception): pass
        calls, st = [], {}
        def gen(b):
            calls.append(len(b))
            if len(b) > 2: raise OOM()
            for j in b: j.result = {"ok": 1}
        jobs = [S.Job({"messages": [1]}) for _ in range(8)]
        S.split_on_oom(gen, lambda e: isinstance(e, OOM), None, st)(jobs)
        self.assertTrue(all(j.result for j in jobs))
        self.assertEqual(calls, [8, 4, 2, 2, 4, 2, 2]); self.assertEqual(st["oom_splits"], 3)

    def test_single_oom_marks_error_only(self):
        import hf_chat_server as S
        class OOM(Exception): pass
        def gen(b):
            if any(j.body.get("big") for j in b): raise OOM()
            for j in b: j.result = {"ok": 1}
        jobs = [S.Job({"messages": [1], "big": i == 0}) for i in range(2)]
        S.split_on_oom(gen, lambda e: isinstance(e, OOM))(jobs)
        self.assertIsNone(jobs[0].result); self.assertIn("OOM", jobs[0].error); self.assertTrue(jobs[1].result)

    def test_other_errors_propagate(self):
        import hf_chat_server as S
        def gen(b): raise ValueError("x")
        with self.assertRaises(ValueError):
            S.split_on_oom(gen, lambda e: False)([S.Job({"messages": [1]})])


if __name__ == "__main__":
    unittest.main(verbosity=1)
