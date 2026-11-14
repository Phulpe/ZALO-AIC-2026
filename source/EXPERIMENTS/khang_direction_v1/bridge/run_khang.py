"""Chay runner cua Khang (khong sua code goc) voi tuy chon --hints-off.

--hints-off: bo dong "Code gợi ý" khoi muc nhan dien de model tu viet code (mo phong dang cau rule chua phu).
Cac goi y chung (ky, ten hang, dang so do) van giu vi chung co san luc thi.
Moi tham so khac chuyen nguyen cho datamate.runner.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_LINE = re.compile(r"^Code gợi ý \([^)]*\): .*$\n?", re.M)


def strip_code_hint(text: str) -> str:
    return CODE_LINE.sub("", text)


def make_logging_client(base_cls, path):
    """Lop con ghi moi loi goi (messages + reply) vao jsonl de dung lai lam du lieu hoc (GD3). Ghi theo dong, fsync."""
    import json
    import threading
    lock = threading.Lock()

    class LoggingClient(base_cls):
        def chat(self, messages, **kw):
            reply = super().chat(messages, **kw)
            rec = {"seed": kw.get("seed"), "affinity": kw.get("affinity"), "temperature": kw.get("temperature"),
                   "enable_thinking": kw.get("enable_thinking"), "messages": messages,
                   "reply": {"content": reply.content, "reasoning": reply.reasoning,
                             "finish_reason": reply.finish_reason, "completion_tokens": reply.completion_tokens}}
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            return reply

    return LoggingClient


def main(argv):
    khang_src = os.environ.get("KHANG_SRC") or os.path.join(HERE, "..", "khang", "src")
    sys.path.insert(0, khang_src)
    hints_off = "--hints-off" in argv
    argv = [a for a in argv if a != "--hints-off"]
    from datamate import runner
    traj = os.environ.get("TRAJ_LOG")
    if traj:
        runner.LLMClient = make_logging_client(runner.LLMClient, traj)
    if hints_off:
        orig = runner.question_hints
        runner.question_hints = lambda *a, **k: strip_code_hint(orig(*a, **k))
    return runner.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
