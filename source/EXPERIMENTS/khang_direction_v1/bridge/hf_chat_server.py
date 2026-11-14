"""Server OpenAI-compatible toi gian chay Qwen3.5-9B (HF + bnb 4-bit, cuda:0) cho runner cua Khang.

Chi nghe 127.0.0.1. Khong goi mang ngoai. Gom cac request dong thoi thanh micro-batch
(cung tham so lay mau) roi goi model.generate mot lan.

Phan logic (Batcher, format_response, split_thinking, truncate_stop) khong can GPU va co test CPU.
Phan nap model (load_model) dung lai cau hinh da kiem tren Kaggle: nf4 double-quant, fp16, sdpa, device_map cuda:0.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_thinking(text: str) -> tuple[str | None, str]:
    """Tach phan suy nghi. Template co the da mo <think> san nen chi co </think>."""
    if THINK_CLOSE in text:
        head, tail = text.split(THINK_CLOSE, 1)
        return head.replace(THINK_OPEN, "").strip(), tail.strip()
    if THINK_OPEN in text:  # bi cat giua phan suy nghi
        return text.replace(THINK_OPEN, "").strip(), ""
    return None, text.strip()


def truncate_stop(text: str, stops: list[str] | None) -> tuple[str, bool]:
    if not stops:
        return text, False
    cut = min((text.find(s) for s in stops if s and s in text), default=-1)
    return (text[:cut], True) if cut >= 0 else (text, False)


def format_response(model_name: str, content: str, reasoning: str | None, finish: str,
                    prompt_tokens: int, completion_tokens: int) -> dict:
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {
        "object": "chat.completion",
        "model": model_name,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


@dataclass
class Job:
    body: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: str | None = None

    def key(self) -> tuple:
        b = self.body
        return (float(b.get("temperature", 0.0)), float(b.get("top_p", 1.0)), int(b.get("top_k", -1) or -1),
                bool((b.get("chat_template_kwargs") or {}).get("enable_thinking", True)))


class Batcher:
    """Gom job theo khoa tham so lay mau; goi generate_fn(list[Job]) tren mot thread duy nhat (GPU)."""

    def __init__(self, generate_fn, max_batch: int = 8, wait_s: float = 0.05):
        self.q: queue.Queue[Job] = queue.Queue()
        self.generate_fn = generate_fn
        self.max_batch = max_batch
        self.wait_s = wait_s
        self.stats = {"batches": 0, "jobs": 0, "max_seen": 0}
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def submit(self, body: dict, timeout: float | None = None) -> Job:
        job = Job(body)
        self.q.put(job)
        job.done.wait(timeout)
        return job

    def _loop(self):
        pending: list[Job] = []
        while True:
            if not pending:
                pending.append(self.q.get())
            deadline = time.monotonic() + self.wait_s
            while len(pending) < 4 * self.max_batch:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    pending.append(self.q.get(timeout=left))
                except queue.Empty:
                    break
            k0 = pending[0].key()
            batch = [j for j in pending if j.key() == k0][: self.max_batch]
            pending = [j for j in pending if j not in batch]
            self.stats["batches"] += 1
            self.stats["jobs"] += len(batch)
            self.stats["max_seen"] = max(self.stats["max_seen"], len(batch))
            try:
                self.generate_fn(batch)
            except Exception as exc:  # loi cua batch duoc tra ve tung request, server van chay
                for j in batch:
                    j.error = f"{type(exc).__name__}: {exc}"
            for j in batch:
                j.done.set()


def load_model(base_dir: str, adapter: str | None):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    try:
        from transformers import Qwen3_5ForCausalLM as M
    except ImportError:  # ten lop phu thuoc phien ban transformers
        from transformers import AutoModelForCausalLM as M
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tok = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = M.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False, quantization_config=quant,
                              device_map={"": 0}, dtype=torch.float16, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True, is_trainable=False)
    count = model.num_parameters()
    assert count <= 15_000_000_000, "Parameter ceiling exceeded"
    model.eval()
    model.config.use_cache = True
    return model, tok


def split_on_oom(gen_fn, is_oom, on_oom=None, stats=None):
    """Bao generate: neu het VRAM thi chia doi batch va chay lai; batch 1 van OOM thi bao loi cho request do."""
    def run(batch):
        oom = None
        try:
            gen_fn(batch)
        except Exception as exc:
            if not is_oom(exc):
                raise
            oom = f"{type(exc).__name__}: {str(exc)[:200]}"
        if oom is not None:  # ra khoi except de traceback giai phong tensor truoc khi empty_cache
            if on_oom:
                on_oom()
            if stats is not None:
                stats["oom_splits"] = stats.get("oom_splits", 0) + 1
            if len(batch) == 1:
                raise RuntimeError("OOM at batch 1: " + oom)
            h = len(batch) // 2
            for part in (batch[:h], batch[h:]):
                try:
                    run(part)
                except Exception as e2:
                    for j in part:
                        if j.result is None:
                            j.error = f"{type(e2).__name__}: {str(e2)[:300]}"
    return run


def system_prefix_len(tok, messages, prompt: str, ids) -> int:
    """So token cua phan system (gom ca the dong) neu tach duoc an toan; 0 neu khong."""
    if not messages or messages[0].get("role") != "system":
        return 0
    try:
        sys_str = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=False)
    except Exception:
        return 0
    if not prompt.startswith(sys_str):
        return 0
    sys_ids = tok(sys_str, add_special_tokens=False)["input_ids"]
    L = len(sys_ids)
    if L == 0 or L > ids.shape[1] or ids[0, :L].tolist() != list(sys_ids):
        return 0
    return L


def make_generate_fn(model, tok, model_name: str, seed: int, max_new_cap: int, stats: dict | None = None,
                     prefill_chunk: int = 2048, prefix_cache: dict | None = None):
    """prefix_cache: dict dung chung; luu trang thai cache sau system prompt (giong nhau moi cau) de bo prefill lai.
    Model lai (linear attention) khong cat cache duoc, nen snapshot chup dung tai ranh gioi system."""
    import copy, time
    import torch

    def generate(batch: list[Job]):
        b0 = batch[0].body
        temp, top_p, top_k, thinking = batch[0].key()
        prompts = [tok.apply_chat_template(j.body["messages"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking) for j in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        width = enc["input_ids"].shape[1]
        if stats is not None:
            stats["max_prompt_tokens"] = max(stats.get("max_prompt_tokens", 0), int(width))
        max_new = min(max(int(j.body.get("max_tokens", 1024)) for j in batch), max_new_cap)
        kw = dict(max_new_tokens=max_new, use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        if temp > 0:
            kw.update(do_sample=True, temperature=temp, top_p=top_p)
            if top_k > 0:
                kw["top_k"] = top_k
            torch.manual_seed(seed + int(b0.get("seed") or 0))
        else:
            kw.update(do_sample=False, num_beams=1)
        gen_in = dict(enc)
        if len(batch) == 1:
            # khong padding: bo attention_mask de SDPA dung nhan causal tiet kiem bo nho (prompt ~18k token tren T4)
            gen_in.pop("attention_mask", None)
        with torch.inference_mode():
            out = None
            if len(batch) == 1 and width > prefill_chunk:
                # prefill tung doan: bo nho attention ~ chunk x width thay vi width^2
                try:
                    ids, cache = gen_in["input_ids"], None
                    t0 = time.time()
                    start = 0
                    if prefix_cache is not None and prefix_cache.get("ids") is not None:
                        L = prefix_cache["ids"].shape[1]
                        if L < width - 1 and torch.equal(ids[:, :L], prefix_cache["ids"]):
                            cache, start = copy.deepcopy(prefix_cache["cache"]), L
                            if stats is not None:
                                stats["prefix_hits"] = stats.get("prefix_hits", 0) + 1
                    L_sys = 0
                    if prefix_cache is not None and start == 0:
                        L_sys = system_prefix_len(tok, batch[0].body["messages"], prompts[0], ids)
                    bounds = sorted({*range(start, width - 1, prefill_chunk), *( [L_sys] if start < L_sys < width - 1 else [] )})
                    bounds = [b for b in bounds if b >= start] + [width - 1]
                    for s, e in zip(bounds[:-1], bounds[1:]):
                        o = model(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True, logits_to_keep=1)
                        cache = o.past_key_values
                        del o
                        if e == L_sys and L_sys > 0:
                            prefix_cache["cache"] = copy.deepcopy(cache)
                            prefix_cache["ids"] = ids[:, :L_sys].clone()
                            if stats is not None:
                                stats["prefix_len"] = int(L_sys)
                    t1 = time.time()
                    out = model.generate(input_ids=ids, past_key_values=cache, **kw)
                    if stats is not None:
                        stats["chunked_prefill"] = stats.get("chunked_prefill", 0) + 1
                        stats["prefill_s"] = round(stats.get("prefill_s", 0) + t1 - t0, 1)
                        stats["decode_s"] = round(stats.get("decode_s", 0) + time.time() - t1, 1)
                        stats["new_tokens"] = stats.get("new_tokens", 0) + int(out.shape[1] - width)
                except torch.OutOfMemoryError:
                    raise
                except Exception as exc:  # cache lai khong ho tro: quay ve generate thuong
                    if stats is not None:
                        stats["chunk_fallback"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    out = None
            if out is None:
                out = model.generate(**gen_in, **kw)
        lens = enc["attention_mask"].sum(dim=1).tolist()
        for j, seq, plen in zip(batch, out[:, width:].tolist(), lens):
            own_max = int(j.body.get("max_tokens", 1024))
            hit_eos = tok.eos_token_id in seq
            if hit_eos:
                seq = seq[: seq.index(tok.eos_token_id)]
            seq = seq[:own_max]
            text = tok.decode(seq, skip_special_tokens=True)
            reasoning, content = split_thinking(text) if thinking else (None, text.strip())
            content, stopped = truncate_stop(content, j.body.get("stop"))
            finish = "stop" if (hit_eos and len(seq) <= own_max) or stopped else "length"
            j.result = format_response(model_name, content, reasoning, finish, int(plen), len(seq))

    return generate


def make_handler(batcher: Batcher, model_name: str, request_timeout: float):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # khong ghi noi dung request ra log
            pass

        def _send(self, code: int, obj: dict):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            if self.path.rstrip("/") == "/stats":
                st = dict(batcher.stats)
                try:
                    import torch
                    if torch.cuda.is_available():
                        st["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated(0))
                        st["gpu_name"] = torch.cuda.get_device_name(0)
                except Exception:
                    pass
                return self._send(200, st)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n))
                assert isinstance(body.get("messages"), list) and body["messages"]
            except Exception:
                return self._send(400, {"error": "bad request"})
            job = batcher.submit(body, timeout=request_timeout)
            if job.result is None:
                return self._send(500, {"error": job.error or "timeout"})
            self._send(200, job.result)

    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-name", default="qwen3.5-9b")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--wait-ms", type=float, default=50)
    ap.add_argument("--max-new-cap", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--request-timeout", type=float, default=3600)
    ap.add_argument("--prefix-cache", action="store_true")
    a = ap.parse_args(argv)
    model, tok = load_model(a.base, a.adapter)
    import torch
    extra: dict = {}
    gen = split_on_oom(make_generate_fn(model, tok, a.model_name, a.seed, a.max_new_cap, extra,
                                       prefix_cache={} if a.prefix_cache else None),
                       lambda e: isinstance(e, torch.OutOfMemoryError), torch.cuda.empty_cache, extra)
    batcher = Batcher(gen, a.max_batch, a.wait_ms / 1000)
    batcher.stats = extra.update(batcher.stats) or extra  # chung mot dict de /stats thay oom_splits
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(batcher, a.model_name, a.request_timeout))
    print(json.dumps({"server_ready": True, "port": a.port, "parameters": model.num_parameters()}), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
