"""Client toi server OpenAI-compatible chay noi bo (vLLM trong Docker; mlx_lm.server khi phat trien).

Khong goi dich vu ngoai: base_url luon la localhost. Chi dung thu vien chuan.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


@dataclass
class LLMReply:
    content: str
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLMClient:
    def __init__(self, base_urls: list[str], model: str, timeout: float = 900.0, backend: str = "vllm"):
        self.base_urls = [u.rstrip("/") for u in base_urls]
        self.model = model
        self.timeout = timeout
        self.backend = backend  # "vllm" | "mlx"
        self._rr = 0
        self._lock = threading.Lock()

    def _url(self, affinity: str | None) -> str:
        if len(self.base_urls) == 1:
            return self.base_urls[0]
        if affinity is not None:  # cung cua hang -> cung server (tan dung prefix cache)
            return self.base_urls[sum(affinity.encode()) % len(self.base_urls)]
        with self._lock:
            self._rr += 1
            return self.base_urls[self._rr % len(self.base_urls)]

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 4096,
        seed: int | None = None,
        enable_thinking: bool = True,
        stop: list[str] | None = None,
        affinity: str | None = None,
    ) -> LLMReply:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        if self.backend == "vllm":
            if top_k and top_k > 0:
                body["top_k"] = top_k
            body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        else:  # mlx_lm.server: dung cong tac mem cua Qwen3 trong tin nhan cuoi
            msgs = [dict(m) for m in messages]
            tag = "/think" if enable_thinking else "/no_think"
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["content"] = m["content"] + "\n" + tag
                    break
            body["messages"] = msgs
            if top_k and top_k > 0:
                body["top_k"] = top_k
        url = self._url(affinity) + "/v1/chat/completions"
        last_exc = None
        for attempt in range(5):
            t0 = time.time()
            data = json.dumps(body).encode()
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                last_exc = RuntimeError(f"HTTP {exc.code}: {detail[:300]}")
                # hoi thoai dai: giam max_tokens cho vua context
                m = re.search(r"maximum context length is (\d+)", detail)
                u = re.search(r"(\d+) (?:tokens )?in the messages|(\d+) (?:input|prompt) tokens", detail)
                if exc.code == 400 and m and u:
                    limit, used = int(m.group(1)), int(u.group(1) or u.group(2))
                    room = limit - used - 16
                    if room < 256:
                        raise RuntimeError("het context") from None
                    body["max_tokens"] = min(body["max_tokens"], room)
                    continue
                if exc.code == 400:
                    raise last_exc from None
                time.sleep(2 + 3 * attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last_exc = exc
                time.sleep(2 + 3 * attempt)
        else:
            raise RuntimeError(f"LLM server loi: {last_exc}")
        ch = out["choices"][0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning is None:
            m = _THINK_RE.search(content)
            if m:
                reasoning = m.group(1).strip()
                content = _THINK_RE.sub("", content).strip()
            elif "</think>" in content:  # template da mo <think> san
                reasoning, content = content.split("</think>", 1)
                reasoning = reasoning.replace("<think>", "").strip()
                content = content.strip()
        usage = out.get("usage") or {}
        return LLMReply(
            content=content,
            reasoning=reasoning,
            finish_reason=ch.get("finish_reason"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            seconds=round(time.time() - t0, 3),
        )

    def wait_ready(self, max_wait: float = 1800.0) -> None:
        t0 = time.time()
        pending = list(self.base_urls)
        while pending and time.time() - t0 < max_wait:
            for u in list(pending):
                try:
                    with urllib.request.urlopen(u + "/v1/models", timeout=5) as resp:
                        if resp.status == 200:
                            pending.remove(u)
                except Exception:
                    pass
            if pending:
                time.sleep(3)
        if pending:
            raise RuntimeError(f"LLM server chua san sang: {pending}")
