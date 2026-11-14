"""Vong lap agent cho mot cau hoi va bieu quyet giua nhieu lan chay.

Mot lan chay (sample): model viet code -> sandbox chay -> tra ket qua -> ... -> answer().
Nhieu lan chay voi seed khac nhau; chi tra loi khi du so lan dong y (sai bi tru diem, bo trong = 0).
"""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from .answers import answer_key
from .kernel import BusinessKernel
from .llm import LLMClient
from .prompts import HINT_CONTINUE, HINT_LAST, HINT_NO_CODE, OBSERVATION_TEMPLATE

_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


@dataclass
class AgentConfig:
    max_steps: int = 6
    max_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True
    soft_timeout: float = 30.0
    n_samples: int = 3
    min_votes: int = 2
    base_seed: int = 1234


@dataclass
class SampleResult:
    answer: dict | None
    steps: list[dict] = field(default_factory=list)
    thinking: list[str | None] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def extract_code(text: str) -> str | None:
    blocks = _CODE_RE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    if text and "```" in text:  # khoi chua dong (bi cat max_tokens)
        tail = text.split("```", 1)[1]
        tail = re.sub(r"^(python|py)\s*\n", "", tail)
        return tail.strip() or None
    return None


def run_sample(
    llm: LLMClient,
    kernel: BusinessKernel,
    system: str,
    first_user: str,
    today: str,
    cfg: AgentConfig,
    seed: int,
    affinity: str,
    deadline: float | None = None,
) -> SampleResult:
    sid = uuid.uuid4().hex
    t0 = time.time()
    res = SampleResult(answer=None)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": first_user}]
    try:
        for step in range(1, cfg.max_steps + 1):
            if deadline is not None and time.time() > deadline:
                res.error = "deadline"
                break
            reply = llm.chat(
                messages,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_tokens=cfg.max_tokens,
                seed=seed + step,
                enable_thinking=cfg.enable_thinking,
                affinity=affinity,
            )
            res.prompt_tokens += reply.prompt_tokens
            res.completion_tokens += reply.completion_tokens
            res.thinking.append(reply.reasoning)
            code = extract_code(reply.content)
            messages.append({"role": "assistant", "content": reply.content})
            if not code:
                res.steps.append({"step": step, "content": reply.content, "code": None, "output": None})
                messages.append({"role": "user", "content": HINT_NO_CODE})
                continue
            out = kernel.execute(sid, code, today, soft_timeout=cfg.soft_timeout)
            text = out.get("stdout") or ""
            if out.get("error"):
                text = (text + "\n" if text else "") + "LỖI:\n" + out["error"]
            res.steps.append({"step": step, "code": code, "output": text, "answer": out.get("answer"), "exec_seconds": out.get("seconds")})
            if out.get("answer") is not None and not out.get("error"):
                res.answer = out["answer"]
                break
            hint = HINT_LAST if step == cfg.max_steps - 1 else HINT_CONTINUE
            messages.append(
                {
                    "role": "user",
                    "content": OBSERVATION_TEMPLATE.format(step=step, max_steps=cfg.max_steps, output=text.strip() or "(không in gì)", hint=hint),
                }
            )
    except Exception as exc:  # loi server/sandbox: lan chay nay bo trong
        res.error = f"{type(exc).__name__}: {exc}"
    finally:
        kernel.end(sid)
    res.seconds = round(time.time() - t0, 3)
    return res


def vote(samples: list[SampleResult], min_votes: int) -> tuple[dict, dict]:
    """Chon dap an duoc nhieu lan chay dong y nhat; khong du phieu -> abstain."""
    keys = {}
    counts: Counter = Counter()
    for s in samples:
        if s.answer is None:
            continue
        k = answer_key(s.answer)
        counts[k] += 1
        keys.setdefault(k, s.answer)
    info = {"votes": dict(counts), "n_samples": len(samples), "n_answered": sum(counts.values())}
    if not counts:
        return {"type": "abstain"}, info
    (best, n), *rest = counts.most_common()
    tie = bool(rest) and rest[0][1] == n
    info["top_votes"] = n
    if n >= min_votes and not tie:
        return keys[best], info
    return {"type": "abstain"}, info
