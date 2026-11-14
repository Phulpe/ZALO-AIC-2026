"""khang_v3 (12/11/2026): vi du giai mau (few-shot) lay tu TRAIN cho cau KHONG co code goi y.

Ngan hang = cac cau TRAIN learning ma code goi y tat dinh cho dap an dung (doi chieu gold TRAIN, dung ngoai luc suy luan).
Truy hoi TF-IDF (tu don + cap tu) tren cau hoi da chuan hoa (_norm cua hints); uu tien nhe cung cua hang.
Khong co tham so model moi; khong dung du lieu test. Chi them vao phan goi y cua cau se goi model.
"""
from __future__ import annotations

import json
import math
from collections import Counter

from .hints import _norm


def _feats(text: str) -> Counter:
    w = _norm(text).split()
    return Counter(w + [a + "_" + b for a, b in zip(w, w[1:])])


class FewShotBank:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.vecs = [_feats(r["question"]) for r in rows]
        df = Counter()
        for v in self.vecs:
            df.update(v.keys())
        n = len(rows)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.norms = [math.sqrt(sum((c * self.idf[t]) ** 2 for t, c in v.items())) or 1.0 for v in self.vecs]

    @classmethod
    def load(cls, path: str) -> "FewShotBank":
        with open(path, encoding="utf-8") as fh:
            return cls([json.loads(line) for line in fh if line.strip()])

    def top(self, question: str, business_id: str | None = None, k: int = 3, exclude_id: str | None = None,
            same_biz_bonus: float = 0.05, max_per_fn: int = 2) -> list[dict]:
        q = _feats(question)
        qn = math.sqrt(sum((c * self.idf.get(t, 1.0)) ** 2 for t, c in q.items())) or 1.0
        scored = []
        for i, v in enumerate(self.vecs):
            r = self.rows[i]
            if r.get("id") == exclude_id:
                continue
            dot = sum(c * v[t] * self.idf.get(t, 1.0) ** 2 for t, c in q.items() if t in v)
            if dot <= 0:
                continue
            s = dot / (qn * self.norms[i]) + (same_biz_bonus if business_id and r.get("business_id") == business_id else 0.0)
            scored.append((s, r.get("id", ""), i))
        scored.sort(key=lambda x: (-x[0], x[1]))  # tat dinh
        out, per_fn = [], Counter()
        for s, _, i in scored:
            r = self.rows[i]
            if per_fn[r.get("fn")] >= max_per_fn:
                continue
            per_fn[r.get("fn")] += 1
            out.append(r)
            if len(out) >= k:
                break
        return out


def fewshot_text(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["Ví dụ câu hỏi tương tự (của shop khác hoặc kỳ khác) đã giải đúng bằng một dòng code — chỉ tham khảo cách chọn hàm "
             "dựng sẵn và tham số; kỳ, tên hàng, bộ lọc phải lấy theo câu hỏi và dữ liệu của shop này:"]
    for r in examples:
        lines.append(f"- Hỏi: {r['question'].strip()}\n  Code: {r['code'].strip()}")
    return "\n".join(lines)
