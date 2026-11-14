"""Doc README.md cua tung cua hang: danh sach file, mo ta cot, quy uoc, ngay cuoi du lieu."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field


@dataclass
class Section:
    """Muc '## Cot trong `pattern`[, sheet `sheet`]'."""

    pattern: str
    sheet: str | None
    columns: dict[str, str] = field(default_factory=dict)  # ten cot -> mo ta

    def file_regex(self) -> re.Pattern:
        parts = re.split(r"(<[^>]+>)", self.pattern)
        rx = "".join(".+?" if p.startswith("<") and p.endswith(">") else re.escape(p) for p in parts)
        return re.compile("^" + rx + "$", re.IGNORECASE)

    def matches(self, filename: str, sheet: str | None) -> bool:
        if not self.file_regex().match(filename):
            return False
        return self.sheet is None or (sheet is not None and sheet.strip().lower() == self.sheet.strip().lower())


@dataclass
class Readme:
    text: str
    summary: str
    files: list[str]
    sections: list[Section]
    conventions: list[str]
    data_end: dt.date | None

    def all_column_names(self) -> set[str]:
        return {c for s in self.sections for c in s.columns}

    def description(self, column: str) -> str | None:
        for s in self.sections:
            if column in s.columns:
                return s.columns[column]
        return None


_RE_SECTION = re.compile(r"^## Cột trong `([^`]+)`(?:, sheet `([^`]+)`)?\s*$")
_RE_COL = re.compile(r"^- \*\*(.+?)\*\*: (.*)$")
_RE_FILE = re.compile(r"^- `([^`]+)`")
_RE_END = re.compile(r"Dữ liệu có đến hết (\d{1,2})/(\d{1,2})/(\d{4})")


def parse_readme(text: str) -> Readme:
    lines = text.splitlines()
    sections: list[Section] = []
    files: list[str] = []
    conventions: list[str] = []
    summary_lines: list[str] = []
    mode = "head"
    cur: Section | None = None
    for line in lines:
        if line.startswith("# "):
            continue
        m = _RE_SECTION.match(line)
        if m:
            cur = Section(m.group(1), m.group(2))
            sections.append(cur)
            mode = "cols"
            continue
        if line.startswith("## "):
            title = line[3:].strip().lower()
            mode = {"file": "files", "quy ước": "conv", "bảng": "tables"}.get(title, "other")
            cur = None
            continue
        if mode == "head" and line.strip():
            summary_lines.append(line.strip())
        elif mode == "files":
            m = _RE_FILE.match(line)
            if m:
                files.append(m.group(1))
        elif mode == "cols" and cur is not None:
            m = _RE_COL.match(line)
            if m:
                cur.columns[m.group(1).strip()] = m.group(2).strip()
        elif mode == "conv" and line.strip():
            conventions.append(line.rstrip())
    m = _RE_END.search(text)
    end = dt.date(int(m[3]), int(m[2]), int(m[1])) if m else None
    return Readme(text, " ".join(summary_lines), files, sections, conventions, end)
