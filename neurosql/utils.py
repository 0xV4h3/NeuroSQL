from __future__ import annotations

import re
from typing import Optional


SPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: Optional[str]) -> str:
    if text is None:
        return ""
    return SPACE_RE.sub(" ", str(text)).strip()


def build_prompt(query: str, context: str) -> str:
    q = normalize_whitespace(query)
    c = normalize_whitespace(context)
    return f"translate to SQL: context: {c} question: {q}"


def postprocess_sql(sql: str) -> str:
    s = normalize_whitespace(sql)
    if s and not s.endswith(";"):
        s += ";"
    return s