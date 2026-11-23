"""Split optional thinking tags from the decoded string. Text never leaves Mac."""

from __future__ import annotations

import re

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def assemble(text: str) -> tuple[str, str]:
    chunks = [match.group(1).strip() for match in _THINK.finditer(text)]
    if not chunks:
        return "", text
    answer = _THINK.sub("", text).strip()
    return "\n".join(chunks), answer
