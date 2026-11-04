"""Labelled-item records shared by pools, teachers and students.

An item is one (script, line, smell) decision:
    {id, stage: verify|recover, tech, file, path, line, smell, source, rule, split,
     claude_label?: TP|FP, oracle_label?: 0|1, context_fallback?: str}
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from resa.context import extract_context, student_input
from resa.utils import read_lines, sha1_text


def item_id(stage: str, tech: str, file: str, line: int, smell: str) -> str:
    return sha1_text(f"{stage}|{tech}|{file}|{line}|{smell}")


@lru_cache(maxsize=4096)
def _cached_lines(path: str) -> tuple:
    return tuple(read_lines(path))


def item_lines(item: Dict) -> Optional[Sequence[str]]:
    path = item.get("path")
    if path and Path(path).exists():
        return _cached_lines(str(path))
    return None


def item_context(item: Dict, window: int, mode: str = "lines", max_chars: int = 1800, mark_target: bool = True) -> str:
    lines = item_lines(item)
    if lines is not None:
        return extract_context(lines, int(item["line"]), window, mode, max_chars, mark_target)
    fallback = item.get("context_fallback") or item.get("content") or ""
    if window <= 0 and item.get("content"):
        return str(item["content"]).strip()
    return str(fallback)[:max_chars]


def item_text(item: Dict, ctx_cfg: Dict) -> str:
    context = item_context(
        item,
        int(ctx_cfg.get("window", 5)),
        ctx_cfg.get("mode", "lines"),
        int(ctx_cfg.get("max_chars", 1800)),
        bool(ctx_cfg.get("mark_target", True)),
    )
    return student_input(item["tech"], item["smell"], item["stage"], context)


def by_id(items: List[Dict]) -> Dict[str, Dict]:
    return {it["id"]: it for it in items}
