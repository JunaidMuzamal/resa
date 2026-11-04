"""Code-context extraction around a target line (the G2 contribution) and model input formatting."""
from __future__ import annotations

from typing import List, Sequence, Tuple

TARGET_MARK = ">>"


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip(" \t"))


def window_bounds(n_lines: int, line: int, window: int) -> Tuple[int, int]:
    """1-based inclusive bounds of a +/- ``window`` line window."""
    lo = max(1, line - window)
    hi = min(n_lines, line + window)
    return lo, hi


def block_bounds(lines: Sequence[str], line: int, max_span: int = 30) -> Tuple[int, int]:
    """Bounds of the enclosing block (YAML task, Puppet resource, Ruby do/end) by indentation."""
    n = len(lines)
    idx = line - 1
    if idx < 0 or idx >= n:
        return max(1, line), min(n, line)
    target_indent = _indent(lines[idx]) if lines[idx].strip() else 0
    start = idx
    if target_indent > 0:
        probe = idx - 1
        while probe >= 0 and idx - probe <= max_span:
            text = lines[probe]
            if text.strip() and _indent(text) < target_indent:
                start = probe
                break
            probe -= 1
    header_indent = _indent(lines[start]) if lines[start].strip() else 0
    end = idx
    probe = idx + 1
    while probe < n and probe - idx <= max_span:
        text = lines[probe]
        if text.strip() and _indent(text) <= header_indent:
            if text.strip() in ("}", "end", "},", "};", "]"):
                end = probe  # keep the block's closing token
            break
        end = probe
        probe += 1
    return start + 1, end + 1


def extract_context(
    lines: Sequence[str],
    line: int,
    window: int = 5,
    mode: str = "lines",
    max_chars: int = 1800,
    mark_target: bool = True,
) -> str:
    """Numbered code context around ``line`` (1-based). ``window=0`` returns the line only."""
    n = len(lines)
    if n == 0 or line < 1 or line > n:
        return ""
    if mode == "block":
        lo, hi = block_bounds(lines, line)
        wlo, whi = window_bounds(n, line, window)
        lo, hi = min(lo, wlo), max(hi, whi)
    elif window <= 0:
        return lines[line - 1].strip()
    else:
        lo, hi = window_bounds(n, line, window)
    rendered: List[str] = []
    for no in range(lo, hi + 1):
        prefix = f"{TARGET_MARK} " if (mark_target and no == line) else "   "
        rendered.append(f"{prefix}{no}: {lines[no - 1].rstrip()}")
    text = "\n".join(rendered)
    if len(text) <= max_chars:
        return text
    # trim symmetrically around the target line, keeping it whole
    target_idx = line - lo
    keep = [rendered[target_idx]]
    budget = max_chars - len(rendered[target_idx])
    up, down = target_idx - 1, target_idx + 1
    while budget > 0 and (up >= 0 or down < len(rendered)):
        for pos in (up, down):
            if 0 <= pos < len(rendered) and len(rendered[pos]) + 1 <= budget:
                budget -= len(rendered[pos]) + 1
                if pos < target_idx:
                    keep.insert(0, rendered[pos])
                else:
                    keep.append(rendered[pos])
        up -= 1
        down += 1
        if up < 0 and down >= len(rendered):
            break
    return "\n".join(keep)


def student_input(tech: str, smell: str, stage: str, context: str) -> str:
    """Text fed to a student classifier. Smell and stage condition one model across all smells."""
    return f"stage: {stage} | tech: {tech} | smell: {smell}\n{context}"
