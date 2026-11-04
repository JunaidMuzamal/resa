"""Assemble (text, label) training data for a student from pools + teacher labels."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from resa.config import Config
from resa.items import item_text
from resa.utils import get_logger, read_jsonl

LOG = get_logger("resa.student.data")


def _labels(cfg: Config, stage: str, source: str) -> Dict[str, int]:
    ldir = cfg.path("labels") / stage
    if source == "committee":
        path = ldir / "committee.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m resa.teachers.committee label --stage {stage}`")
        return {r["id"]: int(r["label"]) for r in read_jsonl(path)}
    if source.startswith("teacher:"):
        name = source.split(":", 1)[1]
        path = ldir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        return {r["id"]: int(r["p_yes"] >= 0.5) for r in read_jsonl(path) if r.get("p_yes") is not None}
    if source == "claude":
        return {}  # resolved from the items themselves
    raise ValueError(f"Unknown label source {source!r}")


def load_stage_items(
    cfg: Config,
    stage: str,
    label_source: str = "committee",
    smells: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    pool = read_jsonl(cfg.path("processed") / "pools" / f"{stage}.jsonl")
    labels = _labels(cfg, stage, label_source)
    train, val = [], []
    for it in pool:
        if smells and it["smell"] not in smells:
            continue
        if sources and it["source"] not in sources:
            continue
        if label_source == "claude":
            if not it.get("claude_label"):
                continue
            y = 1 if it["claude_label"] == "TP" else 0
        else:
            if it["id"] not in labels:
                continue
            y = labels[it["id"]]
        rec = dict(it, y=y)
        (val if it.get("split") == "val" else train).append(rec)
    LOG.info(
        "[%s/%s] train=%d (pos %.2f) val=%d (pos %.2f)",
        stage,
        label_source,
        len(train),
        sum(r["y"] for r in train) / max(1, len(train)),
        len(val),
        sum(r["y"] for r in val) / max(1, len(val)),
    )
    return train, val


def texts_and_labels(items: List[dict], ctx_cfg: dict) -> Tuple[List[str], List[int]]:
    return [item_text(it, ctx_cfg) for it in items], [int(it["y"]) for it in items]


def oversample(items: List[dict], rare_smells: List[str], factor: int) -> List[dict]:
    if factor <= 1:
        return items
    out = list(items)
    for it in items:
        if it["smell"] in rare_smells:
            out.extend([it] * (factor - 1))
    return out


def label_stats(items: List[dict]) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Counter] = {}
    for it in items:
        stats.setdefault(it["smell"], Counter())[str(it["y"])] += 1
    return {k: dict(v) for k, v in stats.items()}
