"""Monetary cost comparison (RQ5): LLM APIs vs IntelliSA distillation vs ReSA, per 1,000 scripts.

API token counts come from the *actual* prompts IntelliSA released (zero-shot baselines) and from
its released teacher prompts (distillation); no API is called. Prices are configuration values -
set them to the prices on the date you run the experiment and report that date with the results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from resa.config import Config
from resa.data.oracle import TECHS, hub_dir
from resa.utils import read_json, read_jsonl

# zero-shot baseline -> (price key, prompt style in IntelliSA's prompt file names)
ZERO_SHOT = {
    "Claude-4 (rules)": ("claude-sonnet-4", "static_analysis_rules"),
    "Claude-4 (def)": ("claude-sonnet-4", "definition_based"),
    "GPT-5 (rules)": ("gpt-5", "static_analysis_rules"),
    "GPT-5 (def)": ("gpt-5", "definition_based"),
    "Grok-4 (rules)": ("grok-4-fast", "static_analysis_rules"),
    "Grok-4 (def)": ("grok-4-fast", "definition_based"),
}
OUTPUT_TOKENS_PER_SCRIPT = 120  # CSV findings; reasoning tokens (GPT-5) are not included -> lower bound


def _tokens(chars: int, cfg: Config) -> float:
    return chars / float(cfg["cost"]["chars_per_token"])


def zero_shot_costs(cfg: Config) -> List[Dict]:
    prices = cfg["cost"]["api_prices"]
    prompt_root = hub_dir(cfg) / "replication" / "RQ1" / "data" / "prompts" / "prompts"
    rows = []
    for method, (price_key, style) in ZERO_SHOT.items():
        chars = n = 0
        for tech in TECHS:
            for path in (prompt_root / tech).glob(f"prompt_{style}_*"):
                chars += len(path.read_text(encoding="utf-8", errors="replace"))
                n += 1
        if not n:
            continue
        tin = _tokens(chars, cfg) / n
        price = prices[price_key]
        per_script = tin * price["input"] / 1e6 + OUTPUT_TOKENS_PER_SCRIPT * price["output"] / 1e6
        rows.append({"method": method, "one_off_usd": 0.0, "usd_per_1k_scripts": 1000 * per_script,
                     "input_tokens_per_script": round(tin), "needs_external_api": True, "basis": "released prompts"})
    return rows


def intellisa_cost(cfg: Config) -> Dict:
    """One-off Claude-4 pseudo-labelling of IntelliSA's training data + local inference."""
    base = hub_dir(cfg) / "replication" / "datasets" / "training" / "combined"
    chars = n = 0
    for split in ("train", "val"):
        path = base / f"{split}.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                chars += len(row.get("with_prompt") or "")
                n += 1
    price = cfg["cost"]["api_prices"]["claude-sonnet-4"]
    one_off = _tokens(chars, cfg) * price["input"] / 1e6 + n * 2 * price["output"] / 1e6
    return {"method": "IntelliSA", "one_off_usd": one_off, "teacher_items": n, "needs_external_api": "labelling only",
            "basis": "released teacher prompts (Claude-4)"}


def resa_cost(cfg: Config, scaling_summary: Optional[Path] = None) -> Dict:
    """Committee labelling GPU-hours (one-off) + measured inference cost per 1,000 scripts."""
    gpu_usd = float(cfg["cost"]["gpu_hour_usd"])
    cpu_usd = float(cfg["cost"]["cpu_core_hour_usd"])
    seconds = 0.0
    items = 0
    for stage in ("verify", "recover"):
        for timing in (cfg.path("labels") / stage).glob("*.timing.json"):
            t = read_json(timing)
            seconds += float(t.get("seconds", 0.0))
            items += int(t.get("items", 0))
    row: Dict = {"method": "ReSA (ours)", "one_off_usd": seconds / 3600 * gpu_usd, "teacher_gpu_hours": seconds / 3600,
                 "teacher_items": items, "needs_external_api": False, "basis": "measured"}
    scaling_summary = scaling_summary or cfg.path("results") / "rq4_scaling_summary.json"
    if scaling_summary.exists():
        s = read_json(scaling_summary)
        best = s.get("reference_config", {})
        if best:
            cpu_core_s = float(best.get("cpu_seconds_per_1k", 0.0))
            gpu_s = float(best.get("gpu_seconds_per_1k", 0.0))
            row["usd_per_1k_scripts"] = cpu_core_s / 3600 * cpu_usd + gpu_s / 3600 * gpu_usd
            row["cpu_core_seconds_per_1k"] = cpu_core_s
            row["gpu_seconds_per_1k"] = gpu_s
    return row


def cost_table(cfg: Config) -> List[Dict]:
    rows = zero_shot_costs(cfg)
    rows.append(intellisa_cost(cfg))
    rows.append(resa_cost(cfg))
    return rows
