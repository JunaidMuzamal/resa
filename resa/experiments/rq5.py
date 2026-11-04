"""RQ5 - inspection effort (IntelliSA Table 3 protocol) and monetary cost.

Baseline effort numbers are recomputed from IntelliSA's released rankings (and checked against
their released Table 3); ReSA is ranked with the same construction.

    python -m resa.experiments.rq5
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from resa.config import add_config_args, config_from_args
from resa.eval.cost import cost_table
from resa.eval.effort import check_published_effort, evaluate_effort, published_effort, scored_from_run
from resa.experiments.common import ensure_model, run_and_eval, save_table
from resa.utils import get_logger, write_json

LOG = get_logger("resa.rq5")


def run(cfg) -> dict:
    check = check_published_effort(cfg)
    write_json(cfg.path("results") / "rq5_effort_check.json", check)
    base = published_effort(cfg)
    ours: List[Dict] = []
    for seed in cfg["student"]["seeds"]:
        verifier = ensure_model(cfg, "verify", seed)
        recovery = ensure_model(cfg, "recover", seed)
        row = run_and_eval(cfg, verifier, recovery, f"resa__s{seed}")
        rows = evaluate_effort(cfg, {"ReSA (ours)": scored_from_run(Path(row["run_dir"]))})
        ours += [dict(r, seed=seed) for r in rows]
    table = []
    for tech in ("puppet", "ansible", "chef"):
        for r in base:
            if r["tech"] == tech:
                table.append({"tech": tech, "method": r["method"], "effort_at_60pct_recall": r["effort_at_60pct_recall"] if r["reached_60pct"] else None,
                              "f1_at_1pct": r["f1_at_1pct"], "max_recall": r["max_recall"]})
        mine = [r for r in ours if r["tech"] == tech]
        eff = [r["effort_at_60pct_recall"] for r in mine if r["reached_60pct"]]
        table.append({"tech": tech, "method": "ReSA (ours)", "effort_at_60pct_recall": float(np.mean(eff)) if len(eff) == len(mine) else None,
                      "f1_at_1pct": float(np.mean([r["f1_at_1pct"] for r in mine])), "max_recall": float(np.mean([r["max_recall"] for r in mine])),
                      "n_seeds": len(mine)})
    save_table(cfg, "rq5_effort", table, ["tech", "method", "effort_at_60pct_recall", "f1_at_1pct", "max_recall"],
               ["Tech", "Method", "Effort@60\\%Recall (\\%LOC)", "F1@1\\%LOC", "Max recall"],
               "Inspection effort (lower Effort is better; -- = 60\\% recall unreachable). ReSA: mean over seeds.")
    costs = cost_table(cfg)
    save_table(cfg, "rq5_cost", costs, ["method", "one_off_usd", "usd_per_1k_scripts", "needs_external_api", "basis"],
               ["Method", "One-off (USD)", "USD / 1k scripts", "External API", "Basis"],
               "Monetary cost. API prices from configs/default.yaml (cost.api_prices); GPU price cost.gpu_hour_usd.")
    summary = {"effort": table, "cost": costs, "effort_check_ok": all(check.values()) if check else None}
    write_json(cfg.path("results") / "rq5_summary.json", summary)
    return summary


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    args = parser.parse_args(argv)
    run(config_from_args(args))


if __name__ == "__main__":
    main()
