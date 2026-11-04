"""RQ1 - accuracy of ReSA vs rule-based analyzers, zero-shot LLMs and IntelliSA (Table 2 layout).

Baselines are recomputed from IntelliSA's released detection files (checked against IntelliSA's published results).
ReSA is trained and evaluated once per seed; mean +/- std is reported, and the median-seed run is
used for the per-smell table and the paired script-level bootstrap against IntelliSA and GLITCH.

    python -m resa.experiments.rq1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from resa.config import add_config_args, config_from_args
from resa.data.oracle import published_detections
from resa.eval.accuracy import bootstrap, check_published, evaluate_systems, load_run, per_smell
from resa.experiments.common import TABLE2_COLUMNS, TABLE2_HEADERS, aggregate, ensure_model, median_run, run_and_eval, save_table
from resa.smells import SMELL_DISPLAY, SMELLS
from resa.utils import get_logger, write_json

LOG = get_logger("resa.rq1")


def run(cfg, force: bool = False) -> dict:
    check = check_published(cfg)
    write_json(cfg.path("results") / "rq1_baseline_check.json", check)
    if not all(r["ok"] for r in check.values()) and not force:
        raise RuntimeError("Published baselines do not reproduce IntelliSA's published results; inspect results/rq1_baseline_check.json (or --force)")

    seed_rows = []
    for seed in cfg["student"]["seeds"]:
        verifier = ensure_model(cfg, "verify", seed)
        recovery = ensure_model(cfg, "recover", seed) if cfg["pipeline"].get("use_recovery", True) else None
        row = run_and_eval(cfg, verifier, recovery, f"resa__s{seed}")
        row["seed"] = seed
        seed_rows.append(row)
        LOG.info("seed %d: Macro-F1 %.3f", seed, row["macro_f1"])

    published = published_detections(cfg)
    base_rows = evaluate_systems(cfg, published)
    resa_mean = aggregate(seed_rows, "ReSA (ours)")
    save_table(cfg, "rq1_table2", base_rows + [resa_mean], TABLE2_COLUMNS, TABLE2_HEADERS,
               "Detection accuracy on the oracle (241 scripts, 213 smells). Baselines recomputed from IntelliSA's released outputs; "
               "ReSA mean$\\pm$std over seeds.", bold_best=TABLE2_COLUMNS[1:])
    save_table(cfg, "rq1_seeds", seed_rows, ["seed"] + TABLE2_COLUMNS[1:], ["Seed"] + TABLE2_HEADERS[1:], "ReSA per seed.")

    rep = median_run(seed_rows)
    systems = dict(published)
    systems["ReSA (ours)"] = load_run(Path(rep["run_dir"]))
    boot = bootstrap(cfg, {k: systems[k] for k in ("ReSA (ours)", "IntelliSA", "GLITCH", "Grok-4 (rules)")}, reference="IntelliSA",
                     iters=int(cfg["evaluation"]["bootstrap_iters"]), ci=float(cfg["evaluation"]["ci"]), seed=int(cfg["seed"]))
    boot += [dict(r, delta_vs="GLITCH") for r in bootstrap(cfg, {k: systems[k] for k in ("ReSA (ours)", "GLITCH")}, reference="GLITCH",
                                                            iters=int(cfg["evaluation"]["bootstrap_iters"]), seed=int(cfg["seed"])) if r["method"] == "ReSA (ours)"]
    save_table(cfg, "rq1_bootstrap", boot, ["method", "macro_f1_ci_low", "macro_f1_ci_high", "delta_vs", "delta_mean", "delta_ci_low", "delta_ci_high", "p_value_delta_le_0"],
               ["Method", "CI low", "CI high", "vs", "$\\Delta$", "$\\Delta$ low", "$\\Delta$ high", "p"],
               f"Paired script-level bootstrap ({cfg['evaluation']['bootstrap_iters']} resamples) for the median seed (seed {rep['seed']}).")

    smell_rows = per_smell(cfg, {k: systems[k] for k in ("GLITCH", "IntelliSA", "ReSA (ours)")})
    pivot = []
    for smell in SMELLS:
        row = {"smell": SMELL_DISPLAY[smell], "occurrences": next(r["occurrences"] for r in smell_rows if r["smell"] == smell)}
        for r in smell_rows:
            if r["smell"] == smell:
                key = {"GLITCH": "glitch", "IntelliSA": "intellisa", "ReSA (ours)": "resa"}[r["method"]]
                row[f"{key}_p"], row[f"{key}_r"], row[f"{key}_f1"] = r["precision"], r["recall"], r["f1"]
        pivot.append(row)
    save_table(cfg, "rq1_per_smell", pivot,
               ["smell", "occurrences", "glitch_f1", "intellisa_p", "intellisa_r", "intellisa_f1", "resa_p", "resa_r", "resa_f1"],
               ["Smell", "Occ.", "GLITCH F1", "IntelliSA P", "IntelliSA R", "IntelliSA F1", "ReSA P", "ReSA R", "ReSA F1"],
               f"Per-smell results pooled over technologies (ReSA: median seed {rep['seed']}).")
    summary = {"resa_mean": resa_mean, "median_seed": rep["seed"], "bootstrap": boot,
               "beats_intellisa_macro_f1": resa_mean.get("macro_f1", 0) > 0.831}
    write_json(cfg.path("results") / "rq1_summary.json", summary)
    return summary


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Continue even if baseline recomputation mismatches")
    add_config_args(parser)
    args = parser.parse_args(argv)
    run(config_from_args(args), args.force)


if __name__ == "__main__":
    main()
