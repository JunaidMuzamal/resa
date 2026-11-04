"""RQ2 - what does the recovery module add?

Reports, for every seed's ReSA run:
  * candidate coverage: share of oracle smells missed by GLITCH that the relaxed generator proposes
  * recovered true / false positives by smell and technology
  * with vs without recovery (same verifier): Delta precision / recall / F1
  * recovery-threshold sweep (precision-recall trade-off for practitioners) + figure

    python -m resa.experiments.rq2
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from resa.candidates import generate_candidates
from resa.config import add_config_args, config_from_args
from resa.data.oracle import TECHS, glitch_oracle_warnings, load_oracle
from resa.eval.accuracy import compute_metrics, evaluate_systems, load_run
from resa.experiments.common import aggregate, ensure_model, run_and_eval, save_table
from resa.smells import SMELL_DISPLAY, SMELLS
from resa.utils import ensure_dir, get_logger, read_csv, read_json, write_json

LOG = get_logger("resa.rq2")


def candidate_coverage(cfg) -> List[Dict]:
    rows = []
    totals = Counter()
    for tech in TECHS:
        oracle = load_oracle(cfg, tech)
        stage1 = {(w["file"], w["line"], w["smell"]) for w in glitch_oracle_warnings(cfg, tech)}
        missed = oracle.truth - stage1
        cands = set()
        n_cands = 0
        for name in oracle.files:
            if not oracle.has_script(name):
                continue
            exclude = {(l, s) for (f, l, s) in stage1 if f == name}
            cs = generate_candidates(tech, oracle.lines(name), exclude)
            n_cands += len(cs)
            cands |= {(name, c.line, c.smell) for c in cs}
        by_smell = defaultdict(lambda: [0, 0])
        for e in missed:
            by_smell[e[2]][0] += 1
            by_smell[e[2]][1] += int(e in cands)
        rows.append({"tech": tech, "glitch_missed": len(missed), "covered_by_candidates": len(missed & cands),
                     "coverage": len(missed & cands) / len(missed) if missed else None, "candidates": n_cands,
                     "candidates_per_kloc": 1000 * n_cands / max(1, len(oracle.nonblank_lines())),
                     "missed_by_smell": {SMELL_DISPLAY[s]: f"{v[1]}/{v[0]}" for s, v in by_smell.items()}})
        totals["missed"] += len(missed)
        totals["covered"] += len(missed & cands)
    rows.append({"tech": "all", "glitch_missed": totals["missed"], "covered_by_candidates": totals["covered"],
                 "coverage": totals["covered"] / totals["missed"] if totals["missed"] else None})
    return rows


def recovered_breakdown(cfg, run_dir: Path) -> List[Dict]:
    rows = []
    for tech in TECHS:
        truth = load_oracle(cfg, tech).truth
        dec = read_csv(run_dir / "decisions" / f"{tech}.csv")
        kept = [d for d in dec if d["source"] == "recovered" and d["kept"] == "True"]
        counts = defaultdict(lambda: [0, 0])
        for d in kept:
            hit = (d["file"], int(d["line"]), d["smell"]) in truth
            counts[d["smell"]][0 if hit else 1] += 1
        for smell in SMELLS:
            if smell in counts:
                rows.append({"tech": tech, "smell": SMELL_DISPLAY[smell], "recovered_tp": counts[smell][0], "recovered_fp": counts[smell][1]})
    return rows


def threshold_sweep(cfg, run_dir: Path, taus: List[float]) -> List[Dict]:
    """Re-decide recovered candidates at other recovery thresholds (verifier threshold unchanged)."""
    rows = []
    base = load_run(run_dir)
    meta = read_json(run_dir / "run.json")
    for tau in taus:
        f1s, added_tp, added_fp = [], 0, 0
        for tech in TECHS:
            truth = load_oracle(cfg, tech).truth
            dec = read_csv(run_dir / "decisions" / f"{tech}.csv")
            tau_v = meta["techs"][tech]["thresholds"]["verify"] or 0.5
            preds = {e for e in base.get(tech, set())}
            preds -= {(d["file"], int(d["line"]), d["smell"]) for d in dec if d["source"] == "recovered"}
            for d in dec:
                if d["source"] != "recovered":
                    continue
                pr = float(d["p_recover"]) if d["p_recover"] not in ("", "None") else 0.0
                pv = float(d["p_verify"]) if d["p_verify"] not in ("", "None") else None
                if pr >= tau and (pv is None or pv >= tau_v):
                    e = (d["file"], int(d["line"]), d["smell"])
                    preds.add(e)
                    added_tp += int(e in truth)
                    added_fp += int(e not in truth)
            f1s.append(compute_metrics(truth, preds)["f1"])
        rows.append({"tau_recover": tau, "macro_f1": float(np.mean(f1s)), "recovered_tp": added_tp, "recovered_fp": added_fp})
    return rows


def plot_sweep(cfg, rows: List[Dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    taus = [r["tau_recover"] for r in rows]
    ax.plot(taus, [r["recovered_tp"] for r in rows], marker="o", label="recovered TP")
    ax.plot(taus, [r["recovered_fp"] for r in rows], marker="s", label="recovered FP")
    ax.set_xlabel("recovery threshold $\\tau_r$")
    ax.set_ylabel("count (oracle)")
    ax2 = ax.twinx()
    ax2.plot(taus, [r["macro_f1"] for r in rows], color="black", linestyle="--", label="Macro-F1")
    ax2.set_ylabel("Macro-F1")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    out = ensure_dir(cfg.path("results") / "figures")
    fig.savefig(out / "rq2_threshold_sweep.pdf")
    fig.savefig(out / "rq2_threshold_sweep.png", dpi=200)
    plt.close(fig)


def run(cfg) -> dict:
    coverage = candidate_coverage(cfg)
    save_table(cfg, "rq2_candidate_coverage", [{k: v for k, v in r.items() if k != "missed_by_smell"} for r in coverage],
               ["tech", "glitch_missed", "covered_by_candidates", "coverage", "candidates", "candidates_per_kloc"],
               ["Tech", "GLITCH missed", "Covered", "Coverage", "Candidates", "Cand./KLOC"],
               "How many oracle smells missed by GLITCH the relaxed candidate generator proposes.")
    with_rec, without_rec, breakdown = [], [], []
    for seed in cfg["student"]["seeds"]:
        verifier = ensure_model(cfg, "verify", seed)
        recovery = ensure_model(cfg, "recover", seed)
        with_rec.append(run_and_eval(cfg, verifier, recovery, f"resa__s{seed}"))
        without_rec.append(run_and_eval(cfg, verifier, None, f"resa_norecovery__s{seed}", use_recovery=False))
        breakdown += [dict(r, seed=seed) for r in recovered_breakdown(cfg, Path(with_rec[-1]["run_dir"]))]
    a, b = aggregate(with_rec, "ReSA (verify + recover)"), aggregate(without_rec, "ReSA (verify only)")
    delta = {"method": "Delta (recovery)"}
    for k in a:
        if isinstance(a.get(k), float) and not k.endswith("_std") and isinstance(b.get(k), float):
            delta[k] = a[k] - b[k]
    cols = ["method"] + [f"{t}_{m}" for t in ("puppet", "ansible", "chef") for m in ("precision", "recall", "f1")] + ["macro_f1"]
    save_table(cfg, "rq2_with_without_recovery", [b, a, delta], cols,
               ["Variant", "Pup P", "Pup R", "Pup F1", "Ans P", "Ans R", "Ans F1", "Chef P", "Chef R", "Chef F1", "Macro-F1"],
               "Effect of the recovery module (same verifier), mean over seeds.")
    agg_break = defaultdict(lambda: [0, 0])
    for r in breakdown:
        agg_break[(r["tech"], r["smell"])][0] += r["recovered_tp"]
        agg_break[(r["tech"], r["smell"])][1] += r["recovered_fp"]
    n = len(cfg["student"]["seeds"])
    rows = [{"tech": t, "smell": s, "recovered_tp": v[0] / n, "recovered_fp": v[1] / n} for (t, s), v in sorted(agg_break.items())]
    save_table(cfg, "rq2_recovered_by_smell", rows, ["tech", "smell", "recovered_tp", "recovered_fp"],
               ["Tech", "Smell", "Recovered TP", "Recovered FP"], "Smells recovered beyond GLITCH (mean over seeds).")
    rep = sorted(with_rec, key=lambda r: r["macro_f1"])[len(with_rec) // 2]
    sweep = threshold_sweep(cfg, Path(rep["run_dir"]), list(cfg["experiments"]["recovery_threshold_sweep"]))
    save_table(cfg, "rq2_threshold_sweep", sweep, ["tau_recover", "macro_f1", "recovered_tp", "recovered_fp"],
               ["$\\tau_r$", "Macro-F1", "Rec. TP", "Rec. FP"], "Recovery threshold sweep (median seed).")
    plot_sweep(cfg, sweep)
    summary = {"coverage": coverage, "with": a, "without": b, "delta": delta}
    write_json(cfg.path("results") / "rq2_summary.json", summary)
    return summary


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    args = parser.parse_args(argv)
    run(config_from_args(args))


if __name__ == "__main__":
    main()
