"""Detection accuracy under IntelliSA's exact protocol (RQ1/RQ3).

* events are (file, line, canonical smell) sets; a hit needs all three to match exactly
* precision/recall/F1 are micro-averaged per technology; Macro-F1 is their unweighted mean
* oracle "none" rows are ignored (clean scripts carry no events)

Baseline rows are *recomputed* from IntelliSA's released detection files, never rerun, and are
checked against IntelliSA's published numbers (``check_published``).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from resa.config import Config
from resa.data.oracle import PUBLISHED_TABLE2, TECHS, Event, load_oracle, published_detections
from resa.smells import SMELLS, normalize_category
from resa.utils import get_logger, read_csv

LOG = get_logger("resa.eval.accuracy")

Systems = Dict[str, Dict[str, Set[Event]]]


def compute_metrics(truth: Set[Event], preds: Optional[Set[Event]]) -> Dict[str, Optional[float]]:
    """Byte-for-byte the formula of IntelliSA's ``evaluate.compute_metrics``."""
    if preds is None:
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None}
    tp = len(truth & preds)
    fp = len(preds - truth)
    fn = len(truth - preds)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not truth and not preds else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not truth else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def load_detection_file(path: Path) -> Set[Event]:
    events: Set[Event] = set()
    for row in read_csv(path):
        cat = normalize_category(row.get("CATEGORY", ""))
        if not cat:
            continue
        try:
            line = int(float(row["LINE"]))
        except (KeyError, ValueError):
            continue
        events.add((row["PATH"].strip(), line, cat))
    return events


def load_run(run_dir: Path) -> Dict[str, Set[Event]]:
    return {t: load_detection_file(run_dir / "detections" / f"{t}.csv") for t in TECHS if (run_dir / "detections" / f"{t}.csv").exists()}


def evaluate_systems(cfg: Config, systems: Systems) -> List[Dict]:
    """One row per system: per-tech P/R/F1 + Macro-F1 (IntelliSA Table 2 layout)."""
    truths = {t: load_oracle(cfg, t).truth for t in TECHS}
    rows = []
    for name, per_tech in systems.items():
        row: Dict[str, object] = {"method": name}
        f1s = []
        for tech in TECHS:
            m = compute_metrics(truths[tech], per_tech.get(tech))
            for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
                row[f"{tech}_{key}"] = m[key]
            if m["f1"] is not None:
                f1s.append(m["f1"])
        row["macro_f1"] = sum(f1s) / len(f1s) if f1s else None
        rows.append(row)
    return rows


def per_smell(cfg: Config, systems: Systems) -> List[Dict]:
    """Per-smell precision/recall/F1 pooled over the three technologies (IntelliSA Table 7 layout)."""
    truths = {t: load_oracle(cfg, t).truth for t in TECHS}
    rows = []
    for name, per_tech in systems.items():
        for smell in SMELLS:
            truth = {(t,) + e for t in TECHS for e in truths[t] if e[2] == smell}
            preds = {(t,) + e for t in TECHS for e in per_tech.get(t, set()) if e[2] == smell}
            m = compute_metrics(truth, preds)
            rows.append({"method": name, "smell": smell, "occurrences": len(truth), **m})
    return rows


def _file_groups(events: Set[Event]) -> Dict[str, Set[Event]]:
    grouped: Dict[str, Set[Event]] = defaultdict(set)
    for e in events:
        grouped[e[0]].add(e)
    return grouped


def bootstrap(
    cfg: Config,
    systems: Systems,
    reference: Optional[str] = None,
    iters: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> List[Dict]:
    """Script-level bootstrap (resample oracle scripts per technology, paired across systems).

    Returns per system: Macro-F1 CI and, if ``reference`` is given, the paired difference
    Delta = system - reference with its CI and a one-sided p-value P(Delta <= 0).
    """
    rng = np.random.default_rng(seed)
    oracles = {t: load_oracle(cfg, t) for t in TECHS}
    files = {t: sorted(oracles[t].files) for t in TECHS}
    truth_g = {t: _file_groups(oracles[t].truth) for t in TECHS}
    sys_g = {name: {t: _file_groups(per.get(t, set())) for t in TECHS} for name, per in systems.items()}
    samples = {name: np.zeros(iters) for name in systems}
    for b in range(iters):
        picks = {t: rng.integers(0, len(files[t]), len(files[t])) for t in TECHS}
        for name in systems:
            f1s = []
            for t in TECHS:
                tp = fp = fn = 0
                for idx in picks[t]:
                    f = files[t][idx]
                    tr = truth_g[t].get(f, set())
                    pr = sys_g[name][t].get(f, set())
                    tp += len(tr & pr)
                    fp += len(pr - tr)
                    fn += len(tr - pr)
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                f1s.append(2 * p * r / (p + r) if p + r else 0.0)
            samples[name][b] = float(np.mean(f1s))
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = []
    for name in systems:
        row = {"method": name, "macro_f1_ci_low": float(np.quantile(samples[name], lo)), "macro_f1_ci_high": float(np.quantile(samples[name], hi))}
        if reference and reference in systems and name != reference:
            diff = samples[name] - samples[reference]
            row.update(
                {
                    "delta_vs": reference,
                    "delta_mean": float(diff.mean()),
                    "delta_ci_low": float(np.quantile(diff, lo)),
                    "delta_ci_high": float(np.quantile(diff, hi)),
                    "p_value_delta_le_0": float((diff <= 0).mean()),
                }
            )
        out.append(row)
    return out


def check_published(cfg: Config, tol: float = 0.0015) -> Dict[str, Dict]:
    """Recompute every published baseline from its released file and compare with IntelliSA's published numbers."""
    systems = published_detections(cfg)
    rows = {r["method"]: r for r in evaluate_systems(cfg, systems)}
    report: Dict[str, Dict] = {}
    for method, techs in PUBLISHED_TABLE2.items():
        if method not in rows:
            continue
        diffs = {}
        for tech, (p, r, f) in techs.items():
            got = (rows[method][f"{tech}_precision"], rows[method][f"{tech}_recall"], rows[method][f"{tech}_f1"])
            diffs[tech] = max(abs(a - b) for a, b in zip(got, (p, r, f)))
        report[method] = {"max_abs_diff": max(diffs.values()), "per_tech": diffs, "ok": max(diffs.values()) <= tol, "macro_f1": rows[method]["macro_f1"]}
    bad = [m for m, r in report.items() if not r["ok"]]
    if bad:
        LOG.warning("Published baselines differ from IntelliSA's reported numbers for: %s", bad)
    else:
        LOG.info("All %d published baselines reproduce IntelliSA Table 2 within %.4f", len(report), tol)
    return report
