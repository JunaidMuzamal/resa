"""Cost-effectiveness metrics under IntelliSA's RQ2 protocol (Effort@N%Recall, F1@N%LOC).

Ranking construction (reverse-engineered from IntelliSA's released ranking files and validated by
``check_published_effort``): one row per non-blank line of the oracle scripts (|U|); kept alerts
first, by descending score with a stable hash tie-break, at most one alert per line; then every
remaining non-blank line with score 0. Ground-truth count = number of oracle smells.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from resa.config import Config
from resa.data.oracle import RANKING_FILES, TECHS, hub_dir, load_oracle, published_ranking_path
from resa.utils import get_logger, read_csv, sha1_text

LOG = get_logger("resa.eval.effort")


def build_ranking(method: str, tech: str, scored: Iterable[Tuple[str, int, str, float]], nonblank: List[Tuple[str, int]]) -> pd.DataFrame:
    universe = set(nonblank)
    alerts = []
    dropped = 0
    for file, line, smell, score in scored:
        if (file, line) not in universe:
            dropped += 1
            continue
        alerts.append((float(score), sha1_text(f"{method}|{tech}|{smell}|{file}|{line}"), smell, file, line))
    if dropped:
        LOG.debug("%s/%s: %d alerts on blank/out-of-range lines ignored", method, tech, dropped)
    alerts.sort(key=lambda a: (-a[0], a[1]))
    seen: Set[Tuple[str, int]] = set()
    rows = []
    for score, aid, smell, file, line in alerts:
        if (file, line) in seen:
            continue
        seen.add((file, line))
        rows.append({"score": score, "method": method, "tech": tech, "smell": smell, "file_path": file, "line": line, "alert_id": aid})
    background = sorted(("zz_background_" + sha1_text(f"{tech}|{f}|{l}"), f, l) for f, l in universe - seen)
    rows.extend({"score": 0.0, "method": method, "tech": tech, "smell": "__no_alert__", "file_path": f, "line": l, "alert_id": aid} for aid, f, l in background)
    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def label_ranking(ranking: pd.DataFrame, truth: Set[Tuple[str, int, str]]) -> pd.DataFrame:
    df = ranking.copy()
    df["is_tp"] = [1 if (f, int(l), s) in truth else 0 for f, l, s in zip(df["file_path"], df["line"], df["smell"])]
    return df


def effort_metrics(df: pd.DataFrame, total_gt: int, f1_n: Iterable[float], recall_targets: Iterable[int]) -> Dict[str, float]:
    """Exactly IntelliSA's compute_f1_at_n_loc.py and compute_effort_at_n_recall.py."""
    df = df.sort_values("rank").reset_index(drop=True)
    total_lines = len(df)
    out: Dict[str, float] = {"total_lines": total_lines, "total_positives_gt": total_gt, "method_tp_coverage": int(df["is_tp"].sum())}
    for n in f1_n:
        k = min(math.ceil(n / 100.0 * total_lines), total_lines)
        tp = int(df.head(k)["is_tp"].sum())
        p = tp / k if k else 0.0
        r = tp / total_gt if total_gt else 0.0
        out[f"f1_at_{('%g' % n).replace('.', 'p')}pct"] =2 * p * r / (p + r) if p + r else 0.0
    cum = df["is_tp"].cumsum() / total_gt if total_gt else df["is_tp"] * 0
    max_recall = df["is_tp"].sum() / total_gt if total_gt else 0.0
    out["max_recall"] = float(max_recall)
    for target in recall_targets:
        r = target / 100.0
        reached = cum[cum >= r]
        if max_recall < r or reached.empty:
            out[f"effort_at_{target}pct_recall"] = 100.0
            out[f"reached_{target}pct"] = False
        else:
            k = int(reached.index[0]) + 1
            out[f"effort_at_{target}pct_recall"] = 100.0 * k / total_lines
            out[f"reached_{target}pct"] = True
    return out


def evaluate_effort(cfg: Config, scored_systems: Dict[str, Dict[str, List[Tuple[str, int, str, float]]]]) -> List[Dict]:
    """scored_systems: method -> tech -> [(file, line, smell, score)] of *kept* detections."""
    ecfg = cfg["evaluation"]
    rows = []
    for tech in TECHS:
        oracle = load_oracle(cfg, tech)
        nonblank = oracle.nonblank_lines()
        for method, per_tech in scored_systems.items():
            ranking = build_ranking(method, tech, per_tech.get(tech, []), nonblank)
            df = label_ranking(ranking, oracle.truth)
            m = effort_metrics(df, len(oracle.truth), ecfg["f1_n_percent"], ecfg["effort_recall_targets"])
            rows.append({"tech": tech, "method": method, **m})
    return rows


def published_effort(cfg: Config) -> List[Dict]:
    """Recompute Table 3 for every baseline from IntelliSA's released ranking files (no rerun)."""
    ecfg = cfg["evaluation"]
    rows = []
    for tech in TECHS:
        oracle_rows = read_csv(hub_dir(cfg) / "replication" / "datasets" / "oracle" / f"{tech}.csv")
        total_gt = len(oracle_rows)
        truth = {(r["PATH"].strip(), int(r["LINE"]), r["CATEGORY"].strip()) for r in oracle_rows}
        for method in RANKING_FILES:
            path = published_ranking_path(cfg, tech, method)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower()
            df = label_ranking(df, truth)
            rows.append({"tech": tech, "method": method, **effort_metrics(df, total_gt, ecfg["f1_n_percent"], ecfg["effort_recall_targets"])})
    return rows


def check_published_effort(cfg: Config, tol: float = 1e-6) -> Dict[str, bool]:
    """Our metric code on their rankings must equal IntelliSA's released Table 3 (``paper_main_results.csv``)."""
    ref_path = hub_dir(cfg) / "replication" / "RQ2" / "outputs" / "tables" / "final" / "paper_main_results.csv"
    if not ref_path.exists():
        return {}
    ref = pd.read_csv(ref_path)
    inv = {v: k for k, v in RANKING_FILES.items()}
    ours = {(r["tech"], r["method"]): r for r in published_effort(cfg)}
    report = {}
    for _, row in ref.iterrows():
        method = inv.get(row["method"])
        if method is None or (row["tech"], method) not in ours:
            continue
        mine = ours[(row["tech"], method)]
        ok = abs(mine["f1_at_1pct"] - row["f1_at_1pct"]) <= tol and abs(mine["effort_at_60pct_recall"] - row["effort_at_60pct_recall"]) <= tol
        report[f"{row['tech']}/{method}"] = bool(ok)
    bad = [k for k, v in report.items() if not v]
    if bad:
        LOG.warning("Effort metrics differ from IntelliSA's outputs for %s", bad)
    else:
        LOG.info("Effort metrics reproduce IntelliSA Table 3 for all %d (tech, method) pairs", len(report))
    return report


def scored_from_run(run_dir: Path) -> Dict[str, List[Tuple[str, int, str, float]]]:
    out = {}
    for tech in TECHS:
        path = run_dir / "detections" / f"{tech}.csv"
        if path.exists():
            out[tech] = [(r["PATH"], int(r["LINE"]), r["CATEGORY"], float(r.get("SCORE", 1.0) or 1.0)) for r in read_csv(path)]
    return out
