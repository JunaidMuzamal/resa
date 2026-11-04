"""Shared helpers for the RQ scripts: train-if-missing, run+evaluate, seed aggregation, LaTeX."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from resa.config import Config
from resa.data.oracle import TECHS
from resa.eval.accuracy import evaluate_systems, load_run
from resa.pipeline import run_on_oracle
from resa.student.train import default_run_name, train_student
from resa.utils import ensure_dir, get_logger, read_json, write_csv

LOG = get_logger("resa.experiments")

METRIC_KEYS = [f"{t}_{m}" for t in TECHS for m in ("precision", "recall", "f1")] + ["macro_f1"]


def variant(cfg: Config, **dotted) -> Config:
    """Copy of ``cfg`` with dotted-key overrides, e.g. variant(cfg, **{"context.window": 0})."""
    data = copy.deepcopy(dict(cfg))
    for key, value in dotted.items():
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Config(data)


def ensure_model(cfg: Config, stage: str, seed: int, label_source: str = "committee", smells=None, sources=None, tag: str = "") -> Path:
    name = default_run_name(stage, label_source, cfg, seed) + (f"__{tag}" if tag else "")
    path = cfg.path("models") / name
    if (path / "thresholds.json").exists():
        return path
    return train_student(cfg, stage, seed, label_source, name, smells, sources)


def run_and_eval(cfg: Config, verifier: Optional[Path], recovery: Optional[Path], run_name: str,
                 verify_mode: Optional[str] = None, use_recovery: Optional[bool] = None) -> Dict:
    run_dir = cfg.path("runs") / run_name
    if not (run_dir / "run.json").exists():
        run_on_oracle(cfg, verifier, recovery, run_name, verify_mode, use_recovery)
    row = evaluate_systems(cfg, {run_name: load_run(run_dir)})[0]
    row["run_dir"] = str(run_dir)
    return row


def aggregate(rows: List[Dict], label: str) -> Dict:
    out: Dict[str, object] = {"method": label, "n_seeds": len(rows)}
    for key in METRIC_KEYS:
        vals = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
        if len(vals):
            out[key] = float(vals.mean())
            out[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def median_run(rows: List[Dict]) -> Dict:
    ordered = sorted(rows, key=lambda r: r["macro_f1"])
    return ordered[len(ordered) // 2]


def fmt(value: Optional[float], std: Optional[float] = None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.3f}" if std is None else f"{value:.3f}$\\pm${std:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def latex_table(rows: List[Dict], columns: Sequence[str], headers: Sequence[str], caption: str, label: str, bold_best: Sequence[str] = ()) -> str:
    best = {}
    for col in bold_best:
        vals = [r.get(col) for r in rows if isinstance(r.get(col), (int, float))]
        best[col] = max(vals) if vals else None
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{l" + "r" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = []
        for col in columns:
            v = r.get(col)
            if isinstance(v, float):
                std = r.get(f"{col}_std")
                cells.append(fmt(v, std if isinstance(std, float) and r.get("n_seeds", 1) > 1 else None, col in best and best[col] is not None and abs(v - best[col]) < 1e-12))
            else:
                cells.append(str(v).replace("_", "\\_").replace("&", "\\&") if v is not None else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def save_table(cfg: Config, stem: str, rows: List[Dict], columns: Sequence[str], headers: Sequence[str], caption: str, bold_best: Sequence[str] = ()) -> None:
    out = ensure_dir(cfg.path("results"))
    write_csv(out / f"{stem}.csv", rows)
    (out / f"{stem}.tex").write_text(latex_table(rows, columns, headers, caption, f"tab:{stem}", bold_best), encoding="utf-8")
    LOG.info("Wrote results/%s.{csv,tex}", stem)


TABLE2_COLUMNS = ["method"] + [f"{t}_{m}" for t in ("puppet", "ansible", "chef") for m in ("precision", "recall", "f1")] + ["macro_f1"]
TABLE2_HEADERS = ["Method", "Pup P", "Pup R", "Pup F1", "Ans P", "Ans R", "Ans F1", "Chef P", "Chef R", "Chef F1", "Macro-F1"]
