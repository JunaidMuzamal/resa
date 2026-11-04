"""ReSA inference: stage-1 rules -> context-aware verifier -> candidate recovery -> verifier.

    python -m resa.pipeline --verifier models/<run> --recovery models/<run> --name resa_s42

On the oracle, stage 1 is GLITCH's *published* oracle output, so the test-time rule layer is
byte-identical to the GLITCH row of IntelliSA's Table 2.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from resa.candidates import generate_candidates
from resa.config import Config, add_config_args, config_from_args
from resa.data.oracle import TECHS, glitch_oracle_warnings, load_oracle
from resa.items import item_id, item_text
from resa.smells import smells_to_verify
from resa.student.model import Scorer
from resa.utils import ensure_dir, get_logger, write_csv, write_json

LOG = get_logger("resa.pipeline")


@dataclass
class Decision:
    tech: str
    file: str
    line: int
    smell: str
    source: str  # glitch | recovered
    p_verify: Optional[float]
    p_recover: Optional[float]
    kept: bool
    score: float


def _ctx(scorer: Scorer, cfg: Config) -> dict:
    return dict(scorer.meta.get("context") or cfg["context"])


def _score_items(scorer: Scorer, items: List[dict], cfg: Config, batch_size: int = 32) -> List[float]:
    ctx = _ctx(scorer, cfg)
    texts = [item_text(it, ctx) for it in items]
    return [float(p) for p in scorer.score(texts, batch_size)]


def run_on_scripts(
    cfg: Config,
    tech: str,
    scripts: Dict[str, Path],
    stage1: Dict[str, Set[Tuple[int, str]]],
    lines_of,
    verifier: Optional[Scorer],
    recovery: Optional[Scorer],
    verify_mode: str = "all",
    use_recovery: bool = True,
    recovery_requires_verifier: bool = True,
    batch_size: int = 32,
) -> List[Decision]:
    """Apply ReSA to ``scripts`` (name -> path) given stage-1 warnings (name -> {(line, smell)})."""
    verify_set = set(smells_to_verify(verify_mode))
    decisions: List[Decision] = []

    # ---- stage 2: verify rule warnings
    warn_items, direct = [], []
    for name in sorted(scripts):
        for line, smell in sorted(stage1.get(name, set())):
            item = {"id": item_id("verify", tech, name, line, smell), "stage": "verify", "tech": tech, "file": name,
                    "path": str(scripts[name]), "line": line, "smell": smell}
            (warn_items if (verifier is not None and smell in verify_set) else direct).append(item)
    p_warn = _score_items(verifier, warn_items, cfg, batch_size) if warn_items else []
    tau_v = verifier.threshold(tech) if verifier is not None else 0.5
    for it, p in zip(warn_items, p_warn):
        decisions.append(Decision(tech, it["file"], it["line"], it["smell"], "glitch", p, None, p >= tau_v, p))
    for it in direct:
        decisions.append(Decision(tech, it["file"], it["line"], it["smell"], "glitch", None, None, True, 1.0))

    # ---- stage 3: recover missed smells, then verify them
    if use_recovery and recovery is not None:
        rec_items = []
        for name in sorted(scripts):
            lines = lines_of(name)
            for cand in generate_candidates(tech, lines, stage1.get(name, set())):
                rec_items.append({"id": item_id("recover", tech, name, cand.line, cand.smell), "stage": "recover", "tech": tech,
                                  "file": name, "path": str(scripts[name]), "line": cand.line, "smell": cand.smell, "rule": cand.rule})
        p_rec = _score_items(recovery, rec_items, cfg, batch_size) if rec_items else []
        tau_r = recovery.threshold(tech)
        use_verifier = recovery_requires_verifier and verifier is not None
        # every candidate also gets a verifier score, so recovery thresholds can be swept exactly (RQ2)
        p_ver = _score_items(verifier, [dict(it, stage="verify") for it in rec_items], cfg, batch_size) if (use_verifier and rec_items) else [None] * len(rec_items)
        for it, pr, pv in zip(rec_items, p_rec, p_ver):
            kept = pr >= tau_r and (pv is None or pv >= tau_v)
            score = pv if pv is not None else pr
            decisions.append(Decision(tech, it["file"], it["line"], it["smell"], "recovered", pv, pr, kept, score))
    return decisions


def run_on_oracle(
    cfg: Config,
    verifier_dir: Optional[Path],
    recovery_dir: Optional[Path],
    run_name: str,
    verify_mode: Optional[str] = None,
    use_recovery: Optional[bool] = None,
    device: Optional[str] = None,
) -> Path:
    pcfg = cfg["pipeline"]
    verify_mode = verify_mode or pcfg.get("verify_smells", "all")
    use_recovery = pcfg.get("use_recovery", True) if use_recovery is None else use_recovery
    verifier = Scorer(verifier_dir, device=device) if verifier_dir else None
    recovery = Scorer(recovery_dir, device=device) if (recovery_dir and use_recovery) else None
    out_dir = ensure_dir(cfg.path("runs") / run_name)
    summary = {"verifier": str(verifier_dir) if verifier_dir else None, "recovery": str(recovery_dir) if recovery_dir else None,
               "verify_mode": verify_mode, "use_recovery": use_recovery, "techs": {}}
    for tech in TECHS:
        oracle = load_oracle(cfg, tech)
        scripts = {n: oracle.script_dir / n for n in oracle.files if oracle.has_script(n)}
        missing = [n for n in oracle.files if not oracle.has_script(n)]
        if missing:
            LOG.warning("[%s] %d oracle scripts missing - results for this tech are NOT comparable", tech, len(missing))
        stage1: Dict[str, Set[Tuple[int, str]]] = {}
        for w in glitch_oracle_warnings(cfg, tech):
            stage1.setdefault(w["file"], set()).add((w["line"], w["smell"]))
        decisions = run_on_scripts(
            cfg, tech, scripts, stage1, oracle.lines, verifier, recovery, verify_mode, use_recovery,
            bool(pcfg.get("recovery_requires_verifier", True)),
        )
        rows = [asdict(d) for d in decisions]
        write_csv(out_dir / "decisions" / f"{tech}.csv", rows)
        kept = [d for d in decisions if d.kept]
        write_csv(
            out_dir / "detections" / f"{tech}.csv",
            [{"PATH": d.file, "LINE": d.line, "CATEGORY": d.smell, "SCORE": round(d.score, 6), "SOURCE": d.source} for d in kept],
            ["PATH", "LINE", "CATEGORY", "SCORE", "SOURCE"],
        )
        summary["techs"][tech] = {
            "scripts": len(scripts),
            "missing_scripts": len(missing),
            "stage1_warnings": sum(len(v) for v in stage1.values()),
            "kept": len(kept),
            "kept_recovered": sum(1 for d in kept if d.source == "recovered"),
            "candidates_scored": sum(1 for d in decisions if d.source == "recovered"),
            "thresholds": {"verify": verifier.threshold(tech) if verifier else None, "recover": recovery.threshold(tech) if recovery else None},
        }
        LOG.info("[%s] kept %d detections (%d recovered)", tech, len(kept), summary["techs"][tech]["kept_recovered"])
    summary["inference_seconds"] = {"verifier": verifier.seconds if verifier else 0.0, "recovery": recovery.seconds if recovery else 0.0}
    write_json(out_dir / "run.json", summary)
    return out_dir


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verifier", type=Path, default=None)
    parser.add_argument("--recovery", type=Path, default=None)
    parser.add_argument("--name", required=True, help="Run name under runs/")
    parser.add_argument("--verify-mode", choices=["all", "intellisa4"], default=None)
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--device", default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    run_on_oracle(cfg, args.verifier, args.recovery, args.name, args.verify_mode, False if args.no_recovery else None, args.device)


if __name__ == "__main__":
    main()
