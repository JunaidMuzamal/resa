"""RQ3 - design choices (each varied alone, everything else at the default):

  A  context window of the verifier          0 (line only, = IntelliSA) / 3 / 5 / 10 / enclosing block
  B  label source on the same items          open-weight committee vs each single teacher vs Claude-4
  C  class / rare-smell rebalancing          none / class_weighted / smell_weighted
  D  student backbone                        CodeT5p-220M / 770M, CodeBERT, UniXcoder
  E  verified smell coverage                 all 9 vs IntelliSA's 4
  F  teacher quality on the oracle           IntelliSA Table 4 analogue (evaluation only)

    python -m resa.experiments.rq3 --parts A B C D E F
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional

from resa.config import add_config_args, config_from_args
from resa.experiments.common import aggregate, ensure_model, run_and_eval, save_table, variant
from resa.smells import SMELLS
from resa.utils import get_logger, read_json, write_json

LOG = get_logger("resa.rq3")
COLS = ["method", "puppet_f1", "ansible_f1", "chef_f1", "macro_f1"]
HEADS = ["Variant", "Puppet F1", "Ansible F1", "Chef F1", "Macro-F1"]


def _seeds(cfg) -> List[int]:
    return list(cfg["experiments"]["ablation_seeds"])


def _condition(cfg, label: str, tag: str, verify_kwargs: Optional[dict] = None, use_recovery: bool = True,
               verify_mode: Optional[str] = None, recovery_cfg=None) -> Dict:
    rows = []
    for seed in _seeds(cfg):
        verifier = ensure_model(cfg, "verify", seed, **(verify_kwargs or {}))
        recovery = ensure_model(recovery_cfg or cfg, "recover", seed) if use_recovery else None
        rows.append(run_and_eval(cfg, verifier, recovery, f"rq3__{tag}__s{seed}", verify_mode, use_recovery))
    out = aggregate(rows, label)
    LOG.info("%s: Macro-F1 %.3f", label, out.get("macro_f1", float("nan")))
    return out


def part_a(cfg) -> List[Dict]:
    rows = []
    for w in cfg["experiments"]["context_windows"]:
        v = variant(cfg, **{"context.window": int(w), "context.mode": "lines"})
        rows.append(_condition(v, f"window +/-{w}" + (" (IntelliSA input)" if w == 0 else ""), f"ctx{w}", recovery_cfg=cfg))
    v = variant(cfg, **{"context.mode": "block"})
    rows.append(_condition(v, "enclosing block", "ctxblock", recovery_cfg=cfg))
    save_table(cfg, "rq3a_context", rows, COLS, HEADS, "Verifier context window (recovery fixed).", bold_best=COLS[1:])
    return rows


def part_b(cfg) -> List[Dict]:
    """All label sources trained on the *same* items (IntelliSA's released items), verifier only."""
    rows = []
    sources = ["committee"] + [f"teacher:{m['name']}" for m in cfg["teachers"]["models"]] + ["claude"]
    for src in sources:
        tag = "labels_" + src.replace(":", "-")
        rows.append(_condition(cfg, f"labels: {src}", tag, verify_kwargs={"label_source": src, "sources": ["intellisa"], "tag": "isaitems"},
                               use_recovery=False, verify_mode="intellisa4"))
    agreement = cfg.path("labels") / "verify" / "agreement.json"
    if agreement.exists():
        write_json(cfg.path("results") / "rq3b_label_agreement.json", read_json(agreement))
    save_table(cfg, "rq3b_label_source", rows, COLS, HEADS,
               "Teacher labels on identical training items (IntelliSA's released items, 4 smells, no recovery).", bold_best=COLS[1:])
    return rows


def part_c(cfg) -> List[Dict]:
    rows = []
    for mode in cfg["experiments"]["rebalance_modes"]:
        v = variant(cfg, **{"student.rebalance": mode})
        row = _condition(v, f"rebalance: {mode}", f"reb_{mode}", recovery_cfg=v)
        rows.append(row)
    save_table(cfg, "rq3c_rebalance", rows, COLS, HEADS, "Rebalancing for rare smells (verifier and recovery).", bold_best=COLS[1:])
    return rows


def part_d(cfg) -> List[Dict]:
    rows = []
    for model in cfg["experiments"]["backbones"]:
        v = variant(cfg, **{"student.model_name": model})
        rows.append(_condition(v, model.split("/")[-1], "bb_" + model.split("/")[-1], recovery_cfg=v))
    save_table(cfg, "rq3d_backbone", rows, COLS, HEADS, "Student backbone (verifier and recovery).", bold_best=COLS[1:])
    return rows


def part_e(cfg) -> List[Dict]:
    rows = [
        _condition(cfg, "verify all 9 smells", "cov_all9", verify_mode="all"),
        _condition(cfg, "verify IntelliSA's 4 smells", "cov_isa4", verify_mode="intellisa4"),
    ]
    save_table(cfg, "rq3e_coverage", rows, COLS, HEADS, "Smell coverage of the verifier.", bold_best=COLS[1:])
    return rows


def part_f(cfg) -> Dict:
    from resa.teachers.committee import oracle_eval

    report = oracle_eval(cfg)
    rows = []
    for subset in ("intellisa4", "all9"):
        for name, m in report.get(subset, {}).items():
            rows.append({"subset": subset, "teacher": name, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "n": m["n"]})
    for name, m in report["reference_intellisa_table4_fp_detection"].items():
        rows.append({"subset": "intellisa4 (published)", "teacher": name, **m, "n": 230})
    save_table(cfg, "rq3f_teacher_quality", rows, ["subset", "teacher", "precision", "recall", "f1", "n"],
               ["Subset", "Teacher", "P (FP class)", "R (FP class)", "F1", "n"],
               "Teacher quality at detecting GLITCH false positives on the oracle (evaluation only; IntelliSA Table 4 layout).")
    return report


PARTS = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e, "F": part_f}


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="*", default=list(PARTS), choices=list(PARTS))
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    for p in args.parts:
        LOG.info("RQ3 part %s", p)
        PARTS[p](cfg)


if __name__ == "__main__":
    main()
