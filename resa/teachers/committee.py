"""Open-weight teacher committee: resumable labelling, aggregation and agreement statistics.

    python -m resa.teachers.committee label   --stage verify     # every teacher, cached + resumable
    python -m resa.teachers.committee label   --stage recover
    python -m resa.teachers.committee aggregate --stage verify   # -> labels/verify/committee.jsonl
    python -m resa.teachers.committee oracle-eval                # teacher quality on GLITCH oracle warnings (RQ3)
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from resa.config import Config, add_config_args, config_from_args
from resa.items import item_context
from resa.teachers.backends import make_backend
from resa.teachers.prompts import build_prompt
from resa.utils import append_jsonl, ensure_dir, get_logger, read_json, read_jsonl, sha1_text, write_json, write_jsonl

LOG = get_logger("resa.committee")
TEACHER_WINDOW = 5  # IntelliSA's teacher saw a 5-line window; we use +/-5 for every teacher


def pool_path(cfg: Config, stage: str) -> Path:
    return cfg.path("processed") / "pools" / f"{stage}.jsonl"


def labels_dir(cfg: Config, stage: str) -> Path:
    return ensure_dir(cfg.path("labels") / stage)


def prompt_for(item: Dict) -> str:
    context = item_context(item, TEACHER_WINDOW, "lines", 2400, True)
    return build_prompt(item["stage"], item["tech"], item["smell"], item["file"], int(item["line"]), context, TEACHER_WINDOW)


def label_items(cfg: Config, items: List[Dict], out_dir: Path, teachers: Optional[List[str]] = None, chunk: int = 512) -> None:
    tcfg = dict(cfg["teachers"])
    for model_cfg in tcfg["models"]:
        if teachers and model_cfg["name"] not in teachers:
            continue
        cache = out_dir / f"{model_cfg['name']}.jsonl"
        done = {row["id"] for row in read_jsonl(cache)} if cache.exists() else set()
        todo = [it for it in items if it["id"] not in done]
        LOG.info("[%s] %d/%d items already labelled, %d to go", model_cfg["name"], len(done), len(items), len(todo))
        if not todo:
            continue
        backend = make_backend(model_cfg, tcfg)
        timing_path = out_dir / f"{model_cfg['name']}.timing.json"
        timing = read_json(timing_path) if timing_path.exists() else {"seconds": 0.0, "items": 0, "prompt_chars": 0}
        try:
            for start in range(0, len(todo), chunk):
                batch = todo[start : start + chunk]
                prompts = [prompt_for(it) for it in batch]
                t0 = time.perf_counter()
                scores = backend.score(prompts)
                timing["seconds"] += time.perf_counter() - t0
                timing["items"] += len(batch)
                timing["prompt_chars"] += sum(len(p) for p in prompts)
                for it, prompt, sc in zip(batch, prompts, scores):
                    append_jsonl(
                        cache,
                        {"id": it["id"], "p_yes": sc["p_yes"], "text": sc["text"], "prompt_sha1": sha1_text(prompt)},
                    )
                write_json(timing_path, timing)
                LOG.info("[%s] %d/%d", model_cfg["name"], min(start + chunk, len(todo)), len(todo))
        finally:
            backend.close()


def _fleiss_kappa(matrix: np.ndarray) -> float:
    """matrix: items x categories counts (each row sums to number of raters)."""
    n = matrix.sum(axis=1)
    if len(matrix) == 0 or np.any(n != n[0]) or n[0] < 2:
        return float("nan")
    r = n[0]
    p_j = matrix.sum(axis=0) / (len(matrix) * r)
    p_i = (np.square(matrix).sum(axis=1) - r) / (r * (r - 1))
    p_bar, p_e = p_i.mean(), np.square(p_j).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def _cohen_kappa(a: List[int], b: List[int]) -> float:
    if not a:
        return float("nan")
    a_arr, b_arr = np.array(a), np.array(b)
    po = float((a_arr == b_arr).mean())
    pe = float(a_arr.mean() * b_arr.mean() + (1 - a_arr.mean()) * (1 - b_arr.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def load_teacher_scores(out_dir: Path, names: Iterable[str]) -> Dict[str, Dict[str, Optional[float]]]:
    scores: Dict[str, Dict[str, Optional[float]]] = {}
    for name in names:
        path = out_dir / f"{name}.jsonl"
        if path.exists():
            scores[name] = {row["id"]: row.get("p_yes") for row in read_jsonl(path)}
    return scores


def aggregate(cfg: Config, items: List[Dict], out_dir: Path, teachers: Optional[List[str]] = None) -> Dict:
    tcfg = cfg["teachers"]
    names = [m["name"] for m in tcfg["models"] if not teachers or m["name"] in teachers]
    scores = load_teacher_scores(out_dir, names)
    names = [n for n in names if n in scores]
    if not names:
        raise RuntimeError(f"No teacher labels found in {out_dir}; run `label` first")
    thr = float(tcfg.get("decision_threshold", 0.5))
    mode = tcfg.get("aggregation", "soft")
    rows, votes_matrix, abstain = [], [], Counter()
    for it in items:
        ps = {n: scores[n].get(it["id"]) for n in names}
        valid = [p for p in ps.values() if p is not None]
        for n, p in ps.items():
            if p is None:
                abstain[n] += 1
        if not valid:
            continue
        p_mean = float(np.mean(valid))
        yes_votes = sum(1 for p in valid if p >= 0.5)
        if mode == "majority":
            label = 1 if yes_votes * 2 > len(valid) else (1 if yes_votes * 2 == len(valid) and p_mean >= thr else 0)
        else:
            label = 1 if p_mean >= thr else 0
        rows.append({"id": it["id"], "p_yes": p_mean, "label": label, "votes": {n: p for n, p in ps.items()}, "n_teachers": len(valid)})
        if len(valid) == len(names):
            votes_matrix.append([len(valid) - yes_votes, yes_votes])
    write_jsonl(out_dir / "committee.jsonl", rows)
    # agreement statistics
    pair = {}
    for a, b in itertools.combinations(names, 2):
        common = [i for i in scores[a] if i in scores[b] and scores[a][i] is not None and scores[b][i] is not None]
        pair[f"{a}~{b}"] = _cohen_kappa([int(scores[a][i] >= 0.5) for i in common], [int(scores[b][i] >= 0.5) for i in common])
    stats = {
        "teachers": names,
        "items": len(items),
        "labelled": len(rows),
        "positive_rate": float(np.mean([r["label"] for r in rows])) if rows else None,
        "fleiss_kappa": _fleiss_kappa(np.array(votes_matrix)) if votes_matrix else None,
        "pairwise_cohen_kappa": pair,
        "abstentions": dict(abstain),
        "unanimous_rate": float(np.mean([1 if (v[0] == 0 or v[1] == 0) else 0 for v in votes_matrix])) if votes_matrix else None,
    }
    # agreement with IntelliSA's Claude-4 labels where available
    row_by_id = {r["id"]: r for r in rows}
    claude = [(it["claude_label"], row_by_id.get(it["id"])) for it in items if it.get("claude_label")]
    claude = [(1 if c == "TP" else 0, r) for c, r in claude if r is not None]
    if claude:
        y_claude = [c for c, _ in claude]
        y_comm = [r["label"] for _, r in claude]
        stats["vs_claude"] = {
            "n": len(claude),
            "agreement": float(np.mean([a == b for a, b in zip(y_claude, y_comm)])),
            "cohen_kappa": _cohen_kappa(y_claude, y_comm),
            "committee_tp_rate": float(np.mean(y_comm)),
            "claude_tp_rate": float(np.mean(y_claude)),
        }
        for n in names:
            pairs = [(1 if it["claude_label"] == "TP" else 0, scores[n].get(it["id"])) for it in items if it.get("claude_label")]
            pairs = [(c, int(p >= 0.5)) for c, p in pairs if p is not None]
            if pairs:
                stats["vs_claude"][f"kappa_{n}"] = _cohen_kappa([c for c, _ in pairs], [p for _, p in pairs])
    write_json(out_dir / "agreement.json", stats)
    LOG.info("Committee for %s: %s", out_dir.name, {k: v for k, v in stats.items() if k != "pairwise_cohen_kappa"})
    return stats


def oracle_eval_items(cfg: Config, only_intellisa4: bool = False) -> List[Dict]:
    """GLITCH's published oracle warnings, labelled TP/FP by the human oracle (evaluation only)."""
    from resa.data.oracle import TECHS, glitch_oracle_warnings, load_oracle
    from resa.items import item_id
    from resa.smells import INTELLISA_FILTERED

    items = []
    for tech in TECHS:
        oracle = load_oracle(cfg, tech)
        for w in glitch_oracle_warnings(cfg, tech):
            if only_intellisa4 and w["smell"] not in INTELLISA_FILTERED:
                continue
            path = oracle.script_dir / w["file"]
            items.append(
                {
                    "id": item_id("verify", tech, w["file"], w["line"], w["smell"]),
                    "stage": "verify",
                    "tech": tech,
                    "file": w["file"],
                    "path": str(path) if path.exists() else None,
                    "line": w["line"],
                    "smell": w["smell"],
                    "source": "glitch_oracle",
                    "oracle_label": int((w["file"], w["line"], w["smell"]) in oracle.truth),
                }
            )
    return items


def oracle_eval(cfg: Config, teachers: Optional[List[str]] = None) -> Dict:
    """Teacher quality on the oracle (IntelliSA Table 4 analogue). Oracle is used for *evaluation only*."""
    items = [it for it in oracle_eval_items(cfg) if it["path"]]
    out_dir = labels_dir(cfg, "oracle_eval")
    label_items(cfg, items, out_dir, teachers)
    names = [m["name"] for m in cfg["teachers"]["models"] if not teachers or m["name"] in teachers]
    scores = load_teacher_scores(out_dir, names)
    from resa.smells import INTELLISA_FILTERED

    report: Dict[str, Dict] = {}
    for subset_name, subset in (("all9", items), ("intellisa4", [i for i in items if i["smell"] in INTELLISA_FILTERED])):
        y_true_fp = [1 - it["oracle_label"] for it in subset]  # IntelliSA Table 4: positive class = false positive
        for name in list(scores) + ["committee"]:
            if name == "committee":
                valid = [[scores[n].get(it["id"]) for n in scores] for it in subset]
                ps = [float(np.mean([p for p in v if p is not None])) if any(p is not None for p in v) else None for v in valid]
            else:
                ps = [scores[name].get(it["id"]) for it in subset]
            pairs = [(t, int(p < 0.5)) for t, p in zip(y_true_fp, ps) if p is not None]
            report.setdefault(subset_name, {})[name] = _binary_metrics([t for t, _ in pairs], [p for _, p in pairs])
            report[subset_name][name]["n"] = len(pairs)
    report["reference_intellisa_table4_fp_detection"] = {
        "claude-4": {"precision": 0.924, "recall": 0.853, "f1": 0.887},
        "gpt-5": {"precision": 0.952, "recall": 0.692, "f1": 0.802},
        "grok-4": {"precision": 0.894, "recall": 0.531, "f1": 0.667},
    }
    write_json(cfg.path("results") / "rq3_teacher_quality_oracle.json", report)
    return report


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["label", "aggregate", "oracle-eval"])
    parser.add_argument("--stage", choices=["verify", "recover"], default="verify")
    parser.add_argument("--teachers", nargs="*", default=None, help="Subset of teacher names (default: all)")
    parser.add_argument("--chunk", type=int, default=512)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if args.action == "oracle-eval":
        oracle_eval(cfg, args.teachers)
        return
    items = read_jsonl(pool_path(cfg, args.stage))
    out_dir = labels_dir(cfg, args.stage)
    if args.action == "label":
        label_items(cfg, items, out_dir, args.teachers, args.chunk)
        if not args.teachers:  # a per-teacher job (e.g. SLURM) aggregates later, once all teachers finished
            aggregate(cfg, items, out_dir)
    else:
        aggregate(cfg, items, out_dir, args.teachers)


if __name__ == "__main__":
    main()
