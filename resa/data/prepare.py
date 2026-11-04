"""Build every dataset the experiments need (no manual labelling; teachers label the pools).

    python -m resa.data.prepare                     # oracle + corpus pools
    python -m resa.data.prepare --no-corpus         # oracle + IntelliSA items only (quick start)

Outputs (data/processed/):
    oracle/<tech>/*            the 241 test scripts (validated against IntelliSA's |U|)
    stage1/<tech>.jsonl        GLITCH's published warnings on the oracle (ReSA stage 1 at test time)
    pools/verify.jsonl         verifier items: corpus GLITCH warnings (9 smells) + IntelliSA items
    pools/recover.jsonl        recovery items: relaxed candidates GLITCH did not flag
    pools/summary.json         counts, leakage removals, split sizes
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from resa.candidates import generate_candidates
from resa.config import Config, add_config_args, config_from_args
from resa.data import corpus as corpus_mod
from resa.data.oracle import TECHS, glitch_oracle_warnings, hub_dir, load_oracle, materialize_oracle
from resa.items import item_id
from resa.smells import INTELLISA_TRAIN_SMELLS, SMELLS
from resa.utils import ensure_dir, get_logger, md5_bytes, normalize_snippet, read_jsonl, read_lines, sha1_text, write_json, write_jsonl

LOG = get_logger("resa.prepare")
EXT_TECH = {".pp": "puppet", ".rb": "chef", ".yml": "ansible", ".yaml": "ansible"}


def _split_for(file_key: str, val_frac: float, seed: int) -> str:
    bucket = int(sha1_text(f"{seed}|{file_key}")[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_frac else "train"


class Leakage:
    """Everything that must never appear in training data: oracle files and oracle decisions."""

    def __init__(self, cfg: Config):
        self.md5s: Set[str] = set()
        self.names: Set[str] = set()
        self.decisions: Set[Tuple[str, str]] = set()  # (normalised line text, smell)
        for tech in TECHS:
            oracle = load_oracle(cfg, tech)
            warn = {(w["file"], w["line"], w["smell"]) for w in glitch_oracle_warnings(cfg, tech)}
            for name in oracle.files:
                self.names.add(name)
                if not oracle.has_script(name):
                    continue
                path = oracle.script_dir / name
                self.md5s.add(md5_bytes(path.read_bytes()))
                lines = oracle.lines(name)
                file_decisions = {(l, s) for (f, l, s) in oracle.truth | warn if f == name}
                exclude = {(l, s) for (f, l, s) in warn if f == name}
                file_decisions |= {(c.line, c.smell) for c in generate_candidates(tech, lines, exclude)}
                for line, smell in file_decisions:
                    if 1 <= line <= len(lines):
                        self.decisions.add((normalize_snippet(lines[line - 1]), smell))
        self.removed = Counter()

    def file_ok(self, md5: Optional[str], names: Iterable[str]) -> bool:
        if md5 and md5 in self.md5s:
            self.removed["file_md5"] += 1
            return False
        if any(n in self.names for n in names):
            self.removed["file_name"] += 1
            return False
        return True

    def item_ok(self, line_text: str, smell: str) -> bool:
        if (normalize_snippet(line_text), smell) in self.decisions:
            self.removed["decision"] += 1
            return False
        return True


def _stratified(items: List[dict], cap: int, seed: int) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for it in items:
        groups[(it["tech"], it["smell"])].append(it)
    rng = random.Random(seed)
    out = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["id"])
        rng.shuffle(group)
        out.extend(group[:cap])
    return sorted(out, key=lambda r: r["id"])


def intellisa_items(cfg: Config, leak: Leakage) -> List[dict]:
    """IntelliSA's released Claude-4-labelled train/val items (never its oracle-derived test split)."""
    base = hub_dir(cfg) / "replication" / "datasets" / "training" / "combined"
    flat_idx = {t: corpus_mod.flat_name_index(cfg, t) for t in TECHS} if corpus_mod.corpus_available(cfg) else {}
    out, unresolved = [], 0
    for split in ("train", "val"):
        path = base / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            tech = EXT_TECH.get(Path(row["file"]).suffix.lower())
            smell = INTELLISA_TRAIN_SMELLS.get(row["smell"])
            if tech is None or smell is None:
                continue
            if row["file"] in leak.names or not leak.item_ok(row.get("content", ""), smell):
                continue
            resolved = flat_idx.get(tech, {}).get(row["file"])
            if resolved:
                lines = read_lines(resolved)
                line = int(row["line"])
                if not (1 <= line <= len(lines)) or normalize_snippet(lines[line - 1]) != normalize_snippet(row.get("content", "")):
                    resolved = None  # file found but content differs -> fall back to the released context
            if not resolved:
                unresolved += 1
            out.append(
                {
                    "id": item_id("verify", tech, row["file"], int(row["line"]), smell),
                    "stage": "verify",
                    "tech": tech,
                    "file": row["file"],
                    "path": resolved,
                    "line": int(row["line"]),
                    "smell": smell,
                    "source": "intellisa",
                    "rule": "glitch",
                    "split": split,
                    "claude_label": row["label"],
                    "content": row.get("content", ""),
                    "context_fallback": row.get("with_context", ""),
                }
            )
    LOG.info("IntelliSA items: %d (%d without a corpus file; their released context is used)", len(out), unresolved)
    return out


def _sample_files(rows: List[dict], n: Optional[int], max_lines: int, seed: int) -> List[dict]:
    eligible = [r for r in rows if r["n_lines"] <= max_lines]
    rng = random.Random(seed)
    eligible = sorted(eligible, key=lambda r: r["rel"])
    rng.shuffle(eligible)
    return eligible if n is None else eligible[: int(n)]


def corpus_pools(cfg: Config, leak: Leakage, glitch_source: str, workers: int) -> Tuple[List[dict], List[dict], dict]:
    pcfg = cfg["pools"]
    seed = int(cfg["seed"])
    val_frac = float(pcfg["split"]["val"])
    verify, recover, stats = [], [], {}
    corpus_mod.index_corpus(cfg)
    for tech in TECHS:
        index = corpus_mod.load_corpus_index(cfg, tech)
        if not index:
            LOG.warning("[%s] corpus empty; skipping", tech)
            continue
        by_rel = {r["rel"]: r for r in index}
        vfiles = _sample_files(index, pcfg["verifier"]["files_per_tech_if_run"], pcfg["verifier"]["max_file_lines"], seed)
        rfiles = _sample_files(index, pcfg["recovery"]["files_per_tech"], pcfg["recovery"]["max_file_lines"], seed + 1)
        scan_rels = sorted({r["rel"] for r in vfiles} | {r["rel"] for r in rfiles})
        warnings = corpus_mod.corpus_warnings(cfg, tech, glitch_source, scan_rels, workers)
        meta = corpus_mod.corpus_dir(cfg) / f"{tech}_glitch.meta.json"
        ran_ourselves = meta.exists() and '"source": "run"' in meta.read_text()
        by_file = corpus_mod.warnings_by_file(warnings)
        # ---- verifier pool: GLITCH warnings
        vcands = []
        file_ok: Dict[str, bool] = {}
        lines_cache: Dict[str, List[str]] = {}
        for w in warnings:  # sorted by file, so each script is read once
            row = by_rel.get(w["rel"])
            if row is None or row["n_lines"] > pcfg["verifier"]["max_file_lines"]:
                continue
            if w["rel"] not in file_ok:
                file_ok[w["rel"]] = leak.file_ok(row["md5"], corpus_mod.flat_names(row["rel"]))
                lines_cache.clear()
            if not file_ok[w["rel"]]:
                continue
            if w["rel"] not in lines_cache:
                lines_cache[w["rel"]] = read_lines(row["path"])
            lines = lines_cache[w["rel"]]
            line = int(w["line"])
            if not (1 <= line <= len(lines)) or not lines[line - 1].strip():
                continue
            if not leak.item_ok(lines[line - 1], w["smell"]):
                continue
            vcands.append(
                {
                    "id": item_id("verify", tech, w["rel"], line, w["smell"]),
                    "stage": "verify",
                    "tech": tech,
                    "file": w["rel"],
                    "path": row["path"],
                    "line": line,
                    "smell": w["smell"],
                    "source": "glitch",
                    "rule": w.get("raw", ""),
                    "split": _split_for(w["rel"], val_frac, seed),
                }
            )
        vsel = _stratified(vcands, int(pcfg["verifier"]["per_tech_smell_cap"]), seed)
        verify.extend(vsel)
        # ---- recovery pool: relaxed candidates on lines GLITCH did not flag
        rcands = []
        recovery_files = rfiles if not ran_ourselves else [by_rel[r] for r in scan_rels if r in by_rel and by_rel[r]["n_lines"] <= pcfg["recovery"]["max_file_lines"]]
        for row in recovery_files:
            if not leak.file_ok(row["md5"], corpus_mod.flat_names(row["rel"])):
                continue
            lines = read_lines(row["path"])
            for cand in generate_candidates(tech, lines, by_file.get(row["rel"], set())):
                if not leak.item_ok(lines[cand.line - 1], cand.smell):
                    continue
                rcands.append(
                    {
                        "id": item_id("recover", tech, row["rel"], cand.line, cand.smell),
                        "stage": "recover",
                        "tech": tech,
                        "file": row["rel"],
                        "path": row["path"],
                        "line": cand.line,
                        "smell": cand.smell,
                        "source": "candidate",
                        "rule": cand.rule,
                        "split": _split_for(row["rel"], val_frac, seed),
                    }
                )
        rsel = _stratified(rcands, int(pcfg["recovery"]["per_tech_smell_cap"]), seed)
        recover.extend(rsel)
        stats[tech] = {
            "corpus_files": len(index),
            "glitch_warnings": len(warnings),
            "verify_candidates": len(vcands),
            "verify_selected": len(vsel),
            "recover_files_scanned": len(recovery_files),
            "recover_candidates": len(rcands),
            "recover_selected": len(rsel),
            "verify_by_smell": dict(Counter(i["smell"] for i in vsel)),
            "recover_by_smell": dict(Counter(i["smell"] for i in rsel)),
        }
        LOG.info("[%s] verify %d / recover %d items", tech, len(vsel), len(rsel))
    return verify, recover, stats


def write_stage1(cfg: Config) -> None:
    out = ensure_dir(cfg.path("processed") / "stage1")
    for tech in TECHS:
        write_jsonl(out / f"{tech}.jsonl", glitch_oracle_warnings(cfg, tech))


def prepare(cfg: Config, use_corpus: bool = True, allow_missing_oracle: bool = False, glitch_source: str = "auto", workers: int = 8) -> dict:
    report = {"oracle": materialize_oracle(cfg, allow_missing=allow_missing_oracle)}
    write_stage1(cfg)
    leak = Leakage(cfg)
    verify: List[dict] = []
    recover: List[dict] = []
    if use_corpus:
        if not corpus_mod.corpus_available(cfg):
            raise RuntimeError("GLITCH corpus not extracted; run `python -m resa.data.fetch all` or pass --no-corpus")
        verify, recover, report["corpus"] = corpus_pools(cfg, leak, glitch_source, workers)
    if cfg["pools"]["verifier"].get("include_intellisa_items", True):
        isa = intellisa_items(cfg, leak)
        known = {it["id"] for it in verify}
        verify.extend(it for it in isa if it["id"] not in known)
    pools = ensure_dir(cfg.path("processed") / "pools")
    write_jsonl(pools / "verify.jsonl", verify)
    write_jsonl(pools / "recover.jsonl", recover)
    report["pools"] = {
        "verify": dict(Counter(f"{i['split']}:{i['source']}" for i in verify)),
        "recover": dict(Counter(i["split"] for i in recover)),
        "verify_total": len(verify),
        "recover_total": len(recover),
        "leakage_removed": dict(leak.removed),
    }
    write_json(pools / "summary.json", report)
    LOG.info("Pools written: verify=%d recover=%d (leakage removed: %s)", len(verify), len(recover), dict(leak.removed))
    return report


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-corpus", action="store_true", help="Skip corpus pools (IntelliSA items only)")
    parser.add_argument("--allow-missing-oracle", action="store_true", help="Do not fail if oracle scripts are missing (tests only)")
    parser.add_argument("--glitch-source", choices=["auto", "package", "run"], default="auto")
    parser.add_argument("--workers", type=int, default=8)
    add_config_args(parser)
    args = parser.parse_args(argv)
    prepare(config_from_args(args), not args.no_corpus, args.allow_missing_oracle, args.glitch_source, args.workers)


if __name__ == "__main__":
    main()
