"""Index the GLITCH corpus (196,755 IaC scripts) and its precomputed GLITCH warnings."""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from resa.config import Config
from resa.data.oracle import TECH_EXTS, TECHS
from resa.smells import SMELLS, normalize_category
from resa.utils import ensure_dir, get_logger, iter_jsonl, md5_bytes, read_jsonl, write_json, write_jsonl

LOG = get_logger("resa.corpus")


def results_root(cfg: Config) -> Path:
    return cfg.path("raw") / "glitch_package" / "RESULTS"


def corpus_available(cfg: Config) -> bool:
    return results_root(cfg).exists()


def corpus_dir(cfg: Config) -> Path:
    return ensure_dir(cfg.path("processed") / "corpus")


def _tech_roots(cfg: Config, tech: str) -> List[Path]:
    base = results_root(cfg) / tech
    repos = base / "repos"
    return [repos] if repos.exists() else [base]


def _is_oracle_path(rel: str) -> bool:
    return "oracle" in rel.lower()


def index_corpus(cfg: Config, force: bool = False) -> Dict[str, int]:
    """Write ``processed/corpus/<tech>_files.jsonl`` with {rel, path, n_lines, md5} per script."""
    counts: Dict[str, int] = {}
    for tech in TECHS:
        out = corpus_dir(cfg) / f"{tech}_files.jsonl"
        if out.exists() and not force:
            counts[tech] = sum(1 for _ in iter_jsonl(out))
            continue
        base = results_root(cfg) / tech
        rows = []
        for root in _tech_roots(cfg, tech):
            for path in sorted(root.rglob("*")):
                if not path.is_file() or not path.name.lower().endswith(TECH_EXTS[tech]):
                    continue
                rel = path.relative_to(base).as_posix()
                if _is_oracle_path(rel):
                    continue
                data = path.read_bytes()
                rows.append({"rel": rel, "path": str(path), "n_lines": data.count(b"\n") + 1, "md5": md5_bytes(data)})
        limit = cfg["pools"].get("max_corpus_files_per_tech")
        if limit:
            import random

            rng = random.Random(cfg["seed"])
            rng.shuffle(rows)
            rows = sorted(rows[: int(limit)], key=lambda r: r["rel"])
        counts[tech] = write_jsonl(out, rows)
        LOG.info("[%s] indexed %d corpus scripts", tech, counts[tech])
    return counts


def load_corpus_index(cfg: Config, tech: str) -> List[dict]:
    path = corpus_dir(cfg) / f"{tech}_files.jsonl"
    return read_jsonl(path) if path.exists() else []


def flat_names(rel: str) -> List[str]:
    """Flat file names (IntelliSA style) a nested corpus path may be referred to by."""
    parts = rel.split("/")
    if parts and parts[0] == "repos":
        parts = parts[1:]
    names = []
    for start in range(len(parts)):
        names.append("-".join(parts[start:]))
        if start + 1 < len(parts):
            names.append("_".join(parts[start : start + 1]) + "-" + "-".join(parts[start + 1 :]))
    return names


def flat_name_index(cfg: Config, tech: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for row in load_corpus_index(cfg, tech):
        for name in flat_names(row["rel"]):
            index.setdefault(name, row["path"])
    return index


# ----------------------------------------------------------------------------- warnings
def _glitch_csvs(cfg: Config, tech: str) -> List[Path]:
    base = results_root(cfg) / tech
    found = sorted(p for p in base.rglob("GLITCH-*.csv") if "oracle" not in p.name.lower() and "SLAC" not in p.name and "SLIC" not in p.name)
    return found


def _columns(header: List[str]) -> Optional[Tuple[int, int, int]]:
    low = [h.strip().lower() for h in header]

    def find(*names):
        for name in names:
            if name in low:
                return low.index(name)
        return None

    p, l, e = find("path", "file", "filename"), find("line", "line_number"), find("error", "code", "smell", "category", "issue")
    if p is not None and l is not None and e is not None:
        return p, l, e
    return None


def _resolve(raw: str, tech: str, by_rel: Dict[str, str], by_suffix: Dict[str, str]) -> Optional[str]:
    path = raw.strip().replace("\\", "/")
    marker = f"/RESULTS/{tech}/"
    if marker in path:
        rel = path.split(marker, 1)[1]
        if rel in by_rel:
            return rel
    match = re.search(r"(?:^|/)(repos/.+)$", path)
    if match and match.group(1) in by_rel:
        return match.group(1)
    trimmed = re.sub(r"^(\.\./|\./)+", "", path)
    if trimmed in by_rel:
        return trimmed
    tail = "/".join(trimmed.split("/")[-4:])
    return by_suffix.get(tail)


def load_package_warnings(cfg: Config, tech: str) -> Tuple[List[dict], Dict[str, float]]:
    """Parse the replication package's GLITCH corpus CSV(s) into {rel, line, smell} records."""
    index = load_corpus_index(cfg, tech)
    by_rel = {r["rel"]: r["path"] for r in index}
    by_suffix: Dict[str, str] = {}
    for rel in by_rel:
        by_suffix.setdefault("/".join(rel.split("/")[-4:]), rel)
    rows: Dict[Tuple[str, int, str], dict] = {}
    total = resolved = 0
    for csv_path in _glitch_csvs(cfg, tech):
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                continue
            cols = _columns(header)
            first_rows = [] if cols else [header]
            cols = cols or (0, 1, 2)
            for record in first_rows + list(reader):
                if len(record) <= max(cols):
                    continue
                total += 1
                smell = normalize_category(record[cols[2]])
                if smell not in SMELLS:
                    continue
                try:
                    line = int(float(record[cols[1]]))
                except ValueError:
                    continue
                rel = _resolve(record[cols[0]], tech, by_rel, by_suffix)
                if rel is None:
                    continue
                resolved += 1
                key = (rel, max(line, 1), smell)
                rows.setdefault(key, {"rel": rel, "line": key[1], "smell": smell, "raw": record[cols[2]].strip()})
    stats = {"csv_files": len(_glitch_csvs(cfg, tech)), "rows": total, "resolved_rows": resolved,
             "resolution_rate": (resolved / total) if total else 0.0, "warnings": len(rows)}
    return sorted(rows.values(), key=lambda r: (r["rel"], r["line"], r["smell"])), stats


def run_glitch_warnings(cfg: Config, tech: str, rels: Iterable[str], workers: int) -> List[dict]:
    from resa.data.oracle import cli_dir
    from resa.glitch_runner import scan_files

    base = results_root(cfg) / tech
    rels = list(rels)
    path_to_rel = {str(base / rel): rel for rel in rels}
    out = []
    failed = Counter()
    for path, found, _secs, err in scan_files(cli_dir(cfg), tech, list(path_to_rel), workers=workers):
        if err:
            failed[err] += 1
        for line, smell, code in found:
            out.append({"rel": path_to_rel[path], "line": line, "smell": smell, "raw": code})
    if failed:
        LOG.warning("[%s] GLITCH failed on %d scripts: %s", tech, sum(failed.values()), dict(failed))
    return sorted(out, key=lambda r: (r["rel"], r["line"], r["smell"]))


def corpus_warnings(cfg: Config, tech: str, source: str = "auto", rels_if_run: Optional[List[str]] = None, workers: int = 8) -> List[dict]:
    """Stage-1 warnings for corpus scripts, cached in ``processed/corpus/<tech>_glitch.jsonl``."""
    cache = corpus_dir(cfg) / f"{tech}_glitch.jsonl"
    meta = corpus_dir(cfg) / f"{tech}_glitch.meta.json"
    if cache.exists():
        return read_jsonl(cache)
    stats: Dict[str, object] = {}
    rows: List[dict] = []
    if source in ("auto", "package"):
        rows, pstats = load_package_warnings(cfg, tech)
        stats.update(package=pstats)
        if source == "package" and not rows:
            raise RuntimeError(f"[{tech}] no usable warnings in the package CSVs: {pstats}")
        if source == "auto" and (pstats["resolution_rate"] < 0.5 or not rows):
            LOG.warning("[%s] package CSV unusable (%s); running GLITCH ourselves", tech, pstats)
            rows = []
    if not rows:
        if rels_if_run is None:
            raise RuntimeError("rels_if_run is required when GLITCH must be run")
        rows = run_glitch_warnings(cfg, tech, rels_if_run, workers)
        stats.update(source="run", scanned_files=len(rels_if_run))
    else:
        stats.update(source="package")
    write_jsonl(cache, rows)
    stats["warnings_by_smell"] = dict(Counter(r["smell"] for r in rows))
    write_json(meta, stats)
    LOG.info("[%s] %d corpus GLITCH warnings (%s)", tech, len(rows), stats.get("source"))
    return rows


def warnings_by_file(rows: List[dict]) -> Dict[str, set]:
    grouped: Dict[str, set] = defaultdict(set)
    for r in rows:
        grouped[r["rel"]].add((int(r["line"]), r["smell"]))
    return grouped
