"""Human-labelled oracle (241 scripts, 213 smells) and IntelliSA's published baseline outputs.

Nothing here reruns a baseline: GLITCH, SLAC/SLIC, the six zero-shot LLM runs and IntelliSA
are all loaded from the detection files IntelliSA released with its replication package.
"""
from __future__ import annotations

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from resa.config import Config
from resa.smells import SMELLS, normalize_category
from resa.utils import ensure_dir, get_logger, md5_bytes, read_csv, read_json, read_lines, write_json

LOG = get_logger("resa.oracle")
_LINE_RE = re.compile(r"-?\d+")

Event = Tuple[str, int, str]
TECHS = ("ansible", "chef", "puppet")
TECH_EXTS = {"ansible": (".yml", ".yaml"), "chef": (".rb",), "puppet": (".pp",)}

# Table 2 of IntelliSA (arXiv 2601.14595): (precision, recall, F1) per technology.
# Used only to *check* that our recomputation from their released detection files matches.
PUBLISHED_TABLE2: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "GLITCH": {"puppet": (0.418, 0.938, 0.578), "ansible": (0.741, 0.977, 0.843), "chef": (0.603, 0.731, 0.661)},
    "SLAC/SLIC": {"puppet": (0.275, 0.508, 0.357), "ansible": (0.471, 0.545, 0.505), "chef": (0.367, 0.385, 0.376)},
    "Claude-4 (rules)": {"puppet": (0.500, 0.615, 0.552), "ansible": (0.508, 0.750, 0.606), "chef": (0.457, 0.606, 0.521)},
    "Claude-4 (def)": {"puppet": (0.623, 0.738, 0.676), "ansible": (0.610, 0.818, 0.699), "chef": (0.490, 0.490, 0.490)},
    "Grok-4 (rules)": {"puppet": (0.718, 0.785, 0.750), "ansible": (0.833, 0.909, 0.870), "chef": (0.667, 0.635, 0.650)},
    "Grok-4 (def)": {"puppet": (0.667, 0.738, 0.701), "ansible": (0.857, 0.682, 0.759), "chef": (0.625, 0.577, 0.600)},
    "GPT-5 (rules)": {"puppet": (0.542, 0.692, 0.608), "ansible": (0.678, 0.909, 0.777), "chef": (0.554, 0.596, 0.574)},
    "GPT-5 (def)": {"puppet": (0.593, 0.538, 0.565), "ansible": (0.741, 0.455, 0.563), "chef": (0.655, 0.346, 0.453)},
    "IntelliSA": {"puppet": (0.846, 0.846, 0.846), "ansible": (0.947, 0.818, 0.878), "chef": (0.877, 0.683, 0.768)},
}

# Detection directory (IntelliSA RQ1/data/detections) for each published LLM baseline
LLM_DETECTION_DIRS: Dict[str, str] = {
    "Claude-4 (rules)": "anthropic__claude-sonnet-4-20250514__static_analysis_rules",
    "Claude-4 (def)": "anthropic__claude-sonnet-4-20250514__definition_based",
    "Grok-4 (rules)": "openrouter__x-ai_grok-4-fast_free__static_analysis_rules",
    "Grok-4 (def)": "openrouter__x-ai_grok-4-fast_free__definition_based",
    "GPT-5 (rules)": "openrouter__openai_gpt-5__static_analysis_rules",
    "GPT-5 (def)": "openai__gpt-5-2025-08-07__definition_based",
}

# Ranking files (IntelliSA RQ2/inputs/ranking/by_repo_lines/<tech>/<file>.csv) per method
RANKING_FILES: Dict[str, str] = {
    "GLITCH": "glitch",
    "SLAC/SLIC": "slac_slic",
    "Claude-4 (rules)": "anthropic__claude-sonnet-4-20250514__static_analysis_rules",
    "Claude-4 (def)": "anthropic__claude-sonnet-4-20250514__definition_based",
    "Grok-4 (rules)": "openrouter__x-ai_grok-4-fast__static_analysis_rules",
    "Grok-4 (def)": "openrouter__x-ai_grok-4-fast__definition_based",
    "GPT-5 (rules)": "openai__gpt-5-2025-08-07__static_analysis_rules",
    "GPT-5 (def)": "openai__gpt-5-2025-08-07__definition_based",
    "IntelliSA": "intellisa",
}


def hub_dir(cfg: Config) -> Path:
    return cfg.path("external") / "intellisa-hub"


def cli_dir(cfg: Config) -> Path:
    return cfg.path("external") / "intellisa-cli"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m resa.data.fetch repos` first.")
    return path


def load_events(path: Path, line_field: str = "LINE", cat_field: str = "CATEGORY", path_field: str = "PATH") -> Set[Event]:
    """Load (path, line, canonical smell) events exactly like IntelliSA's ``load_csv_events``."""
    events: Set[Event] = set()
    for row in read_csv(path):
        category = normalize_category(row.get(cat_field, ""))
        if not category:
            continue
        match = _LINE_RE.search(str(row.get(line_field, "")))
        if not match:
            continue
        line_no = int(match.group(0))
        file_path = (row.get(path_field) or "").strip()
        if not file_path:
            continue
        events.add((file_path, line_no, category))
    return events


@dataclass
class Oracle:
    tech: str
    truth: Set[Event]
    files: List[str]
    script_dir: Path
    _lines: Dict[str, List[str]] = field(default_factory=dict, repr=False)

    def lines(self, file_name: str) -> List[str]:
        if file_name not in self._lines:
            self._lines[file_name] = read_lines(self.script_dir / file_name)
        return self._lines[file_name]

    def has_script(self, file_name: str) -> bool:
        return (self.script_dir / file_name).exists()

    def nonblank_lines(self) -> List[Tuple[str, int]]:
        out = []
        for name in sorted(self.files):
            if not self.has_script(name):
                continue
            for idx, text in enumerate(self.lines(name), start=1):
                if text.strip():
                    out.append((name, idx))
        return out

    def truth_by_file(self) -> Dict[str, Set[Event]]:
        grouped: Dict[str, Set[Event]] = defaultdict(set)
        for event in self.truth:
            grouped[event[0]].add(event)
        return grouped


def oracle_labels_path(cfg: Config, tech: str) -> Path:
    return _require(hub_dir(cfg) / "replication" / "datasets" / "oracle" / f"oracle-dataset-{tech}.csv")


def oracle_file_list(cfg: Config, tech: str) -> List[str]:
    path = _require(hub_dir(cfg) / "replication" / "datasets" / "oracle" / tech / f"oracle-dataset-{tech}.csv")
    return sorted({row["PATH"].strip() for row in read_csv(path) if row.get("PATH", "").strip()})


def processed_oracle_dir(cfg: Config, tech: str) -> Path:
    return cfg.path("processed") / "oracle" / tech


def load_oracle(cfg: Config, tech: str) -> Oracle:
    truth = {e for e in load_events(oracle_labels_path(cfg, tech)) if e[2] in SMELLS}
    return Oracle(tech=tech, truth=truth, files=oracle_file_list(cfg, tech), script_dir=processed_oracle_dir(cfg, tech))


def expected_nonblank_loc(cfg: Config, tech: str) -> Optional[int]:
    """|U| used by IntelliSA's RQ2 = number of rows of its published GLITCH ranking for ``tech``."""
    path = hub_dir(cfg) / "replication" / "RQ2" / "inputs" / "ranking" / "by_repo_lines" / tech / "glitch.csv"
    if not path.exists():
        return None
    return len(read_csv(path))


# ----------------------------------------------------------------------------- materialise
def _flat_variants(rel_parts: Tuple[str, ...]) -> Iterable[str]:
    """Flat names under which IntelliSA/GLITCH refer to a nested repository file."""
    for start in range(len(rel_parts)):
        tail = rel_parts[start:]
        if len(tail) >= 1:
            yield "-".join(tail)


def build_name_index(roots: Iterable[Path], exts: Tuple[str, ...]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not path.name.lower().endswith(exts):
                continue
            index[path.name].append(path)
            rel = path.relative_to(root).parts
            for flat in _flat_variants(rel):
                if flat != path.name:
                    index[flat].append(path)
    return index


_PROMPT_LINE = re.compile(r"^Line (\d+): ?(.*)$", re.M)


def reconstruct_from_prompt(cfg: Config, tech: str, name: str) -> Optional[str]:
    """Rebuild an oracle script from IntelliSA's released zero-shot prompt, which embeds the script
    as ``Line N: <text>``. Verified to be line-for-line identical (after trailing blank lines) on all
    161 Ansible/Chef scripts that the hub also ships verbatim."""
    prompt_dir = hub_dir(cfg) / "replication" / "RQ1" / "data" / "prompts" / "prompts" / tech
    prefix = "prompt_static_analysis_rules_"
    for path in sorted(prompt_dir.glob(prefix + "*")):
        target = path.name[len(prefix):].rsplit("_", 2)[0]
        if target != name:
            continue
        text = path.read_text(encoding="utf-8")
        if "### RAW CODE INPUT" not in text:
            return None
        section = text.split("### RAW CODE INPUT", 1)[1].split("### OUTPUT FORMAT", 1)[0]
        lines = {int(m.group(1)): m.group(2) for m in _PROMPT_LINE.finditer(section)}
        if not lines:
            return None
        rebuilt = [lines.get(i, "") for i in range(1, max(lines) + 1)]
        while rebuilt and not rebuilt[-1].strip():
            rebuilt.pop()
        return "\n".join(rebuilt) + "\n"
    return None


def materialize_oracle(cfg: Config, allow_missing: bool = False) -> Dict[str, dict]:
    """Copy the 241 oracle scripts into ``data/processed/oracle/<tech>/`` and validate them.

    Source order: the IntelliSA hub (Ansible, Chef), the GLITCH replication package, then a
    reconstruction from IntelliSA's released prompts (Puppet). The non-blank line total of each
    technology is checked against IntelliSA's |U| so RQ5 effort metrics use exactly the same code.
    """
    report: Dict[str, dict] = {}
    package_results = cfg.path("raw") / "glitch_package" / "RESULTS"
    for tech in TECHS:
        files = oracle_file_list(cfg, tech)
        out_dir = ensure_dir(processed_oracle_dir(cfg, tech))
        hub_scripts = hub_dir(cfg) / "replication" / "datasets" / "oracle" / tech
        missing: List[str] = []
        sources = {"hub": 0, "glitch_package": 0, "prompt_reconstruction": 0, "existing": 0}
        pkg_index: Optional[Dict[str, List[Path]]] = None
        for name in files:
            target = out_dir / name
            if target.exists():
                sources["existing"] += 1
                continue
            candidate = hub_scripts / name
            if candidate.exists():
                shutil.copyfile(candidate, target)
                sources["hub"] += 1
                continue
            if pkg_index is None:
                if (package_results / tech).exists():
                    oracle_roots = list((package_results / tech).glob("oracle*"))
                    pkg_index = build_name_index(oracle_roots or [package_results / tech], TECH_EXTS[tech])
                else:
                    pkg_index = {}
            hits = pkg_index.get(name, [])
            if hits:
                # prefer a copy that lives in an oracle folder
                hits = sorted(hits, key=lambda p: (0 if "oracle" in str(p).lower() else 1, len(str(p))))
                shutil.copyfile(hits[0], target)
                sources["glitch_package"] += 1
                continue
            rebuilt = reconstruct_from_prompt(cfg, tech, name)
            if rebuilt is not None:
                target.write_text(rebuilt, encoding="utf-8")
                sources["prompt_reconstruction"] += 1
            else:
                missing.append(name)
        oracle = load_oracle(cfg, tech)
        loc = len(oracle.nonblank_lines())
        expected = expected_nonblank_loc(cfg, tech)
        md5s = {md5_bytes((out_dir / n).read_bytes()) for n in files if (out_dir / n).exists()}
        report[tech] = {
            "files": len(files),
            "missing": missing,
            "sources": sources,
            "nonblank_loc": loc,
            "expected_nonblank_loc": expected,
            "loc_matches_intellisa": (expected == loc) if expected is not None else None,
            "smells": len(oracle.truth),
            "unique_md5": len(md5s),
        }
        status = "OK" if not missing and (expected is None or expected == loc) else "CHECK"
        LOG.info("[%s] oracle %s: %d files, %d missing, LOC %d (IntelliSA |U| %s)", tech, status, len(files), len(missing), loc, expected)
    write_json(cfg.path("processed") / "oracle" / "manifest.json", report)
    problems = {t: r for t, r in report.items() if r["missing"]}
    if problems and not allow_missing:
        raise RuntimeError(
            "Oracle scripts missing: "
            + ", ".join(f"{t}: {len(r['missing'])}" for t, r in problems.items())
            + ". Download/extract the GLITCH package (python -m resa.data.fetch glitch-package extract)."
        )
    return report


def oracle_manifest(cfg: Config) -> dict:
    path = cfg.path("processed") / "oracle" / "manifest.json"
    return read_json(path) if path.exists() else {}


# ----------------------------------------------------------------------------- stage 1 + baselines
def glitch_oracle_warnings(cfg: Config, tech: str) -> List[dict]:
    """GLITCH's published warnings on the oracle (= ReSA stage 1 on the test set; no rerun)."""
    path = _require(hub_dir(cfg) / "replication" / "RQ1" / "data" / "baselines" / "glitch" / f"GLITCH-{tech}-oracle.csv")
    seen: Dict[Tuple[str, int, str], dict] = {}
    for row in read_csv(path):
        smell = normalize_category(row.get("ERROR", ""))
        if smell not in SMELLS:
            continue
        try:
            line = int(row["LINE"])
        except (KeyError, ValueError):
            continue
        key = (row["PATH"].strip(), line, smell)
        if key not in seen:
            seen[key] = {"file": key[0], "line": line, "smell": smell, "raw": [row.get("ERROR", "").strip()]}
        else:
            seen[key]["raw"].append(row.get("ERROR", "").strip())
    return sorted(seen.values(), key=lambda r: (r["file"], r["line"], r["smell"]))


def published_detections(cfg: Config) -> Dict[str, Dict[str, Set[Event]]]:
    """Detection sets of every baseline in IntelliSA Table 2, loaded from the released files."""
    rq1 = _require(hub_dir(cfg) / "replication" / "RQ1" / "data")
    out: Dict[str, Dict[str, Set[Event]]] = {}
    out["GLITCH"] = {t: load_events(rq1 / "baselines" / "glitch" / f"GLITCH-{t}-oracle.csv", cat_field="ERROR") for t in TECHS}
    slac = {"ansible": "SLAC-ansible-oracle.csv", "chef": "SLAC-chef-oracle.csv", "puppet": "SLIC-puppet-oracle.csv"}
    out["SLAC/SLIC"] = {t: load_events(rq1 / "baselines" / "SLAC_SLIC" / f, cat_field="ERROR") for t, f in slac.items()}
    for label, dirname in LLM_DETECTION_DIRS.items():
        out[label] = {t: load_events(rq1 / "detections" / dirname / f"{t}.csv") for t in TECHS}
    out["IntelliSA"] = {t: load_events(rq1 / "intellisa" / f"testset-{t}.csv") for t in TECHS}
    return out


def published_ranking_path(cfg: Config, tech: str, method: str) -> Path:
    return hub_dir(cfg) / "replication" / "RQ2" / "inputs" / "ranking" / "by_repo_lines" / tech / f"{RANKING_FILES[method]}.csv"
