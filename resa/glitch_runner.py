"""Run GLITCH (the version vendored in the IntelliSA CLI) on scripts, in parallel worker processes.

Used (a) to obtain stage-1 warnings for corpus scripts when the replication package's precomputed
CSV cannot be used, and (b) for the end-to-end cluster-scale timing runs of RQ4. The oracle stage 1
always uses GLITCH's *published* oracle output, so test-set numbers never depend on this module.
"""
from __future__ import annotations

import logging
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from resa.smells import SMELLS, normalize_category

_STATE: Dict[str, object] = {}


def glitch_root(cli_dir: Path) -> Path:
    root = Path(cli_dir) / "packages" / "glitch_core"
    if not (root / "glitch").exists():
        raise FileNotFoundError(f"Vendored GLITCH not found under {root}; run `python -m resa.data.fetch repos`")
    return root


def _init(cli_dir: str, tech: str) -> None:
    root = str(glitch_root(Path(cli_dir)))
    if root not in sys.path:
        sys.path.insert(0, root)
    logging.getLogger("glitch").setLevel(logging.ERROR)
    from importlib.resources import files

    from glitch.analysis.security import SecurityVisitor
    from glitch.parsers.cmof import AnsibleParser, ChefParser, PuppetParser
    from glitch.repr.inter import UnitBlockType
    from glitch.tech import Tech

    parser = {"ansible": AnsibleParser, "chef": ChefParser, "puppet": PuppetParser}[tech]()
    visitor = SecurityVisitor(Tech(tech))
    visitor.config(str(files("glitch").joinpath("configs/default.ini")))
    _STATE.update(parser=parser, visitor=visitor, unit=UnitBlockType.unknown, tech=tech)
    warm_up(parser, UnitBlockType.unknown, tech)


_WARMUP = {"puppet": "class warmup {\n  $x = 'y'\n}\n", "ansible": "- hosts: all\n  tasks: []\n", "chef": "package 'x'\n"}


def warm_up(parser, unit, tech: str) -> None:
    """Parse a trivial script once so ply/puppetparser write their LALR tables before workers fork;
    concurrent first-time table generation makes GLITCH silently fail to parse some scripts."""
    import tempfile

    suffix = {"puppet": ".pp", "ansible": ".yml", "chef": ".rb"}[tech]
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as handle:
        handle.write(_WARMUP[tech])
        name = handle.name
    try:
        parser.parse(name, unit, False)
    except Exception:  # warm-up is best effort
        pass
    finally:
        Path(name).unlink(missing_ok=True)


def scan_file(path: str) -> Tuple[str, List[Tuple[int, str, str]], float, Optional[str]]:
    """Return (path, [(line, canonical smell, glitch code)], seconds, error)."""
    start = time.perf_counter()
    try:
        inter = _STATE["parser"].parse(path, _STATE["unit"], False)  # type: ignore[union-attr]
        if inter is None:
            return path, [], time.perf_counter() - start, "parse_returned_none"
        errors = _STATE["visitor"].check(inter)  # type: ignore[union-attr]
    except Exception as exc:  # GLITCH raises on some malformed scripts
        return path, [], time.perf_counter() - start, type(exc).__name__
    found = set()
    for err in errors:
        smell = normalize_category(getattr(err, "code", ""))
        if smell in SMELLS and getattr(err, "line", 0):
            found.add((int(err.line) if err.line > 0 else 1, smell, err.code))
    return path, sorted(found), time.perf_counter() - start, None


def scan_files(cli_dir: Path, tech: str, paths: Iterable[str], workers: int = 1, chunksize: int = 8):
    """Yield scan_file results; ``workers`` processes each holding their own parser/visitor."""
    paths = list(paths)
    # fail fast in the parent: a crashing pool initializer would otherwise respawn workers forever
    _init(str(cli_dir), tech)
    if workers <= 1:
        for p in paths:
            yield scan_file(p)
        return
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(str(cli_dir), tech)) as pool:
        for result in pool.imap_unordered(scan_file, paths, chunksize=chunksize):
            yield result
