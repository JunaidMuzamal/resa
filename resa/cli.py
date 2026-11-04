"""Single entry point: ``resa <command> [args]`` (or ``python -m resa.cli <command>``).

Commands
  check                 environment + data readiness report, and reproduction of IntelliSA Tables 2/3
  fetch  ...            -> resa.data.fetch
  prepare ...           -> resa.data.prepare
  label  ...            -> resa.teachers.committee label
  aggregate ...         -> resa.teachers.committee aggregate
  oracle-eval ...       -> resa.teachers.committee oracle-eval
  train ...             -> resa.student.train
  run ...               -> resa.pipeline
  scaling ...           -> resa.scaling
  rq1 | rq2 | rq3 | rq4 | rq5 ...
"""
from __future__ import annotations

import argparse
import shutil
import sys
from typing import List, Optional


def _check(argv: List[str]) -> None:
    from resa.config import add_config_args, config_from_args
    from resa.data.corpus import corpus_available
    from resa.data.oracle import hub_dir, cli_dir, oracle_manifest
    from resa.utils import device_info, write_json

    parser = argparse.ArgumentParser(prog="resa check")
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args(argv))
    report = {"device": device_info(), "ruby": bool(shutil.which("ruby")), "docker": bool(shutil.which("docker"))}
    report["intellisa_hub"] = hub_dir(cfg).exists()
    report["intellisa_cli"] = cli_dir(cfg).exists()
    report["glitch_corpus_extracted"] = corpus_available(cfg)
    report["oracle_manifest"] = oracle_manifest(cfg) or "not prepared"
    for mod in ("vllm",):
        try:
            __import__(mod)
            report[mod] = True
        except Exception:
            report[mod] = False
    if report["intellisa_hub"]:
        from resa.eval.accuracy import check_published
        from resa.eval.effort import check_published_effort

        t2 = check_published(cfg)
        t3 = check_published_effort(cfg)
        report["table2_reproduced"] = all(r["ok"] for r in t2.values())
        report["table3_reproduced"] = all(t3.values()) if t3 else None
    for stage in ("verify", "recover"):
        pool = cfg.path("processed") / "pools" / f"{stage}.jsonl"
        lab = cfg.path("labels") / stage / "committee.jsonl"
        report[f"pool_{stage}"] = sum(1 for _ in open(pool)) if pool.exists() else 0
        report[f"labels_{stage}"] = sum(1 for _ in open(lab)) if lab.exists() else 0
    write_json(cfg.path("results") / "environment_check.json", report)
    width = max(len(k) for k in report)
    for key, value in report.items():
        if key == "oracle_manifest" and isinstance(value, dict):
            value = {t: f"{v['files']} files, LOC ok={v['loc_matches_intellisa']}, missing={len(v['missing'])}" for t, v in value.items()}
        print(f"{key:<{width}}  {value}")


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        _check(rest)
    elif cmd == "fetch":
        from resa.data.fetch import main as m

        m(rest)
    elif cmd == "prepare":
        from resa.data.prepare import main as m

        m(rest)
    elif cmd in ("label", "aggregate", "oracle-eval"):
        from resa.teachers.committee import main as m

        m([cmd] + rest)
    elif cmd == "train":
        from resa.student.train import main as m

        m(rest)
    elif cmd == "run":
        from resa.pipeline import main as m

        m(rest)
    elif cmd == "scaling":
        from resa.scaling import main as m

        m(rest)
    elif cmd in ("rq1", "rq2", "rq3", "rq4", "rq5"):
        module = __import__(f"resa.experiments.{cmd}", fromlist=["main"])
        module.main(rest)
    else:
        print(f"Unknown command {cmd!r}\n{__doc__}")
        sys.exit(2)


if __name__ == "__main__":
    main()
