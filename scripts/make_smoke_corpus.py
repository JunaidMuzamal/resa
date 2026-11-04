"""Build a small stand-in for the GLITCH corpus for the smoke test only.

Uses the non-oracle Puppet scripts that ship in the IntelliSA hub and the GLITCH test fixtures, laid
out like the replication package (RESULTS/<tech>/repos/...). The real experiments use the full
196,755-script package (python -m resa.data.fetch all).
    python scripts/make_smoke_corpus.py --out smoke/data/raw/glitch_package
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

EXTS = {"ansible": (".yml", ".yaml"), "chef": (".rb",), "puppet": (".pp",)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="smoke/data/raw/glitch_package")
    parser.add_argument("--hub", default="external/intellisa-hub")
    parser.add_argument("--cli", default="external/intellisa-cli")
    parser.add_argument("--max-per-tech", type=int, default=150)
    args = parser.parse_args()
    out = Path(args.out) / "RESULTS"
    if out.exists():
        print(f"{out} exists")
        return
    oracle_names = set()
    for tech in EXTS:
        listing = Path(args.hub) / "replication/datasets/oracle" / tech / f"oracle-dataset-{tech}.csv"
        if listing.exists():
            with open(listing, newline="", encoding="utf-8") as handle:
                oracle_names |= {r["PATH"] for r in csv.DictReader(handle)}
    sources = {
        "puppet": [Path(args.hub) / "replication/datasets/oracle/puppet"],
        "ansible": [Path(args.cli) / "packages/glitch_core/glitch/tests"],
        "chef": [Path(args.cli) / "packages/glitch_core/glitch/tests"],
    }
    for tech, roots in sources.items():
        count = 0
        for root in roots:
            for path in sorted(root.rglob("*")):
                if count >= args.max_per_tech:
                    break
                if not path.is_file() or not path.name.endswith(EXTS[tech]) or path.name in oracle_names:
                    continue
                if tech != "puppet" and f"/{tech}/" not in path.as_posix():
                    continue
                dest = out / tech / "repos" / "smoke" / f"repo{count // 25}" / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
                count += 1
        print(f"{tech}: {count} scripts")
    (Path(args.out) / ".extracted.json").write_text('{"smoke": true}')


if __name__ == "__main__":
    main()
