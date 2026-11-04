#!/usr/bin/env bash
# End-to-end smoke test on CPU (~10-20 min). Exercises every stage with a tiny model and mock
# teachers; the numbers it prints are meaningless. Use it to validate an installation.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="--config configs/smoke.yaml"

python -m resa.cli fetch repos $CFG
python scripts/make_tiny_model.py --out smoke/tiny-model
python scripts/make_smoke_corpus.py --out smoke/data/raw/glitch_package
python -m resa.cli prepare $CFG --glitch-source run --workers 2
python -m resa.cli label --stage verify $CFG
python -m resa.cli label --stage recover $CFG
python -m resa.cli rq1 $CFG
python -m resa.cli rq2 $CFG
python -m resa.cli rq3 --parts A B C E F $CFG
python -m resa.cli rq4 --source oracle --files 40 $CFG
python -m resa.cli rq5 $CFG
python -m resa.cli check $CFG
echo "Smoke test finished: see smoke/results/"
