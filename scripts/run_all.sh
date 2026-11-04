#!/usr/bin/env bash
# Full ReSA experiment pipeline (full results). Every step is resumable: re-running skips work
# that is already cached (downloads, labels, trained models, runs).
#
#   bash scripts/run_all.sh                 # everything, default config
#   CFG="--config configs/my.yaml" bash scripts/run_all.sh
#
# Approximate cost on 1x A100 40GB + 16 CPU cores (see README for details):
#   fetch+prepare ~1-2 h | teacher labelling ~3-6 h | 2 students x 5 seeds ~1-2 h
#   RQ3 ablations ~6-10 h | RQ4 scaling ~1-2 h
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-}

step() { echo; echo "=== $* ==="; }

step "0. environment"
python -m resa.cli check $CFG || true

step "1. fetch IntelliSA repos (pinned) + GLITCH replication package (1.7 GB) and extract it"
python -m resa.cli fetch all $CFG

step "2. build oracle, stage-1 warnings and training pools (no manual labelling)"
python -m resa.cli prepare $CFG

step "3. teacher committee pseudo-labels (open-weight LLMs; resumable)"
python -m resa.cli label --stage verify $CFG
python -m resa.cli label --stage recover $CFG

step "4. RQ1 accuracy (trains verifier + recovery for every seed)"
python -m resa.cli rq1 $CFG

step "5. RQ2 recovery analysis"
python -m resa.cli rq2 $CFG

step "6. RQ3 design choices + teacher quality"
python -m resa.cli rq3 --parts A B C D E F $CFG

step "7. RQ4 scalability (single node; multi-node: scripts/slurm_scaling.sh)"
python -m resa.cli rq4 $CFG

step "8. RQ5 effort + cost"
python -m resa.cli rq5 $CFG

echo; echo "All results in results/ (CSV + LaTeX tables, figures/)."
