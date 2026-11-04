#!/usr/bin/env bash
# Teacher labelling on a GPU node (one teacher per job keeps memory simple; caches are shared).
#
#   for t in qwen2.5-coder-14b llama-3.1-8b deepseek-coder-v2-lite; do
#     TEACHER=$t sbatch --gres=gpu:1 --time=08:00:00 scripts/slurm_label.sh
#   done
#   # when all finished:
#   python -m resa.cli aggregate --stage verify && python -m resa.cli aggregate --stage recover
#
#SBATCH --job-name=resa-label
#SBATCH --output=logs/label_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
for stage in verify recover; do
  python -m resa.teachers.committee label --stage "$stage" --teachers "${TEACHER:?set TEACHER}" ${CFG:-}
done
