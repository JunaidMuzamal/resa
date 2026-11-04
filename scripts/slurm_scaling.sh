#!/usr/bin/env bash
# Multi-node RQ4 scaling run. Each SLURM task processes a disjoint shard of the workload with the
# same configuration sequence; results are merged afterwards.
#
#   sbatch --nodes=4 --ntasks=4 --cpus-per-task=16 --gres=gpu:1 scripts/slurm_scaling.sh
#   python -m resa.scaling merge        # after all jobs finished (or: bash scripts/slurm_scaling.sh merge)
#
#SBATCH --job-name=resa-scaling
#SBATCH --time=04:00:00
#SBATCH --output=logs/scaling_%j_%t.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

if [[ "${1:-}" == "merge" ]]; then
  python -m resa.scaling merge ${CFG:-}
  exit 0
fi

VERIFIER=${VERIFIER:-$(ls -d models/verify__*__committee__ctxlines5__smell_weighted__s42 | head -1)}
RECOVERY=${RECOVERY:-$(ls -d models/recover__*__committee__ctxlines5__smell_weighted__s42 | head -1)}
CPUS=${SLURM_CPUS_PER_TASK:-16}

srun python -m resa.scaling run \
  --verifier "$VERIFIER" --recovery "$RECOVERY" \
  --cpu-workers "$CPUS" --gpu-workers 1 \
  --files ${FILES:-80000} --repeats ${REPEATS:-3} ${CFG:-}
