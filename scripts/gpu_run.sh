#!/usr/bin/env bash
# Run one ReSA step on the GPU PC, save its log, and push the log + small results back to GitHub
# so they can be read on the laptop (where Claude Code runs).
#
#   bash scripts/gpu_run.sh setup          # first time: venv + dependencies + tests
#   bash scripts/gpu_run.sh check
#   bash scripts/gpu_run.sh fetch
#   bash scripts/gpu_run.sh prepare
#   bash scripts/gpu_run.sh label-verify
#   bash scripts/gpu_run.sh label-recover
#   bash scripts/gpu_run.sh rq1 | rq2 | rq3 | rq4 | rq5
#   bash scripts/gpu_run.sh all            # every step in order
#
# Extra arguments are passed to resa, e.g.  bash scripts/gpu_run.sh rq1 --set student.epochs=3
# Set NO_PUSH=1 to skip the git commit/push.
set -uo pipefail
cd "$(dirname "$0")/.."

STEP=${1:-}
shift || true
EXTRA=("$@")
mkdir -p logs
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/${STAMP}_${STEP}.log"

if [[ -d .venv ]]; then source .venv/bin/activate; fi

run() {  # run a command, tee to log, return its exit code
  echo "+ $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

env_report() {
  {
    echo "=== $(date -Is)  host=$(hostname)  git=$(git rev-parse --short HEAD 2>/dev/null)"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    python -V 2>&1
    nproc 2>/dev/null | sed 's/^/cpu cores: /'
  } | tee -a "$LOG"
}

step() {
  case "$1" in
    setup)
      [[ -d .venv ]] || python3 -m venv .venv
      source .venv/bin/activate
      run python -m pip install -U pip "setuptools<81" wheel || return 1
      if command -v nvidia-smi >/dev/null && ! python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        run python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 || return 1
      fi
      run python -m pip install -r requirements.txt || return 1
      run python -m pip install -e . || return 1
      run python -m pip install vllm || echo "vllm install failed; use --set teachers.backend=hf or openai_compat" | tee -a "$LOG"
      run python -c "import torch;print('cuda available:',torch.cuda.is_available())"
      run python -m pytest -q tests ;;
    check)         run python -m resa.cli check "${EXTRA[@]}" ;;
    fetch)         run python -m resa.cli fetch all "${EXTRA[@]}" ;;
    prepare)       run python -m resa.cli prepare "${EXTRA[@]}" ;;
    label-verify)  run python -m resa.cli label --stage verify "${EXTRA[@]}" ;;
    label-recover) run python -m resa.cli label --stage recover "${EXTRA[@]}" ;;
    rq1|rq2|rq4|rq5) run python -m resa.cli "$1" "${EXTRA[@]}" ;;
    rq3)           run python -m resa.cli rq3 --parts A B C D E F "${EXTRA[@]}" ;;
    smoke)         run bash scripts/smoke_test.sh ;;
    *) echo "unknown step: $1"; return 2 ;;
  esac
}

push_back() {
  [[ "${NO_PUSH:-0}" == "1" ]] && return 0
  # small, shareable outputs only (results tables/figures, label agreement, pool summary, logs)
  git add -f logs/ 2>/dev/null
  [[ -d results ]] && git add -f results/ 2>/dev/null
  for f in data/processed/pools/summary.json data/labels/verify/agreement.json data/labels/recover/agreement.json; do
    [[ -f "$f" ]] && git add -f "$f"
  done
  if git diff --cached --quiet; then echo "nothing new to push"; return 0; fi
  git commit -q -m "GPU run: $STEP ($STAMP) on $(hostname)" && git pull --rebase -q && git push -q \
    && echo "pushed results and log for '$STEP'" || echo "PUSH FAILED: run 'git pull --rebase && git push' manually"
}

[[ -z "$STEP" ]] && { sed -n '2,15p' "$0"; exit 2; }
git pull --rebase -q || echo "warning: git pull failed (continuing with local code)"
env_report
if [[ "$STEP" == "all" ]]; then
  STATUS=0
  for s in check fetch prepare label-verify label-recover rq1 rq2 rq3 rq4 rq5; do
    STEP=$s; LOG="logs/$(date +%Y%m%d-%H%M%S)_${s}.log"
    step "$s"; STATUS=$?
    push_back
    [[ $STATUS -ne 0 ]] && { echo "step $s FAILED (exit $STATUS); see $LOG"; exit $STATUS; }
  done
  exit 0
fi
step "$STEP"; STATUS=$?
echo "=== exit code: $STATUS" | tee -a "$LOG"
push_back
exit $STATUS
