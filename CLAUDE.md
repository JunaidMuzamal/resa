# CLAUDE.md: working on ReSA

Read `PROPOSAL.md` first (goal, RQs, success criteria, status), then `README.md` (how to run, protocol details, known limitations).

## How this project is run (read this first)
Claude Code runs on the **laptop only**. The laptop has no GPU, so **do not run the heavy steps here**
(fetch, prepare, label, rq1 to rq5). The workflow is in `WORKFLOW.md`:
1. Edit code here, run `make test` (CPU), commit and **push** to GitHub.
2. The user runs `bash scripts/gpu_run.sh <step>` on the GPU PC. That script pulls, runs, and pushes back
   `logs/<timestamp>_<step>.log` and small outputs in `results/`.
3. When the user says a step finished, run `git pull`, read the newest log in `logs/` and the files in
   `results/`, report what passed or failed, fix code if needed, push, and tell the user which step to rerun.
Allowed locally: unit tests, `bash scripts/smoke_test.sh` (CPU, small), reading logs and results, writing code.

## Goal of this project
Run the full experiment pipeline for ReSA and produce the result tables and figures in `results/`. The code is complete and validated on CPU; the remaining work is running it for real on a GPU, fixing environment issues, and collecting results.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
# GPU: install the CUDA build of torch first, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121
make install          # pip install -r requirements.txt && pip install -e .
pip install vllm      # optional, fastest teacher labelling (Linux + CUDA)
make test             # unit tests (incl. baseline reproduction)
bash scripts/smoke_test.sh   # CPU end-to-end check, meaningless numbers
```
The package must be installed with `pip install -e .` because worker processes re-import `resa`.

## Run order on the GPU PC (each step is resumable and cached; normally via `scripts/gpu_run.sh`)
```bash
python -m resa.cli check                 # must report 9/9 baselines and 27/27 effort values reproduced
python -m resa.cli fetch all
python -m resa.cli prepare
python -m resa.cli label --stage verify
python -m resa.cli label --stage recover
python -m resa.cli rq1
python -m resa.cli rq2
python -m resa.cli rq3 --parts A B C D E F
python -m resa.cli rq4
python -m resa.cli rq5
```
Or `bash scripts/run_all.sh`. Override config keys with `--set key=value` (e.g. `--set teachers.backend=openai_compat`).

## Rules that protect the validity of the results (do not break)
- Never tune thresholds, candidate patterns or hyperparameters on the oracle (test set). Thresholds come from the validation split only.
- Do not rerun the published baselines; they are recomputed from IntelliSA's released files and checked by `resa.cli check`.
- Do not change the matching protocol in `resa/eval/accuracy.py` (exact file, line, smell) or the effort construction in `resa/eval/effort.py`; they reproduce IntelliSA's numbers.
- Do not remove leakage control in `resa/data/prepare.py`; report the counts from `data/processed/pools/summary.json`.
- `rule_mock` is a test-only teacher backend; never use it for results.
- Do not edit `resa/candidates.py` patterns based on oracle scripts (see PROPOSAL.md, known limitation 1).
- Keep pinned commits and the SHA1 check in `configs/default.yaml`.

## Where things are
- `resa/data/` fetch, oracle and Puppet reconstruction, corpus index, pool preparation
- `resa/teachers/` prompts, backends (vLLM, OpenAI-compatible, HF), committee and agreement
- `resa/student/` training (validation-only thresholds) and batched scorer
- `resa/pipeline.py` inference; `resa/scaling.py` CPU/GPU pipeline and multi-node shards
- `resa/eval/` accuracy (+ bootstrap), effort, cost; `resa/experiments/rq1..rq5.py`
- `scripts/slurm_*.sh` cluster jobs; outputs go to `data/`, `models/`, `runs/`, `results/`

## When something fails
- vLLM/CUDA errors: try `--set teachers.backend=hf`, or `openai_compat` with Ollama (`ollama pull qwen2.5-coder:14b llama3.1:8b deepseek-coder-v2:16b`).
- GPU memory: set `teachers.vllm.quantization=awq` for the 14B teacher, or lower `student.batch_size`.
- GLITCH package CSV paths not resolving: `prepare --glitch-source auto` runs GLITCH itself (needs Ruby for Chef).
- Fix bugs in the code, add a unit test in `tests/` for the fix, and rerun `make test` before continuing.

## Reporting
After each RQ, summarise the key numbers from `results/` and compare with the success criteria in PROPOSAL.md section 7. Record the GPU/CPU model and the run date (needed for RQ4/RQ5 and the cost prices).
