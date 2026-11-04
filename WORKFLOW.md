# Workflow: laptop (Claude Code) → GitHub → GPU PC

```
 Laptop                         GitHub (main repo)                 GPU PC (terminal, via AnyDesk)
 Claude Code edits code  ──push──►  code                  ──pull──►  bash scripts/gpu_run.sh <step>
 Claude Code reads results ◄─pull──  logs/ + results/     ◄─push──   (log + results pushed automatically)
```

- **Code is changed only on the laptop** (by you or Claude Code), committed and pushed.
- **The GPU PC never edits code.** It pulls, runs one step, and pushes back only the log and small results.
- Large files (dataset, labels, models, runs; about 40 GB) stay on the GPU PC and are never committed.

## One-time setup

**Laptop**
```bash
git clone https://github.com/<you>/resa.git
cd resa
claude            # Claude Code reads CLAUDE.md automatically
```

**GPU PC** (open a terminal through AnyDesk)
```bash
git clone https://github.com/<you>/resa.git
cd resa
gh auth login                       # or a GitHub personal access token, so it can push logs back
git config user.name "GPU PC"; git config user.email "<your email>"
tmux new -s resa                    # keeps the job alive if AnyDesk disconnects (reattach: tmux attach -t resa)
bash scripts/gpu_run.sh setup       # venv, CUDA torch, dependencies, vLLM, unit tests
```

## The loop for every step

1. **GPU PC:** run the next step. The script pulls the latest code first, then runs, logs and pushes back.
   ```bash
   bash scripts/gpu_run.sh check
   ```
2. **Laptop:** tell Claude Code, for example: *"I ran `check` on the GPU PC. Pull and read the latest log in logs/ and tell me if it passed."*
3. If it failed, Claude Code fixes the code on the laptop and pushes. Then rerun the same step on the GPU PC.
4. If it passed, move to the next step.

## Step order

| # | Command on GPU PC | What it does | Rough time (1× A100) |
|---|---|---|---|
| 0 | `bash scripts/gpu_run.sh setup` | environment and tests | 15 to 30 min |
| 1 | `bash scripts/gpu_run.sh check` | verifies the 9 baselines and 27 effort values reproduce | minutes |
| 2 | `bash scripts/gpu_run.sh fetch` | pinned IntelliSA repos and the 1.7 GB GLITCH package | 30 to 60 min |
| 3 | `bash scripts/gpu_run.sh prepare` | oracle, stage-1 warnings, training pools, leakage control | 30 to 60 min |
| 4 | `bash scripts/gpu_run.sh label-verify` | teacher committee labels for the verifier | 2 to 4 h |
| 5 | `bash scripts/gpu_run.sh label-recover` | teacher committee labels for recovery | 1 to 2 h |
| 6 | `bash scripts/gpu_run.sh rq1` | trains students (5 seeds) and evaluates accuracy | 1 to 2 h |
| 7 | `bash scripts/gpu_run.sh rq2` | recovery analysis | < 1 h |
| 8 | `bash scripts/gpu_run.sh rq3` | ablations A to F | 6 to 10 h |
| 9 | `bash scripts/gpu_run.sh rq4` | scalability | 1 to 2 h |
| 10 | `bash scripts/gpu_run.sh rq5` | effort and cost | < 1 h |

`bash scripts/gpu_run.sh all` runs steps 1 to 10 in order and stops at the first failure, pushing each log as it goes. Extra options pass through, for example `bash scripts/gpu_run.sh rq1 --set student.epochs=3`. `NO_PUSH=1` skips the push.

## What comes back to GitHub
- `logs/<timestamp>_<step>.log`: full output, GPU model, git commit used
- `results/`: CSV and LaTeX tables and figures for every RQ
- `data/processed/pools/summary.json` (leakage-control counts) and `data/labels/*/agreement.json` (teacher agreement)

## Rules
- Never edit code on the GPU PC. If you must, commit and push it immediately so the laptop stays in sync.
- Always let the GPU PC push its log after a run, even a failed one; Claude Code needs it to diagnose the problem.
