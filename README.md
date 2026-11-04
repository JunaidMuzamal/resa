# ReSA — Recover-and-Verify Static Analysis for IaC security smells

End-to-end implementation of *"Verify and Recover: A Context-Aware, Lightweight Hybrid Framework for
Scalable Security Smell Detection in Infrastructure-as-Code"* (target: Cluster Computing).

ReSA runs a rule-based analyzer (GLITCH) and then applies two small neural models:

1. A **context-aware verifier** removes false alarms. It sees ±5 lines of code around each warning, where IntelliSA saw only the flagged line, and it covers all 9 smell types (IntelliSA covers 4).
2. A **recovery module** finds smells that GLITCH never flagged. Relaxed patterns propose candidate lines, the recovery model scores them, and the verifier confirms them.

Both models are trained on labels from a **committee of open-weight LLM teachers**, so no paid API is used and no human labelling is needed. The pipeline separates CPU work from GPU work and is benchmarked at cluster scale.

---

## 1. What is reused and what is run

| Item | Source | Rerun? |
|---|---|---|
| Oracle (241 scripts, 213 smells) | IntelliSA hub (pinned commit) | no |
| GLITCH, SLAC/SLIC, 6 zero-shot LLM runs, IntelliSA | IntelliSA's released detection files | **no** – recomputed from their files |
| Table 2 / Table 3 numbers of those baselines | checked automatically against IntelliSA's published numbers | — |
| Stage 1 on the test set | GLITCH's *published* oracle output | no |
| ReSA students, teacher labels, all ablations, scaling | this code | yes |

`python -m resa.cli check` confirms that all 9 published baselines reproduce IntelliSA's Table 2 (within 0.001) and that all 27 (tech, method) pairs reproduce its Table 3, before you run anything else.

**The 80 Puppet oracle scripts are not shipped by the IntelliSA hub.** ReSA rebuilds them from the zero-shot prompts IntelliSA released, which embed every script line by line. This method was validated on the 161 Ansible/Chef scripts the hub *does* ship (161/161 identical). The non-blank LOC matches IntelliSA's |U| exactly (3,964), and our GLITCH run on the rebuilt scripts matches 145 of the 146 published warnings.

## 1b. Laptop + GPU PC workflow

Claude Code works on the laptop; the GPU PC only pulls and runs `bash scripts/gpu_run.sh <step>`, which pushes the log and results back. See `WORKFLOW.md`.

## 2. Requirements

* Python 3.10–3.13, git, ~40 GB free disk
* **GPU recommended**: one 24–80 GB NVIDIA GPU for teacher labelling and training. Students alone also train on 8–12 GB.
* Ruby, only if you run GLITCH yourself (Chef parsing). The RQ4 scaling run needs it.
* Optional: `vllm` for fast local teacher inference (Linux + CUDA)

```bash
python -m venv .venv && source .venv/bin/activate
# GPU: install the CUDA build of torch first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
make install            # = pip install -r requirements.txt && pip install -e .
pip install vllm        # optional, recommended for teacher labelling
```

The package must be installed (`pip install -e .`). Worker processes re-import `resa`, so it has to be importable from anywhere.

## 3. Run everything

```bash
bash scripts/run_all.sh          # resumable; re-run after any interruption
```

Or step by step:

| Step | Command | Output |
|---|---|---|
| 0 | `python -m resa.cli check` | environment + baseline reproduction report |
| 1 | `python -m resa.cli fetch all` | `external/` (pinned repos), `data/raw/` (1.7 GB GLITCH package, SHA1-verified, extracted **without Docker**) |
| 2 | `python -m resa.cli prepare` | `data/processed/`: oracle, stage-1 warnings, `pools/verify.jsonl`, `pools/recover.jsonl` |
| 3 | `python -m resa.cli label --stage verify` and `--stage recover` | `data/labels/<stage>/` per-teacher caches, `committee.jsonl`, `agreement.json` |
| 4 | `python -m resa.cli rq1` | `results/rq1_*` (Table 2 layout, bootstrap, per-smell) |
| 5 | `python -m resa.cli rq2` | `results/rq2_*` (coverage, with/without recovery, threshold sweep + figure) |
| 6 | `python -m resa.cli rq3 --parts A B C D E F` | `results/rq3*_*` (context, labels, rebalancing, backbone, coverage, teacher quality) |
| 7 | `python -m resa.cli rq4` | `results/rq4_*` (throughput, speed-up, latency + figure) |
| 8 | `python -m resa.cli rq5` | `results/rq5_*` (Effort@60%Recall, F1@1%LOC, cost) |

Every table is written as CSV **and** LaTeX (`booktabs`), ready to `\input{}` into a LaTeX write-up.

### Teacher backends

Set `teachers.backend` in `configs/default.yaml` (or `--set teachers.backend=...`):

| Backend | When to use |
|---|---|
| `vllm` (default) | local GPU, fastest |
| `openai_compat` | an OpenAI-compatible server: **Ollama** (`ollama pull qwen2.5-coder:14b llama3.1:8b deepseek-coder-v2:16b`), `vllm serve`, LM Studio, or a hosted open-weight endpoint (Together, DeepInfra, …; put the key in `TEACHER_API_KEY` and set `teachers.openai_compat.base_url`) |
| `hf` | plain transformers, no extra dependency |
| `rule_mock` | **tests only** – never for results |

The committee is defined in `teachers.models`. P(YES) is read from token log-probabilities when the backend exposes them; otherwise the YES/NO answer is used.

### On a cluster (SLURM)

```bash
for t in qwen2.5-coder-14b llama-3.1-8b deepseek-coder-v2-lite; do
  TEACHER=$t sbatch --gres=gpu:1 --time=08:00:00 scripts/slurm_label.sh
done
python -m resa.cli aggregate --stage verify && python -m resa.cli aggregate --stage recover

sbatch --nodes=4 --ntasks=4 --cpus-per-task=16 --gres=gpu:1 scripts/slurm_scaling.sh   # RQ4 multi-node
python -m resa.scaling merge
```

### Quick validation (CPU, ~10 min, meaningless numbers)

```bash
bash scripts/smoke_test.sh     # tiny random model + mock teachers + stand-in corpus
make test                      # unit tests, incl. Table 2/3 reproduction and Puppet reconstruction
```

## 4. Protocol details that keep the comparison with IntelliSA fair

* **Matching:** a hit is an exact (file, line, smell) match. P/R/F1 are micro-averaged per technology, and Macro-F1 is the unweighted mean (IntelliSA's `evaluate.py`; `resa/eval/accuracy.py` reimplements it and is checked against their numbers).
* **No test-set tuning:** thresholds are chosen on the **validation** split and frozen (`models/<run>/thresholds.json`). The oracle is used only for final evaluation and for the teacher-quality analysis (RQ3-F, the same use IntelliSA's Table 4 makes of it).
* **Leakage control:** training items are removed if their script's MD5 or name matches an oracle script, or if (normalised line, smell) matches any oracle decision (labels, GLITCH warnings and recovery candidates). IntelliSA's oracle-derived `test.jsonl` is never used. Counts are in `data/processed/pools/summary.json`.
* **Seeds:** 5 seeds for RQ1/RQ2/RQ5 (`student.seeds`) and 3 for ablations (`experiments.ablation_seeds`), reported as mean ± std. The paired script-level bootstrap (2,000 resamples) compares ReSA with IntelliSA and GLITCH.
* **Effort metrics:** one row per non-blank oracle line (|U| = 3,760 / 4,090 / 3,964). Kept alerts are ranked by score, at most one alert per line, followed by the other lines. This construction was reverse-engineered from IntelliSA's ranking files, and our metric code reproduces their Table 3 exactly. Ties among equal scores are broken by a SHA-1 of the alert, which is not their hash. Rebuilding IntelliSA's own ranking with our tie-break gives the same Effort@60%Recall but F1@1%LOC differs by ≤ 0.015 (Chef), so **compare ReSA with the published Table 3 values, as the RQ5 table does**.

## 5. Configuration

All settings live in `configs/default.yaml`. Override any key with `--set key=value`, or merge your own file with `--config my.yaml`. The key settings are:

* `context.window` / `context.mode`: verifier input (0 = IntelliSA's line-only input)
* `pools.*`: sample sizes; `pools.max_corpus_files_per_tech` gives a quicker partial run
* `student.*`: backbone (default `Salesforce/codet5p-220m`), epochs, learning rate, rebalancing, seeds
* `pipeline.verify_smells`: `all` or `intellisa4`
* `cost.*`: **set the API and GPU prices to the values on the date you run, and report that date**

Cached artefacts are reused by name. After changing inference-only settings, run `make clean-runs`. Models are keyed by stage/backbone/labels/context/rebalance/seed, so they do not collide.

## 6. Layout

```
resa/
  data/fetch.py        pinned clones, figshare download with resume + SHA1, Docker-free layer extraction
  data/oracle.py       oracle, Puppet reconstruction, published baselines
  data/corpus.py       196k-script corpus index + GLITCH corpus warnings (package CSV or own run)
  data/prepare.py      verifier / recovery pools, leakage control, splits
  candidates.py        relaxed candidate generator (recovery stage)
  context.py, items.py context windows and student inputs
  glitch_runner.py     vendored GLITCH (IntelliSA CLI copy) in worker processes
  teachers/            prompts, backends (vLLM / OpenAI-compatible / HF), committee + agreement
  student/             data assembly, training (threshold on validation), batched scorer
  pipeline.py          ReSA inference -> runs/<name>/{decisions,detections}
  scaling.py           CPU/GPU pipeline, throughput, latency, multi-node shards
  eval/                accuracy (+bootstrap), effort (Table 3 protocol), cost
  experiments/rq1..5   one script per research question
```

## 7. Known limitations and things to check before submission

1. **Candidate patterns** (`resa/candidates.py`) build on GLITCH's keyword lists. Some patterns (e.g. `-u root` in commands, `root_user` identifiers) were written after looking at examples of what GLITCH misses, including Chef oracle scripts. Before submitting, either (a) describe the patterns as derived from GLITCH's rules plus the smell definitions and report candidate coverage on the *corpus* recovery pool too, or (b) freeze them now and add this point to Threats to Validity. Do not tune them further on the oracle.
2. The real pipeline has **not yet been run end to end with real LLM teachers on a GPU**; only the CPU smoke test and the unit tests have been run. Expect to fix environment details (vLLM/CUDA versions) on the first real run.
3. The format of the GLITCH package's precomputed corpus CSV could not be inspected in advance. `prepare --glitch-source auto` checks how many CSV paths resolve to corpus files. If fewer than 50% resolve, it runs GLITCH itself on the sampled scripts, which needs Ruby for Chef.
4. The IntelliSA hub releases 1,548 Claude-labelled train/val items, not the 2,070 IntelliSA describes. RQ3-B uses what is released.
5. Measured RQ4/RQ5 numbers depend on your hardware. Report the GPU/CPU model from `results/rq4_shards/*.env.json`.
