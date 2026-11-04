# ReSA Project Proposal: Verify and Recover

**Working title:** Verify and Recover: Context-Aware Knowledge Distillation for Precise and Complete Security Smell Detection in Infrastructure as Code at Cluster Scale

**Target journal:** Cluster Computing (Springer, hybrid). **Planned submission:** 15 November 2026
**Authors:** Junaid Hussain Muzamal, Aatira Anum, Mirza Mubasher Baig (FAST School of Computing, FAST-NUCES Lahore)
**Code:** this repository (`resa/`). **Status:** framework complete and validated on CPU; full GPU runs not yet done.

---

## 1. One-paragraph summary

Rule-based analysers such as GLITCH find most security smells in Ansible, Chef and Puppet scripts, but they raise many false alarms and miss smells hidden in shell commands, connection strings and identifiers. The current best hybrid, IntelliSA (arXiv 2601.14595), filters GLITCH warnings with a small model distilled from a proprietary LLM (Claude-4). It reaches Macro-F1 0.831, but it can only *remove* warnings, it looks only at the flagged line, it verifies 4 of the 9 smells and it depends on a commercial API. **ReSA** keeps the rule engine's coverage and adds two small distilled models:

1. a **context-aware verifier** that removes false alarms for all 9 smells, using a window of code around each warning; and
2. a **recovery module** that proposes candidates GLITCH never flagged and keeps only those the verifier confirms.

Both models learn from labels produced by a **committee of open-weight LLM teachers**, so no paid API and no manual labelling are needed. The pipeline splits CPU work from GPU work and is benchmarked from 1 to many workers and nodes, which is the Cluster Computing angle.

---

## 2. Problem and gap

| Gap | IntelliSA (state of the art) | ReSA |
|---|---|---|
| G1 Missed smells | Can only filter GLITCH warnings; a smell GLITCH never flags is lost | Relaxed candidate scan plus recovery model plus verifier confirmation |
| G2 Context | Sees the flagged line only | ±k lines or the enclosing block (k = 5 by default) |
| G3 Coverage | Verifies 4 of 9 smells | Verifies all 9 smells |
| G4 Teacher | Proprietary Claude-4 labels | Committee of 3 open-weight LLMs (soft vote), with agreement statistics |
| G5 Scale | No throughput or cost study | CPU/GPU pipeline, multi-node sharding, throughput, latency and cost |

Evidence for G1: on the standard oracle GLITCH misses 33 smells, 28 of them in Chef (21 are admin-by-default smells inside shell commands). The relaxed candidate scan already proposes 29 of these 33.

---

## 3. Research questions

| RQ | Question | Script | Main output |
|---|---|---|---|
| RQ1 | Is ReSA more accurate than rule-based tools, zero-shot LLMs and IntelliSA? | `resa/experiments/rq1.py` | Table 2 layout (P/R/F1 per tech, Macro-F1), mean ± std over 5 seeds, paired script-level bootstrap |
| RQ2 | What does the recovery module add? | `rq2.py` | Candidate coverage, recovered TP/FP per smell, with vs without recovery, threshold sweep figure |
| RQ3 | Which design choices matter? | `rq3.py` | A context window (0/3/5/10/block); B label source (committee vs single teacher vs Claude-4); C rebalancing; D backbone (CodeT5p-220M/770M, CodeBERT, UniXcoder); E 9 vs 4 smells; F teacher quality on the oracle |
| RQ4 | Does it scale on a cluster? | `rq4.py`, `resa/scaling.py`, `scripts/slurm_scaling.sh` | Throughput, speed-up, per-script latency, 1 to N CPU workers and nodes |
| RQ5 | Is it cost-effective? | `rq5.py` | Effort@60%Recall, F1@1%LOC (IntelliSA Table 3 protocol), USD per 1,000 scripts vs LLM APIs |

---

## 4. Method (what the code does)

```
                OFFLINE (training)                                   ONLINE (inference, resa/pipeline.py)
GLITCH corpus (196,755 scripts)                                 script
   │  leakage filter (oracle MD5/name, oracle decisions)           │
   ├─ GLITCH warnings ──► verifier pool                            ├─ Stage 1: GLITCH rules ─► warnings W(s)
   └─ relaxed candidates (not flagged) ──► recovery pool           ├─ context window ±k ─► verifier f_v ─► keep if ≥ τ_v
          │                                                        └─ relaxed candidates G(s)\W(s)
   Teacher committee (Qwen2.5-Coder-14B, Llama-3.1-8B,                    ─► recovery f_r ≥ τ_r AND verifier f_v ≥ τ_v
   DeepSeek-Coder-V2-Lite): YES/NO prompt, P(YES) averaged          Detections D(s) = verified W ∪ confirmed recovered
          │
   Students (CodeT5p-220M), smell-weighted loss, 5 seeds
   Thresholds τ_v, τ_r chosen on VALIDATION only, then frozen
```

Formally, with c_k(l) the context of ±k lines around line l:

D(s) = {(l,σ) ∈ W(s) : f_v(c_k(l),σ) ≥ τ_v} ∪ {(l,σ) ∈ G(s)\W(s) : f_r(c_k(l),σ) ≥ τ_r ∧ f_v(c_k(l),σ) ≥ τ_v}

Committee label: ŷ(x) = 1[ mean over teachers of P_t(YES | x) ≥ 0.5 ]; Fleiss' kappa is reported.

---

## 5. Data (all public, pinned)

| Item | Source | Notes |
|---|---|---|
| Oracle: 241 scripts, 213 smells (Ansible, Chef, Puppet) | IntelliSA hub, commit `457eb3fd` | The 80 Puppet scripts are rebuilt from IntelliSA's released prompts (validated 161/161 on Ansible/Chef; LOC matches 3,964) |
| Baseline detections: GLITCH, SLAC/SLIC, 6 zero-shot LLM runs, IntelliSA | IntelliSA hub | **Not rerun.** Recomputed from released files; `check` verifies Table 2 within 0.001 and all 27 Table 3 values |
| Training corpus: 196,755 scripts | GLITCH replication package (figshare 19726603 v2, SHA1-checked) | Extracted without Docker |
| IntelliSA Claude-4 labels (1,548 items) | IntelliSA hub | Used only for the RQ3-B comparison |

---

## 6. Evaluation protocol (kept identical to IntelliSA for a fair comparison)

- A hit is an exact **(file, line, smell)** match; P/R/F1 are micro-averaged per technology; Macro-F1 is their unweighted mean.
- **No test-set tuning:** thresholds come from the validation split; the oracle is used only for final evaluation (and RQ3-F teacher quality, as in IntelliSA Table 4).
- **Leakage control:** training items are dropped if their script matches an oracle script by MD5 or name, or if their (normalised line, smell) matches any oracle decision. Counts are written to `data/processed/pools/summary.json`.
- **Seeds:** 5 for RQ1/RQ2/RQ5 and 3 for ablations; mean ± std; paired bootstrap (2,000 resamples) vs IntelliSA and GLITCH.
- **Effort metrics:** one row per non-blank oracle line, alerts ranked by score (construction reverse-engineered and checked against IntelliSA Table 3).

---

## 7. Success criteria

| Criterion | Target |
|---|---|
| Accuracy | Macro-F1 > 0.831 (IntelliSA), significant in the paired bootstrap |
| Recovery | Recovers a meaningful share of the 33 GLITCH-missed oracle smells (most expected in Chef) with few false positives |
| Independence | No proprietary API anywhere in the pipeline |
| Scale | Near-linear CPU speed-up; per-script latency suitable for CI |
| Cost | Lower USD per 1,000 scripts than every zero-shot LLM baseline |

**If Macro-F1 does not beat 0.831**, ReSA still contributes recovery of missed smells, 9-smell coverage, open-weight teachers and cluster-scale efficiency (RQ2 to RQ5).

---

## 8. Current status

**Done and verified (CPU):**
- Full codebase: fetch, prepare, teacher labelling (vLLM / OpenAI-compatible / HF backends), committee aggregation, student training, pipeline, all five RQ scripts, cluster scaling, SLURM scripts.
- `python -m resa.cli check` reproduces all 9 published baselines (within 0.001) and all 27 effort values exactly.
- Puppet oracle reconstruction validated; GLITCH run on rebuilt scripts matches 145 of 146 published warnings.
- Smoke test (tiny random model, mock teachers, stand-in corpus) runs end to end; unit tests pass.

**Not yet done:**
- Real teacher labelling and student training on a GPU (first real run will likely need environment fixes for vLLM/CUDA).
- All result tables and figures for RQ1 to RQ5.
- The manuscript.

**Known limitations to handle before submission** (also in README section 7):
1. Some candidate patterns were written after looking at examples GLITCH misses, including Chef oracle scripts. Either describe them as derived from GLITCH's rules plus the smell definitions and report candidate coverage on the corpus pool too, or freeze them and list this in Threats to Validity. **Do not tune them further on the oracle.**
2. The GLITCH package CSV format was not inspected in advance; `prepare --glitch-source auto` falls back to running GLITCH (needs Ruby for Chef) if fewer than 50% of paths resolve.
3. IntelliSA releases 1,548 Claude-labelled items, not the 2,070 it describes; RQ3-B uses what is released.
4. RQ4/RQ5 numbers depend on hardware; report GPU/CPU models from `results/rq4_shards/*.env.json`, and set the API and GPU prices in `configs/default.yaml` to the values on the run date.

---

## 9. Compute plan

| Step | Command | Approx. time (1× A100 40 GB, 16 cores) |
|---|---|---|
| Environment and baseline check | `python -m resa.cli check` | minutes |
| Fetch (1.7 GB package) and prepare | `fetch all`, `prepare` | 1 to 2 h |
| Teacher labelling (3 models, 2 stages) | `label --stage verify`, `label --stage recover` | 3 to 6 h |
| RQ1 (2 students × 5 seeds) | `rq1` | 1 to 2 h |
| RQ2 | `rq2` | < 1 h |
| RQ3 ablations | `rq3 --parts A B C D E F` | 6 to 10 h |
| RQ4 scaling | `rq4` (+ SLURM for multi-node) | 1 to 2 h |
| RQ5 effort and cost | `rq5` | < 1 h |

Total about 15 to 25 GPU-hours. Without a datacenter GPU: use the `openai_compat` backend with Ollama or a hosted open-weight endpoint for labelling, and train students on an 8 to 12 GB GPU. If short on time: 3 seeds instead of 5, and single-node scaling only.

---

## 10. Write-up outline (results to sections)

1. Introduction: incidents, gap G1 to G5, contributions
2. Background and related work: smells catalogue, GLITCH, SLAC/SLIC, War et al. 2025, IntelliSA, LLM detectors
3. ReSA approach: Section 4 of this file, plus the architecture figure
4. Experimental setup: Sections 5 and 6
5. Results: RQ1 to RQ5 (tables and figures from `results/`)
6. Discussion: when recovery helps, cost and deployment in CI/CD, implications
7. Threats to validity: candidate-pattern design, oracle size, Puppet reconstruction, teacher noise, hardware dependence
8. Conclusion and future work (leads into the next study: Terraform and Kubernetes)

---

## 11. Timeline to submission

| Dates (Nov 2026) | Work |
|---|---|
| 4 to 5 | Local setup, `check`, `fetch`, `prepare`; first GPU run of labelling |
| 6 to 7 | Finish labelling; train students; RQ1 |
| 8 to 10 | RQ2, RQ3 ablations, RQ4 scaling, RQ5 |
| 8 to 14 | Write the manuscript in the Springer Nature template; internal review by co-authors |
| 15 | Submit to Cluster Computing |
