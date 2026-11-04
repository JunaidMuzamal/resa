"""RQ4: cluster-scale throughput, speed-up and per-script latency of the full ReSA pipeline.

The pipeline is split by resource type:
  CPU stage  (process pool, N workers): GLITCH parse+rules, relaxed candidates, context windows
  GPU stage  (G workers, one per device): verifier + recovery student inference, batched
Multi-node runs shard the workload (--shard i --num-shards n); `merge` aggregates the shard reports.

    python -m resa.scaling run     --verifier models/<v> --recovery models/<r> --cpu-workers 1 2 4 8 16
    python -m resa.scaling latency --verifier models/<v> --recovery models/<r> --files 500
    python -m resa.scaling run ... --shard 0 --num-shards 4      # on each node (see scripts/slurm_scaling.sh)
    python -m resa.scaling merge
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from resa.candidates import generate_candidates
from resa.config import Config, add_config_args, config_from_args
from resa.context import extract_context, student_input
from resa.data.oracle import TECHS, cli_dir, load_oracle
from resa.utils import device_info, ensure_dir, get_logger, read_json, read_lines, write_csv, write_json

LOG = get_logger("resa.scaling")
_W: Dict[str, object] = {}


# ----------------------------------------------------------------------------- workload
def workload(cfg: Config, n_files: int, source: str, seed: int) -> List[Tuple[str, str]]:
    """[(tech, path)] - corpus scripts (default) or the oracle scripts repeated (offline fallback)."""
    rng = random.Random(seed)
    if source == "corpus":
        from resa.data.corpus import load_corpus_index

        pool = [(t, r["path"]) for t in TECHS for r in load_corpus_index(cfg, t)]
        if not pool:
            raise RuntimeError("Corpus index empty; run `python -m resa.data.prepare` or use --source oracle")
    else:
        pool = []
        for t in TECHS:
            o = load_oracle(cfg, t)
            pool += [(t, str(o.script_dir / f)) for f in o.files if o.has_script(f)]
    pool.sort()
    rng.shuffle(pool)
    if n_files <= len(pool):
        return pool[:n_files]
    reps = (n_files + len(pool) - 1) // len(pool)
    return (pool * reps)[:n_files]


# ----------------------------------------------------------------------------- CPU stage
def _cpu_init(cli: str, ctx: dict) -> None:
    from resa.glitch_runner import glitch_root

    root = str(glitch_root(Path(cli)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import logging
    from importlib.resources import files

    logging.getLogger("glitch").setLevel(logging.ERROR)
    from glitch.analysis.security import SecurityVisitor
    from glitch.parsers.cmof import AnsibleParser, ChefParser, PuppetParser
    from glitch.repr.inter import UnitBlockType
    from glitch.tech import Tech

    conf = str(files("glitch").joinpath("configs/default.ini"))
    parsers = {"ansible": AnsibleParser(), "chef": ChefParser(), "puppet": PuppetParser()}
    visitors = {}
    for t in TECHS:
        v = SecurityVisitor(Tech(t))
        v.config(conf)
        visitors[t] = v
    from resa.glitch_runner import warm_up

    for t, p in parsers.items():
        warm_up(p, UnitBlockType.unknown, t)
    _W.update(parsers=parsers, visitors=visitors, unit=UnitBlockType.unknown, ctx=ctx)


def cpu_stage(job: Tuple[str, str]) -> Dict:
    from resa.smells import SMELLS, normalize_category

    tech, path = job
    t0 = time.perf_counter()
    warnings = set()
    err = None
    try:
        inter = _W["parsers"][tech].parse(path, _W["unit"], False)  # type: ignore[index]
        if inter is not None:
            for e in _W["visitors"][tech].check(inter):  # type: ignore[index]
                smell = normalize_category(getattr(e, "code", ""))
                if smell in SMELLS:
                    warnings.add((max(1, int(e.line)), smell))
    except Exception as exc:  # GLITCH fails on some scripts; the pipeline still runs recovery
        err = type(exc).__name__
    t_glitch = time.perf_counter() - t0
    lines = read_lines(path)
    ctx = _W["ctx"]  # type: ignore[assignment]
    verify, recover = [], []
    for line, smell in sorted(warnings):
        if 1 <= line <= len(lines):
            verify.append(student_input(tech, smell, "verify", extract_context(lines, line, ctx["window"], ctx["mode"], ctx["max_chars"])))
    for cand in generate_candidates(tech, lines, warnings):
        recover.append(student_input(tech, cand.smell, "recover", extract_context(lines, cand.line, ctx["window"], ctx["mode"], ctx["max_chars"])))
    return {"tech": tech, "path": path, "loc": sum(1 for l in lines if l.strip()), "verify": verify, "recover": recover,
            "cpu_seconds": time.perf_counter() - t0, "glitch_seconds": t_glitch, "error": err}


def run_cpu_stage(cfg: Config, jobs: List[Tuple[str, str]], workers: int, ctx: dict) -> Tuple[List[Dict], float]:
    _cpu_init(str(cli_dir(cfg)), ctx)  # fail fast in the parent (a crashing pool initializer respawns forever)
    t0 = time.perf_counter()
    if workers <= 1:
        results = [cpu_stage(j) for j in jobs]
    else:
        with get_context("spawn").Pool(workers, initializer=_cpu_init, initargs=(str(cli_dir(cfg)), ctx)) as pool:
            results = list(pool.imap_unordered(cpu_stage, jobs, chunksize=4))
    return results, time.perf_counter() - t0


# ----------------------------------------------------------------------------- GPU stage
def _gpu_worker(args) -> Tuple[float, int]:
    device, verifier, recovery, vtexts, rtexts, batch = args
    from resa.student.model import Scorer

    v = Scorer(verifier, device=device)
    r = Scorer(recovery, device=device) if recovery else None
    t0 = time.perf_counter()
    pv = v.score(vtexts, batch)
    if r is not None and rtexts:
        pr = r.score(rtexts, batch)
        tau = r.threshold()
        passed = [t.replace("stage: recover", "stage: verify", 1) for t, p in zip(rtexts, pr) if p >= tau]
        v.score(passed, batch)
    return time.perf_counter() - t0, len(pv)


def run_gpu_stage(verifier: Path, recovery: Optional[Path], results: List[Dict], gpu_workers: int, batch: int) -> float:
    import torch

    vtexts = [t for r in results for t in r["verify"]]
    rtexts = [t for r in results for t in r["recover"]]
    n_dev = torch.cuda.device_count()
    if gpu_workers <= 1 or n_dev <= 1:
        device = "cuda:0" if n_dev else None
        seconds, _ = _gpu_worker((device, str(verifier), str(recovery) if recovery else None, vtexts, rtexts, batch))
        return seconds
    shards = min(gpu_workers, n_dev)
    jobs = [(f"cuda:{i}", str(verifier), str(recovery) if recovery else None, vtexts[i::shards], rtexts[i::shards], batch) for i in range(shards)]
    with get_context("spawn").Pool(shards) as pool:
        results = pool.map(_gpu_worker, jobs)
    return max(seconds for seconds, _ in results)  # devices run concurrently; model loading excluded


# ----------------------------------------------------------------------------- experiments
def _ctx_from_model(model_dir: Path, cfg: Config) -> dict:
    meta = model_dir / "train_meta.json"
    ctx = dict(read_json(meta)["context"]) if meta.exists() else dict(cfg["context"])
    return {"window": int(ctx.get("window", 5)), "mode": ctx.get("mode", "lines"), "max_chars": int(ctx.get("max_chars", 1800))}


def run_scaling(cfg: Config, verifier: Path, recovery: Optional[Path], cpu_workers: List[int], gpu_workers: List[int],
                n_files: int, repeats: int, source: str, shard: int = 0, num_shards: int = 1) -> Path:
    scfg = cfg["scaling"]
    jobs = workload(cfg, n_files, source, int(cfg["seed"]))[shard::num_shards]
    ctx = _ctx_from_model(Path(verifier), cfg)
    # warm-up (parser tables, Ruby, CUDA context, model weights in page cache) - not timed
    warm, _ = run_cpu_stage(cfg, jobs[: min(20, len(jobs))], 1, ctx)
    run_gpu_stage(Path(verifier), Path(recovery) if recovery else None, warm, 1, int(scfg["batch_size"]))
    rows = []
    for g in gpu_workers:
        for w in cpu_workers:
            for rep in range(repeats):
                results, cpu_wall = run_cpu_stage(cfg, jobs, w, ctx)
                gpu_wall = run_gpu_stage(Path(verifier), Path(recovery) if recovery else None, results, g, int(scfg["batch_size"]))
                wall = cpu_wall + gpu_wall  # stage times; worker start-up and model loading excluded
                loc = sum(r["loc"] for r in results)
                row = {
                    "shard": shard, "num_shards": num_shards, "cpu_workers": w, "gpu_workers": g, "repeat": rep,
                    "files": len(results), "loc": loc, "wall_s": wall, "cpu_stage_wall_s": cpu_wall, "gpu_stage_wall_s": gpu_wall,
                    "cpu_core_seconds": sum(r["cpu_seconds"] for r in results), "glitch_seconds": sum(r["glitch_seconds"] for r in results),
                    "verify_items": sum(len(r["verify"]) for r in results), "recover_items": sum(len(r["recover"]) for r in results),
                    "glitch_errors": sum(1 for r in results if r["error"]), "files_per_s": len(results) / wall, "loc_per_s": loc / wall,
                }
                rows.append(row)
                LOG.info("cpu=%d gpu=%d rep=%d: %.1f files/s (cpu %.1fs, gpu %.1fs)", w, g, rep, row["files_per_s"], cpu_wall, gpu_wall)
    out_dir = ensure_dir(cfg.path("results") / "rq4_shards")
    out = out_dir / f"shard_{shard}_of_{num_shards}.csv"
    write_csv(out, rows)
    write_json(out_dir / f"shard_{shard}_of_{num_shards}.env.json", {"device": device_info(), "source": source, "files": len(jobs)})
    if num_shards == 1:
        summarize(cfg)
    return out


def run_latency(cfg: Config, verifier: Path, recovery: Optional[Path], n_files: int, source: str) -> Dict:
    """Per-script end-to-end latency (CI/CD scenario: one commit, one script, warm models)."""
    from resa.student.model import Scorer

    jobs = workload(cfg, n_files, source, int(cfg["seed"]) + 7)
    ctx = _ctx_from_model(Path(verifier), cfg)
    _cpu_init(str(cli_dir(cfg)), ctx)
    v = Scorer(verifier)
    r = Scorer(recovery) if recovery else None
    lat = []
    for job in jobs:
        t0 = time.perf_counter()
        res = cpu_stage(job)
        v.score(res["verify"])
        if r is not None and res["recover"]:
            pr = r.score(res["recover"])
            v.score([t.replace("stage: recover", "stage: verify", 1) for t, p in zip(res["recover"], pr) if p >= r.threshold()])
        lat.append(time.perf_counter() - t0)
    arr = np.array(lat)
    out = {"files": len(lat), "device": str(v.device), "p50_s": float(np.percentile(arr, 50)), "p95_s": float(np.percentile(arr, 95)),
           "p99_s": float(np.percentile(arr, 99)), "mean_s": float(arr.mean()), "max_s": float(arr.max())}
    write_json(cfg.path("results") / f"rq4_latency_{out['device'].split(':')[0]}.json", out)
    LOG.info("Latency: %s", out)
    return out


def summarize(cfg: Config) -> Dict:
    import pandas as pd

    shard_dir = cfg.path("results") / "rq4_shards"
    frames = [pd.read_csv(p) for p in sorted(shard_dir.glob("shard_*.csv"))]
    if not frames:
        raise RuntimeError("No shard results found")
    df = pd.concat(frames, ignore_index=True)
    group = ["num_shards", "cpu_workers", "gpu_workers", "repeat"]
    agg = df.groupby(group).agg(files=("files", "sum"), loc=("loc", "sum"), wall_s=("wall_s", "max"),
                                cpu_core_seconds=("cpu_core_seconds", "sum"), gpu_stage_wall_s=("gpu_stage_wall_s", "max")).reset_index()
    agg["files_per_s"] = agg["files"] / agg["wall_s"]
    agg["loc_per_s"] = agg["loc"] / agg["wall_s"]
    table = agg.groupby(["num_shards", "cpu_workers", "gpu_workers"]).agg(
        files=("files", "first"), loc=("loc", "first"), wall_mean=("wall_s", "mean"), wall_std=("wall_s", "std"),
        files_per_s=("files_per_s", "mean"), loc_per_s=("loc_per_s", "mean"), cpu_core_seconds=("cpu_core_seconds", "mean"),
        gpu_seconds=("gpu_stage_wall_s", "mean")).reset_index()
    shard0, cpu0 = table.num_shards.min(), table.cpu_workers.min()
    base_rows = table[(table.num_shards == shard0) & (table.cpu_workers == cpu0)]
    base_tput = base_rows["files_per_s"].max()
    # speed-up in throughput (robust to shards processing equal-sized slices of the workload)
    table["speedup"] = table["files_per_s"] / base_tput
    table["parallel_efficiency"] = table["speedup"] / ((table["cpu_workers"] * table["num_shards"]) / (cpu0 * shard0))
    write_csv(cfg.path("results") / "rq4_scaling.csv", table.to_dict("records"))
    ref = table.sort_values("files_per_s", ascending=False).iloc[0].to_dict()
    summary = {
        "table": table.to_dict("records"),
        "reference_config": {
            **ref,
            "cpu_seconds_per_1k": ref["cpu_core_seconds"] / ref["files"] * 1000,
            "gpu_seconds_per_1k": ref["gpu_seconds"] / ref["files"] * 1000,
            "hours_for_196755_scripts": 196755 / ref["files_per_s"] / 3600,
        },
    }
    write_json(cfg.path("results") / "rq4_scaling_summary.json", summary)
    LOG.info("Best: %.1f files/s -> full 196,755-script corpus in %.2f h", ref["files_per_s"], summary["reference_config"]["hours_for_196755_scripts"])
    return summary


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["run", "latency", "merge"])
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--recovery", type=Path, default=None)
    parser.add_argument("--cpu-workers", type=int, nargs="*", default=None)
    parser.add_argument("--gpu-workers", type=int, nargs="*", default=None)
    parser.add_argument("--files", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--source", choices=["corpus", "oracle"], default="corpus")
    parser.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    scfg = cfg["scaling"]
    if args.action == "merge":
        summarize(cfg)
        return
    if args.verifier is None:
        parser.error("--verifier is required")
    if args.action == "latency":
        run_latency(cfg, args.verifier, args.recovery, args.files or 500, args.source)
        return
    run_scaling(cfg, args.verifier, args.recovery, args.cpu_workers or list(scfg["cpu_workers"]), args.gpu_workers or list(scfg["gpu_workers"]),
                args.files or int(scfg["sample_files"]), args.repeats or int(scfg["repeats"]), args.source, args.shard, args.num_shards)


if __name__ == "__main__":
    main()
