"""RQ4 - scalability on a single node (multi-node: scripts/slurm_scaling.sh + `python -m resa.scaling merge`).

    python -m resa.experiments.rq4 [--source corpus|oracle] [--files 20000]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from resa.config import add_config_args, config_from_args
from resa.experiments.common import ensure_model, save_table
from resa.scaling import run_latency, run_scaling, summarize
from resa.utils import ensure_dir, get_logger, read_json

LOG = get_logger("resa.rq4")


def plot(cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(cfg.path("results") / "rq4_scaling.csv")
    df = df[df.num_shards == df.num_shards.min()]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for g, sub in df.groupby("gpu_workers"):
        sub = sub.sort_values("cpu_workers")
        ax.plot(sub["cpu_workers"], sub["speedup"], marker="o", label=f"{g} GPU worker(s)")
    xs = sorted(df["cpu_workers"].unique())
    ax.plot(xs, [x / xs[0] for x in xs], linestyle=":", color="grey", label="ideal")
    ax.set_xlabel("CPU workers")
    ax.set_ylabel("throughput speed-up")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = ensure_dir(cfg.path("results") / "figures")
    fig.savefig(out / "rq4_speedup.pdf")
    fig.savefig(out / "rq4_speedup.png", dpi=200)
    plt.close(fig)


def run(cfg, source: str, files: Optional[int], skip_latency: bool = False) -> dict:
    seed = int(cfg["student"]["seeds"][0])
    verifier = ensure_model(cfg, "verify", seed)
    recovery = ensure_model(cfg, "recover", seed)
    run_scaling(cfg, verifier, recovery, list(cfg["scaling"]["cpu_workers"]), list(cfg["scaling"]["gpu_workers"]),
                files or int(cfg["scaling"]["sample_files"]), int(cfg["scaling"]["repeats"]), source)
    summary = summarize(cfg)
    table = summary["table"]
    save_table(cfg, "rq4_scaling_table", table,
               ["num_shards", "cpu_workers", "gpu_workers", "files", "wall_mean", "files_per_s", "loc_per_s", "speedup", "parallel_efficiency"],
               ["Nodes", "CPU w.", "GPU w.", "Scripts", "Wall (s)", "Scripts/s", "LOC/s", "Speed-up", "Efficiency"],
               "Throughput and scaling of the full ReSA pipeline.")
    plot(cfg)
    if not skip_latency:
        run_latency(cfg, verifier, recovery, 500, source)
    return summary


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["corpus", "oracle"], default="corpus")
    parser.add_argument("--files", type=int, default=None)
    parser.add_argument("--skip-latency", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    run(config_from_args(args), args.source, args.files, args.skip_latency)


if __name__ == "__main__":
    main()
