"""Fine-tune a student (verifier or recovery) from committee pseudo-labels.

    python -m resa.student.train --stage verify  --seed 42
    python -m resa.student.train --stage recover --seed 42
    python -m resa.student.train --stage verify  --label-source claude --run-name verify_claude_s42
    python -m resa.student.train --stage verify  --set context.window=0 --run-name verify_ctx0_s42

Writes models/<run>/ with weights, tokenizer, thresholds.json (chosen on validation only) and
train_meta.json (data counts, timings, validation metrics).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from resa.config import Config, add_config_args, config_from_args
from resa.student.data import label_stats, load_stage_items, oversample, texts_and_labels
from resa.student.model import load_classifier, model_inputs, pick_device
from resa.utils import device_info, ensure_dir, get_logger, set_seed, write_json

LOG = get_logger("resa.student.train")


def default_run_name(stage: str, label_source: str, cfg: Config, seed: int) -> str:
    ctx = cfg["context"]
    model = cfg["student"]["model_name"].split("/")[-1]
    src = label_source.replace(":", "-")
    return f"{stage}__{model}__{src}__ctx{ctx['mode']}{ctx['window']}__{cfg['student']['rebalance']}__s{seed}"


def f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    pred = p >= thr
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or y.sum() == 0:
        return 0.5
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_at(y, p, t) for t in grid]
    best = max(scores)
    # among ties prefer the threshold closest to 0.5 (stability)
    candidates = [t for t, s in zip(grid, scores) if s >= best - 1e-9]
    return float(min(candidates, key=lambda t: abs(t - 0.5)))


def _predict(model, tokenizer, texts: List[str], device, max_length: int, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    out = np.zeros(len(texts), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**model_inputs(enc, model)).logits.float()
            out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return out


def train_student(
    cfg: Config,
    stage: str,
    seed: int,
    label_source: str = "committee",
    run_name: Optional[str] = None,
    smells: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    scfg = cfg["student"]
    ctx = cfg["context"]
    set_seed(seed)
    run_name = run_name or default_run_name(stage, label_source, cfg, seed)
    out_dir = ensure_dir(cfg.path("models") / run_name)
    if (out_dir / "thresholds.json").exists():
        LOG.info("%s already trained; skipping", run_name)
        return out_dir

    train_items, val_items = load_stage_items(cfg, stage, label_source, smells, sources)
    if not train_items:
        raise RuntimeError(f"No training items for stage={stage} source={label_source}")
    if scfg.get("rebalance") == "smell_weighted":
        train_items = oversample(train_items, list(scfg.get("rare_smells", [])), int(scfg.get("rare_oversample", 1)))
    x_train, y_train = texts_and_labels(train_items, ctx)
    x_val, y_val = texts_and_labels(val_items, ctx)

    device = pick_device()
    tokenizer, model = load_classifier(scfg["model_name"])
    model.to(device)
    max_len = int(scfg["max_length"])

    pos = max(1, sum(y_train))
    neg = max(1, len(y_train) - pos)
    if scfg.get("rebalance") in ("class_weighted", "smell_weighted"):
        weights = torch.tensor([len(y_train) / (2 * neg), len(y_train) / (2 * pos)], dtype=torch.float32, device=device)
    else:
        weights = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    g = torch.Generator()
    g.manual_seed(seed)
    indices = list(range(len(x_train)))
    loader = DataLoader(indices, batch_size=int(scfg["batch_size"]), shuffle=True, generator=g)
    accum = max(1, int(scfg.get("grad_accum", 1)))
    total_steps = math.ceil(len(loader) / accum) * int(scfg["epochs"])
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": float(scfg["weight_decay"])},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=float(scfg["lr"]))
    sched = get_scheduler(scfg.get("scheduler", "cosine"), optim, num_warmup_steps=min(int(scfg["warmup_steps"]), total_steps // 5 or 1), num_training_steps=max(1, total_steps))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported() and scfg.get("fp16", "auto") in ("auto", True, "bf16")
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast(device_type="cpu", enabled=False)

    best_f1, best_epoch, patience = -1.0, -1, int(scfg.get("early_stopping_patience", 2))
    history = []
    t0 = time.perf_counter()
    step = 0
    for epoch in range(int(scfg["epochs"])):
        model.train()
        running = 0.0
        for b, batch_idx in enumerate(loader):
            texts = [x_train[i] for i in batch_idx.tolist()]
            labels = torch.tensor([y_train[i] for i in batch_idx.tolist()], device=device)
            enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with autocast:
                logits = model(**model_inputs(enc, model)).logits
            loss = loss_fn(logits.float(), labels) / accum
            loss.backward()
            running += loss.item() * accum
            if (b + 1) % accum == 0 or (b + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
        p_val = _predict(model, tokenizer, x_val, device, max_len, 64) if x_val else np.zeros(0)
        yv = np.array(y_val)
        f1v = f1_at(yv, p_val, 0.5) if len(yv) else 0.0
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, len(loader)), "val_f1@0.5": f1v})
        LOG.info("[%s] epoch %d loss %.4f val F1@0.5 %.4f", run_name, epoch + 1, running / max(1, len(loader)), f1v)
        if f1v > best_f1 or not x_val:
            best_f1, best_epoch = f1v, epoch + 1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        elif epoch + 1 - best_epoch >= patience:
            LOG.info("Early stopping at epoch %d (best %d)", epoch + 1, best_epoch)
            break
    train_seconds = time.perf_counter() - t0

    # threshold selection on validation with the best checkpoint (never on the oracle)
    _, best_model = load_classifier(str(out_dir))
    best_model.to(device)
    p_val = _predict(best_model, tokenizer, x_val, device, max_len, 64) if x_val else np.zeros(0)
    yv = np.array(y_val)
    thresholds = {"global": choose_threshold(yv, p_val), "per_tech": {}}
    for tech in sorted({it["tech"] for it in val_items}):
        mask = np.array([it["tech"] == tech for it in val_items])
        if mask.sum() >= 20 and yv[mask].sum() >= 3:
            thresholds["per_tech"][tech] = choose_threshold(yv[mask], p_val[mask])
    write_json(out_dir / "thresholds.json", thresholds)
    write_json(
        out_dir / "train_meta.json",
        {
            "run_name": run_name,
            "stage": stage,
            "seed": seed,
            "label_source": label_source,
            "smells": smells,
            "sources": sources,
            "student": dict(scfg),
            "context": dict(ctx),
            "n_train": len(x_train),
            "n_val": len(x_val),
            "train_label_stats": label_stats(train_items),
            "val_label_stats": label_stats(val_items),
            "best_epoch": best_epoch,
            "val_f1_at_0.5": best_f1,
            "val_f1_at_threshold": f1_at(yv, p_val, thresholds["global"]) if len(yv) else None,
            "history": history,
            "train_seconds": train_seconds,
            "device": device_info(),
        },
    )
    LOG.info("Saved %s (val F1 %.3f @ %.2f, %.0fs)", out_dir, best_f1, thresholds["global"], train_seconds)
    return out_dir


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["verify", "recover"], required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label-source", default="committee", help="committee | claude | teacher:<name>")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smells", nargs="*", default=None)
    parser.add_argument("--sources", nargs="*", default=None, help="Restrict training items by source (glitch, intellisa, candidate)")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    train_student(cfg, args.stage, seed, args.label_source, args.run_name, args.smells, args.sources)


if __name__ == "__main__":
    main()
