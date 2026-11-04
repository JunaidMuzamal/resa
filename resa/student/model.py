"""Student classifier loading and batched scoring (P(true smell))."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from resa.utils import get_logger, read_json

LOG = get_logger("resa.student.model")


def pick_device(prefer: Optional[str] = None):
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_name(name_or_path: str) -> str:
    """Local directories may be given relative to the project root; hub ids pass through."""
    from resa.config import PROJECT_ROOT

    local = PROJECT_ROOT / name_or_path
    return str(local) if not Path(name_or_path).is_absolute() and local.exists() else name_or_path


def model_inputs(enc, model) -> dict:
    """Keep only tokenizer outputs the model's forward() accepts (e.g. T5 rejects token_type_ids)."""
    import inspect

    accepted = getattr(model, "_resa_accepted", None)
    if accepted is None:
        accepted = set(inspect.signature(model.forward).parameters)
        model._resa_accepted = accepted
    return {k: v for k, v in enc.items() if k in accepted}


def load_classifier(name_or_path: str, num_labels: int = 2):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name_or_path = resolve_model_name(name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_path, num_labels=num_labels, id2label={0: "FP", 1: "TP"}, label2id={"FP": 0, "TP": 1}
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


class Scorer:
    """Wraps a fine-tuned student directory (``models/<run>/``) for inference."""

    def __init__(self, model_dir: Path | str, device: Optional[str] = None, max_length: int = 512, dtype: str = "auto"):
        import torch

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.tokenizer, self.model = load_classifier(str(self.model_dir))
        self.device = pick_device(device)
        self.model.to(self.device)
        # T5 overflows in fp16, so only bf16 is used for reduced-precision inference
        if dtype == "auto" and self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.model.to(torch.bfloat16)
        self.model.eval()
        meta = self.model_dir / "train_meta.json"
        self.meta = read_json(meta) if meta.exists() else {}
        # inference must truncate exactly like training did
        self.max_length = int((self.meta.get("student") or {}).get("max_length", max_length))
        thr = self.model_dir / "thresholds.json"
        self.thresholds = read_json(thr) if thr.exists() else {"global": 0.5}
        self.seconds = 0.0
        self.n_scored = 0

    def threshold(self, tech: Optional[str] = None) -> float:
        if tech and tech in self.thresholds.get("per_tech", {}):
            return float(self.thresholds["per_tech"][tech])
        return float(self.thresholds.get("global", 0.5))

    def score(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        if len(texts) == 0:
            return np.zeros(0, dtype=np.float32)
        order = np.argsort([len(t) for t in texts])
        probs = np.zeros(len(texts), dtype=np.float32)
        start_time = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                idx = order[start : start + batch_size]
                enc = self.tokenizer(
                    [texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**model_inputs(enc, self.model)).logits.float()
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs[idx] = p
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - start_time
        self.n_scored += len(texts)
        return probs
