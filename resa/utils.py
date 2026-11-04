"""Small shared helpers: logging, JSONL/CSV I/O, hashing, seeding, file reading."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("RESA_LOGLEVEL", "INFO"), format=LOG_FORMAT, stream=sys.stderr)
    return logging.getLogger(name)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(Path(path).parent)
    count = 0
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def append_jsonl(path: Path | str, row: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_csv(path: Path | str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(Path(path).parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path | str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path | str, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def file_digest(path: Path | str, algo: str = "sha1", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_text(path: Path | str) -> str:
    """Read a script exactly as IntelliSA counts it (UTF-8 with replacement)."""
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8", errors="replace")


def read_lines(path: Path | str) -> List[str]:
    return read_text(path).splitlines()


def normalize_snippet(text: str) -> str:
    """Whitespace/case-insensitive key used for leakage de-duplication."""
    return " ".join(text.lower().split())


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


@contextmanager
def timed(label: str, sink: Optional[Dict[str, float]] = None, logger: Optional[logging.Logger] = None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if sink is not None:
        sink[label] = sink.get(label, 0.0) + elapsed
    if logger is not None:
        logger.info("%s took %.2fs", label, elapsed)


def device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:  # pragma: no cover
        info["torch"] = None
    return info
