"""LLM teacher backends. Each returns P(YES) per prompt (from token log-probs when available).

* ``vllm``          - local GPU inference (fastest; Linux + CUDA)
* ``openai_compat`` - any OpenAI-compatible server: Ollama, vLLM serve, LM Studio, TGI, or a hosted
                      open-weight endpoint (Together, Fireworks, DeepInfra, OpenRouter ...)
* ``hf``            - plain transformers on GPU/CPU (no extra dependency)
* ``rule_mock``     - deterministic keyword heuristic for unit tests only; never use for results
"""
from __future__ import annotations

import gc
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from resa.teachers.prompts import SYSTEM
from resa.utils import get_logger

LOG = get_logger("resa.teachers")

YES_STRINGS = {"yes", "y", "true"}
NO_STRINGS = {"no", "n", "false"}


def _parse_text(text: str) -> Optional[float]:
    token = re.sub(r"[^a-z]", " ", (text or "").strip().lower()).split()
    if not token:
        return None
    if token[0] in YES_STRINGS:
        return 1.0
    if token[0] in NO_STRINGS:
        return 0.0
    return None


def _p_yes_from_top(top: Dict[str, float]) -> Optional[float]:
    """``top`` maps decoded token -> logprob for the first generated position."""
    yes = sum(math.exp(lp) for tok, lp in top.items() if tok.strip().lower() in YES_STRINGS)
    no = sum(math.exp(lp) for tok, lp in top.items() if tok.strip().lower() in NO_STRINGS)
    if yes + no <= 0:
        return None
    return yes / (yes + no)


class TeacherBackend:
    name: str = "base"

    def score(self, prompts: List[str]) -> List[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------- vLLM
class VLLMBackend(TeacherBackend):
    def __init__(self, name: str, hf_id: str, cfg: dict, max_new_tokens: int = 4, temperature: float = 0.0):
        from vllm import LLM, SamplingParams  # imported lazily: optional dependency

        self.name = name
        vcfg = cfg.get("vllm", {})
        kwargs = dict(
            model=hf_id,
            tensor_parallel_size=int(vcfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(vcfg.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(vcfg.get("max_model_len", 4096)),
            trust_remote_code=True,
            seed=0,
        )
        if vcfg.get("quantization"):
            kwargs["quantization"] = vcfg["quantization"]
        self.llm = LLM(**kwargs)
        self.params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, logprobs=20, seed=0)

    def score(self, prompts: List[str]) -> List[dict]:
        messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(messages, self.params, use_tqdm=True)
        results = []
        for out in outputs:
            comp = out.outputs[0]
            p = None
            if comp.logprobs:
                first = comp.logprobs[0]
                top = {lp.decoded_token: lp.logprob for lp in first.values() if lp.decoded_token is not None}
                p = _p_yes_from_top(top)
            if p is None:
                p = _parse_text(comp.text)
            results.append({"p_yes": p, "text": comp.text.strip()[:32]})
        return results

    def close(self) -> None:
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
        del self.llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatBackend(TeacherBackend):
    def __init__(self, name: str, served_name: str, cfg: dict, max_new_tokens: int = 4, temperature: float = 0.0):
        import requests

        self.name = name
        self.model = served_name
        ocfg = cfg.get("openai_compat", {})
        self.url = ocfg.get("base_url", "http://localhost:11434/v1").rstrip("/") + "/chat/completions"
        self.key = os.environ.get(ocfg.get("api_key_env", "TEACHER_API_KEY"), "")
        self.timeout = float(ocfg.get("timeout", 120))
        self.retries = int(ocfg.get("max_retries", 5))
        self.concurrency = int(ocfg.get("concurrency", 8))
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.session = requests.Session()
        self.logprobs_supported = True

    def _one(self, prompt: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "seed": 0,
        }
        if self.logprobs_supported:
            body.update({"logprobs": True, "top_logprobs": 10})
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = self.session.post(self.url, json=body, headers=headers, timeout=self.timeout)
                if resp.status_code == 400 and "logprobs" in resp.text.lower() and self.logprobs_supported:
                    self.logprobs_supported = False
                    body.pop("logprobs", None)
                    body.pop("top_logprobs", None)
                    continue
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
                p = None
                content = ((choice.get("logprobs") or {}).get("content")) or []
                if content:
                    top = {t["token"]: t["logprob"] for t in content[0].get("top_logprobs", []) if "token" in t}
                    if not top and "token" in content[0]:
                        top = {content[0]["token"]: content[0].get("logprob", 0.0)}
                    p = _p_yes_from_top(top)
                if p is None:
                    p = _parse_text(text)
                return {"p_yes": p, "text": text.strip()[:32]}
            except Exception as exc:  # network / server errors -> retry with backoff
                last_exc = exc
                time.sleep(min(30, 2 ** attempt))
        LOG.error("Teacher %s failed on a prompt after %d retries: %s", self.name, self.retries, last_exc)
        return {"p_yes": None, "text": f"ERROR: {last_exc}"[:64]}

    def score(self, prompts: List[str]) -> List[dict]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, prompts))


# ---------------------------------------------------------------------------- transformers
class HFBackend(TeacherBackend):
    def __init__(self, name: str, hf_id: str, cfg: dict, batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True
        )
        self.model.eval()
        self.batch_size = batch_size
        vocab = self.tok.get_vocab()
        self.yes_ids = sorted({i for t, i in vocab.items() if t.replace("Ġ", "").replace("▁", "").strip().lower() == "yes"})
        self.no_ids = sorted({i for t, i in vocab.items() if t.replace("Ġ", "").replace("▁", "").strip().lower() == "no"})

    def score(self, prompts: List[str]) -> List[dict]:
        torch = self.torch
        results = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in chunk
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            for row in probs:
                yes = float(row[self.yes_ids].sum()) if self.yes_ids else 0.0
                no = float(row[self.no_ids].sum()) if self.no_ids else 0.0
                p = yes / (yes + no) if yes + no > 0 else None
                results.append({"p_yes": p, "text": "YES" if (p or 0) >= 0.5 else "NO"})
        return results

    def close(self) -> None:
        del self.model
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------- tests only
class RuleMockBackend(TeacherBackend):
    """Deterministic keyword heuristic used by the test-suite to exercise the pipeline offline."""

    def __init__(self, name: str, bias: float = 0.0):
        self.name = name
        self.bias = bias

    def score(self, prompts: List[str]) -> List[dict]:
        out = []
        for prompt in prompts:
            marked = next((l for l in prompt.splitlines() if l.startswith(">>")), prompt)
            low = marked.lower()
            smell = re.search(r"(?:Detection Rule|Security Smell): (.+)", prompt)
            smell = smell.group(1).strip() if smell else ""
            p = 0.2
            if "{{" in low or "#{" in low or "$" in low or "lookup(" in low:
                p = 0.1
            elif smell == "Use of HTTP without SSL/TLS" and "http://" in low:
                p = 0.8
            elif smell == "Hard-coded secret" and re.search(r"(password|secret|token)\w*\W+['\"]\w", low):
                p = 0.85
            elif smell == "Admin by default" and re.search(r"(-u\s*root|\broot\b|admin)", low):
                p = 0.75
            elif smell == "Suspicious comment" and re.search(r"(todo|fixme|hack)", low):
                p = 0.7
            elif smell in ("Unrestricted IP Address", "Empty password", "Missing Default in Case Statement"):
                p = 0.7
            p = min(1.0, max(0.0, p + self.bias))
            out.append({"p_yes": p, "text": "YES" if p >= 0.5 else "NO"})
        return out


def make_backend(model_cfg: dict, tcfg: dict) -> TeacherBackend:
    kind = tcfg.get("backend", "vllm")
    max_new = int(tcfg.get("max_new_tokens", 4))
    temp = float(tcfg.get("temperature", 0.0))
    if kind == "vllm":
        return VLLMBackend(model_cfg["name"], model_cfg["hf_id"], tcfg, max_new, temp)
    if kind == "openai_compat":
        return OpenAICompatBackend(model_cfg["name"], model_cfg.get("served_name", model_cfg["hf_id"]), tcfg, max_new, temp)
    if kind == "hf":
        return HFBackend(model_cfg["name"], model_cfg["hf_id"], tcfg)
    if kind == "rule_mock":
        return RuleMockBackend(model_cfg["name"], float(model_cfg.get("mock_bias", 0.0)))
    raise ValueError(f"Unknown teacher backend {kind!r}")
