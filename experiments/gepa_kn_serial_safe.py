#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle-only GEPA runner for a Kannada GGUF model.

Input:
- GEPA dataset JSONL created separately on Colab.
- GGUF model path or Hugging Face repo+filename.

Free-tier API protection:
- Reflection cache.
- Serial GEPA execution.
- max_metric_calls budget.
- Per-provider max_calls.
- Provider cooldown.
- Retry/backoff.
- Optional provider rotation through LiteLLM.

Install on Kaggle:
!pip install -U "datasets<4.0.0" huggingface_hub litellm gepa rapidfuzz tqdm pandas
!pip uninstall -y llama-cpp-python llama_cpp_python llama-cpp || true
!pip cache purge
!pip install -U pip setuptools wheel
!pip install --no-cache-dir --force-reinstall --prefer-binary llama-cpp-python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125

This serial-safe version is designed to avoid llama.cpp CUDA memory-pool crashes during GEPA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import threading

BASE_SYSTEM_PROMPT = """ನೀನು ಸಹಾಯಕ, ನಿಖರ ಮತ್ತು ಸ್ಪಷ್ಟವಾಗಿ ಉತ್ತರಿಸುವ ಕನ್ನಡ AI ಸಹಾಯಕ.
ಬಳಕೆದಾರರ ಪ್ರಶ್ನೆ ಅಥವಾ ಸೂಚನೆಗೆ ಕನ್ನಡದಲ್ಲೇ ಸೂಕ್ತ ಉತ್ತರ ನೀಡಿ."""

OBJECTIVE = """Optimize the Kannada system instruction for a Kannada fine-tuned GGUF model.

Improve:
1. Correct multiple-choice answering for options A through J.
2. Kannada-only behavior when the answer is explanatory.
3. Avoidance of English, Hindi, Tamil, Telugu, Malayalam, Gujarati, Bengali, Chinese, and mixed-script leakage.
4. Direct relevance.
5. Concise school-level Kannada answers.
6. Honest uncertainty.

Only optimize the system instruction. Do not change this fixed template:

<system instruction>

ಬಳಕೆದಾರ: <question>
ಸಹಾಯಕ:

Prefer a Kannada system instruction because this prompt will be used in LM Studio / llama.cpp style inference.
"""

BACKGROUND = """The task model is a Kannada fine-tuned Gemma-style GGUF model loaded locally with llama-cpp-python.

GEPA is optimizing the system prompt only. It is not training model weights.

The GEPA dataset was created separately in Colab. MILU test should not be used for GEPA optimization if it will be reported later.
"""

OPTION_LETTERS = "ABCDEFGHIJ"
OPTION_SET = set(OPTION_LETTERS)

EVAL_COUNTER = 0


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        x = json.dumps(x, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(x)).strip()


def is_kannada_char(ch: str) -> bool:
    return "\u0C80" <= ch <= "\u0CFF"


def kannada_ratio(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(is_kannada_char(c) for c in letters) / len(letters)


def normalize_option_label(value: Any, options: Optional[List[str]] = None) -> str:
    if value is None:
        return ""
    raw = clean_text(value)
    upper = raw.upper().strip()

    m = re.match(r"^([A-J])(?:[\.\)\-:\s]|$)", upper)
    if m:
        return m.group(1)
    if upper in OPTION_SET:
        return upper

    try:
        idx = int(float(raw))
        if 0 <= idx < len(OPTION_LETTERS):
            return OPTION_LETTERS[idx]
        if 1 <= idx <= len(OPTION_LETTERS):
            return OPTION_LETTERS[idx - 1]
    except Exception:
        pass

    if options:
        nr = re.sub(r"\s+", " ", raw).strip().lower()
        for i, opt in enumerate(options[:10]):
            no = re.sub(r"\s+", " ", clean_text(opt)).strip().lower()
            if nr == no or nr in no or no in nr:
                return OPTION_LETTERS[i]
    return ""


def extract_option(text: str) -> Optional[str]:
    t = str(text).strip().upper()
    m = re.match(r"^\s*(?:ಉತ್ತರ\s*[:：\-]?\s*)?([A-J])(?:[\.\)\-: ]|$)", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-J])\b", t)
    if m:
        return m.group(1)
    return None


def parse_options_from_row(row: Dict[str, Any]) -> List[str]:
    groups = [
        ["option1", "option2", "option3", "option4"],
        ["option_1", "option_2", "option_3", "option_4"],
        ["option_a", "option_b", "option_c", "option_d"],
        ["A", "B", "C", "D"],
    ]
    for group in groups:
        vals = [clean_text(row.get(k, "")) for k in group if k in row]
        if len(vals) >= 4 and all(vals[:4]):
            return vals[:4]
    return []


def format_mcq(question: str, options: List[str]) -> str:
    lines = [question.strip()]
    for i, opt in enumerate(options[:10]):
        lines.append(f"{OPTION_LETTERS[i]}. {clean_text(opt)}")
    return "\n".join(lines)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("id", f"row_{i:05d}")
            row.setdefault("task_type", "short_qa")
            row.setdefault("reference_answer", "")
            row.setdefault("expected_keywords", [])
            row.setdefault("constraints", "")
            row.setdefault("max_chars", 800)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_jsonl(rows: List[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    train = [r for r in rows if str(r.get("split", "")).lower() == "train"]
    val = [r for r in rows if str(r.get("split", "")).lower() in {"val", "valid", "validation"}]
    test = [r for r in rows if str(r.get("split", "")).lower() == "test"]
    if train and val:
        return train, val, test

    rows = rows[:]
    random.Random(42).shuffle(rows)
    n = len(rows)
    n_train = max(1, int(0.5 * n))
    n_val = max(1, int(0.4 * n))
    train = rows[:n_train]
    val = rows[n_train:n_train+n_val]
    test = rows[n_train+n_val:]
    return train, val, test


def resolve_hf_gguf(repo_id: str, filename: Optional[str], revision: Optional[str], token_env: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    token = os.environ.get(token_env) if token_env else None

    if not filename:
        print(f"[HF] Listing .gguf files in {repo_id}")
        files = list_repo_files(repo_id=repo_id, repo_type="model", revision=revision, token=token)
        ggufs = sorted([f for f in files if f.lower().endswith(".gguf")])
        if not ggufs:
            raise RuntimeError(f"No GGUF files found in {repo_id}")
        if len(ggufs) > 1:
            print("Multiple GGUF files found:")
            for g in ggufs:
                print(" -", g)
            raise RuntimeError("Pass --hf-filename explicitly.")
        filename = ggufs[0]
        print(f"[HF] Auto-selected {filename}")

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    print(f"[HF] Downloading {repo_id}/{filename}")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
        token=token,
        local_dir=local_dir,
    )


def render_prompt(system_prompt: str, user_message: str) -> str:
    return f"{system_prompt.strip()}\n\nಬಳಕೆದಾರ: {user_message.strip()}\nಸಹಾಯಕ:"


class LocalGGUFModel:
    def __init__(self, gguf_path: str, n_ctx: int, n_gpu_layers: int, n_threads: Optional[int], n_batch: int):
        from llama_cpp import Llama
        try:
            from llama_cpp import llama_cpp
            print(f"[Model] GPU offload supported: {llama_cpp.llama_supports_gpu_offload()}", flush=True)
        except Exception as e:
            print(f"[Model] Could not check GPU offload support: {e}", flush=True)

        self._lock = threading.Lock()
        print(
            f"[Model] Llama init: n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}, n_threads={n_threads or max(2, os.cpu_count() or 2)}, n_batch={n_batch}",
            flush=True,
        )
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads or max(2, os.cpu_count() or 2),
            n_batch=n_batch,
            verbose=False,
        )

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float = 0.9) -> str:
        # llama.cpp CUDA memory pools are not safe for concurrent calls on one model instance.
        # GEPA may evaluate internally in ways that overlap calls, so enforce strict serialization.
        with self._lock:
            out = self.llm(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.08,
                stop=["\nಬಳಕೆದಾರ:", "ಬಳಕೆದಾರ:", "</s>", "<eos>", "<end_of_turn>"],
                echo=False,
            )
        return (out["choices"][0]["text"] or "").strip()


BAD_SCRIPT_PATTERNS = {
    "devanagari_hindi_marathi": r"[\u0900-\u097F]",
    "bengali": r"[\u0980-\u09FF]",
    "gujarati": r"[\u0A80-\u0AFF]",
    "tamil": r"[\u0B80-\u0BFF]",
    "telugu": r"[\u0C00-\u0C7F]",
    "malayalam": r"[\u0D00-\u0D7F]",
    "cjk": r"[\u4E00-\u9FFF]",
}


def detect_bad_scripts(text: str) -> List[str]:
    bad = []
    for name, pat in BAD_SCRIPT_PATTERNS.items():
        if re.search(pat, text):
            bad.append(name)
    words = re.findall(r"\b[A-Za-z]{3,}\b", text)
    if words:
        bad.append("english_words:" + ",".join(words[:5]))
    return bad


def fuzzy_reference_score(output: str, reference: str) -> float:
    if not reference:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return float(fuzz.token_set_ratio(output, reference)) / 100.0
    except Exception:
        out_toks = set(re.findall(r"[\w\u0C80-\u0CFF]+", output.lower()))
        ref_toks = set(re.findall(r"[\w\u0C80-\u0CFF]+", reference.lower()))
        return len(out_toks & ref_toks) / max(1, len(ref_toks))


def keyword_score(output: str, keywords: Iterable[str]) -> float:
    kws = [clean_text(k) for k in keywords if clean_text(k)]
    if not kws:
        return 0.0
    return sum(1 for k in kws if k in output) / len(kws)


def evaluate_output(output: str, ex: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    task_type = str(ex.get("task_type", "short_qa")).lower()
    ref = clean_text(ex.get("reference_answer", ""))
    kws = ex.get("expected_keywords", []) or []
    max_chars = int(ex.get("max_chars", 800) or 800)

    kr = kannada_ratio(output)
    bad_scripts = detect_bad_scripts(output)
    no_bad = 1.0 if not bad_scripts else 0.0
    non_empty = 1.0 if output.strip() else 0.0
    length_score = 1.0 if 1 <= len(output) <= max_chars else 0.4

    if task_type == "mcq":
        pred = extract_option(output)
        gold = normalize_option_label(ex.get("gold_option") or ref)
        content = 1.0 if pred == gold and gold in OPTION_SET else 0.0
        fmt = 1.0 if pred is not None and output.strip().upper().startswith(pred) else 0.5
        score = 0.65 * content + 0.20 * fmt + 0.10 * no_bad + 0.05 * kr
        reason = f"pred_option={pred}, gold_option={gold}"
    else:
        ks = keyword_score(output, kws)
        rs = fuzzy_reference_score(output, ref)
        if kws and ref:
            content = 0.55 * ks + 0.45 * rs
            reason = f"keyword_score={ks:.3f}, reference_fuzzy={rs:.3f}"
        elif kws:
            content = ks
            reason = f"keyword_score={ks:.3f}"
        elif ref:
            content = rs
            reason = f"reference_fuzzy={rs:.3f}"
        else:
            content = 0.5
            reason = "neutral content score"
        fmt = non_empty
        score = 0.40 * content + 0.35 * kr + 0.15 * no_bad + 0.05 * length_score + 0.05 * non_empty

    score = max(0.0, min(1.0, score))
    failures = []
    if not output.strip():
        failures.append("empty_output")
    if task_type != "mcq" and kr < 0.90:
        failures.append(f"low_kannada_ratio={kr:.3f}")
    if bad_scripts:
        failures.append(f"bad_scripts={bad_scripts}")
    if len(output) > max_chars:
        failures.append(f"too_long={len(output)}>{max_chars}")

    side = {
        "score": round(score, 4),
        "task_type": task_type,
        "kannada_ratio": round(kr, 4),
        "bad_scripts": bad_scripts,
        "content_score": round(content, 4),
        "content_reason": reason,
        "format_score": round(fmt, 4),
        "length_score": round(length_score, 4),
        "failures": failures or ["none"],
        "output": output,
    }
    return score, side


@dataclass
class ProviderState:
    model: str
    api_key_env: str
    max_calls: int
    cooldown_sec: float
    calls: int = 0
    failures: int = 0
    next_allowed_ts: float = 0.0


class RotatingReflectionLM:
    def __init__(
        self,
        providers: List[Dict[str, Any]],
        cache_path: str,
        max_retries_per_provider: int = 2,
        default_temperature: float = 1.0,
        default_max_tokens: int = 4096,
    ):
        if not providers:
            raise ValueError("At least one reflection provider is required.")

        self.providers: List[ProviderState] = []
        for p in providers:
            self.providers.append(ProviderState(
                model=str(p["model"]),
                api_key_env=str(p.get("api_key_env", "")),
                max_calls=int(p.get("max_calls", 25)),
                cooldown_sec=float(p.get("cooldown_sec", 5.0)),
            ))
        self.max_retries_per_provider = max_retries_per_provider
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.ptr = 0
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, str] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _normalize_messages(self, messages: Any) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, list):
            out = []
            for m in messages:
                if isinstance(m, dict):
                    out.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
                else:
                    out.append({"role": "user", "content": str(m)})
            return out
        return [{"role": "user", "content": str(messages)}]

    def _cache_key(self, messages: Any) -> str:
        payload = json.dumps(self._normalize_messages(messages), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.cache_path)

    def _pick_provider(self) -> Optional[int]:
        n = len(self.providers)
        for step in range(n):
            idx = (self.ptr + step) % n
            p = self.providers[idx]
            if p.calls < p.max_calls and time.time() >= p.next_allowed_ts:
                if p.api_key_env and not os.environ.get(p.api_key_env):
                    continue
                self.ptr = (idx + 1) % n
                return idx
        return None

    def usage_summary(self) -> Dict[str, Any]:
        return {
            p.model: {
                "api_key_env": p.api_key_env,
                "calls": p.calls,
                "max_calls": p.max_calls,
                "failures": p.failures,
                "cooldown_sec": p.cooldown_sec,
            } for p in self.providers
        }

    def __call__(self, messages: Any, **kwargs: Any) -> str:
        from litellm import completion

        key = self._cache_key(messages)
        if key in self.cache:
            return self.cache[key]

        normalized = self._normalize_messages(messages)
        last_error = None
        attempts_budget = len(self.providers) * self.max_retries_per_provider

        for _ in range(attempts_budget):
            idx = self._pick_provider()
            if idx is None:
                future_times = [p.next_allowed_ts for p in self.providers if p.calls < p.max_calls]
                if future_times:
                    wait = max(1.0, min(max(future_times) - time.time(), 60.0))
                    print(f"[ReflectionRouter] All providers cooling down. Sleeping {wait:.1f}s.")
                    time.sleep(wait)
                    idx = self._pick_provider()
                if idx is None:
                    break

            p = self.providers[idx]
            api_key = os.environ.get(p.api_key_env) if p.api_key_env else None

            try:
                print(f"[ReflectionRouter] Calling {p.model} ({p.calls + 1}/{p.max_calls})")
                resp = completion(
                    model=p.model,
                    messages=normalized,
                    api_key=api_key,
                    temperature=kwargs.get("temperature", self.default_temperature),
                    max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
                )
                p.calls += 1
                p.next_allowed_ts = time.time() + p.cooldown_sec
                text = resp["choices"][0]["message"]["content"]
                self.cache[key] = text
                self._save_cache()
                return text
            except Exception as e:
                last_error = e
                p.failures += 1
                backoff = min(120.0, p.cooldown_sec * (2 ** min(p.failures, 5)))
                p.next_allowed_ts = time.time() + backoff
                print(f"[ReflectionRouter] {p.model} failed: {type(e).__name__}: {e}")
                print(f"[ReflectionRouter] Cooling {p.model} for {backoff:.1f}s.")

        raise RuntimeError(f"All reflection providers exhausted/failed. Last error: {last_error}")


def normalize_candidate(candidate: Any) -> str:
    if isinstance(candidate, dict):
        if "system_prompt" in candidate:
            return str(candidate["system_prompt"])
        if "prompt" in candidate:
            return str(candidate["prompt"])
        if candidate:
            return str(next(iter(candidate.values())))
        return ""
    return str(candidate)


@dataclass
class RunCfg:
    max_new_tokens: int
    temperature: float


def build_evaluator(model: LocalGGUFModel, cfg: RunCfg):
    def evaluator(candidate: Any, example: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        global EVAL_COUNTER
        EVAL_COUNTER += 1
        print(
            f"[Evaluator] local eval #{EVAL_COUNTER} | id={example.get('id')} | task={example.get('task_type')}",
            flush=True,
        )
        system_prompt = normalize_candidate(candidate)
        prompt = render_prompt(system_prompt, str(example["input"]))
        output = model.generate(prompt, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
        score, side = evaluate_output(output, example)
        print(f"[Evaluator] done #{EVAL_COUNTER} | score={score:.4f}", flush=True)

        try:
            import gepa.optimize_anything as oa
            oa.log(f"Example ID: {example.get('id')}")
            oa.log(f"Task type: {example.get('task_type')}")
            oa.log(f"Question: {example.get('input')}")
            oa.log(f"Reference: {example.get('reference_answer', '')}")
            oa.log(f"Gold option: {example.get('gold_option', '')}")
            oa.log(f"Output: {output}")
            oa.log(f"Eval details: {json.dumps(side, ensure_ascii=False)}")
        except Exception:
            pass

        return score, {
            "example_id": example.get("id"),
            "question": example.get("input"),
            "reference_answer": example.get("reference_answer", ""),
            "gold_option": example.get("gold_option", ""),
            **side,
        }
    return evaluator


def run_eval_csv(name: str, system_prompt: str, model: LocalGGUFModel, rows: List[dict], cfg: RunCfg, out_dir: Path) -> Dict[str, Any]:
    csv_path = out_dir / f"{name}_outputs.csv"
    scores = []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "task_type", "input", "output", "score", "kannada_ratio",
            "bad_scripts", "content_score", "content_reason", "format_score", "failures"
        ])
        writer.writeheader()
        for idx, ex in enumerate(rows, 1):
            if idx == 1 or idx % 10 == 0 or idx == len(rows):
                print(f"[Eval:{name}] {idx}/{len(rows)}", flush=True)
            output = model.generate(render_prompt(system_prompt, str(ex["input"])), cfg.max_new_tokens, cfg.temperature)
            score, side = evaluate_output(output, ex)
            scores.append(score)
            writer.writerow({
                "id": ex.get("id"),
                "task_type": ex.get("task_type"),
                "input": ex.get("input"),
                "output": output,
                "score": side["score"],
                "kannada_ratio": side["kannada_ratio"],
                "bad_scripts": json.dumps(side["bad_scripts"], ensure_ascii=False),
                "content_score": side["content_score"],
                "content_reason": side["content_reason"],
                "format_score": side["format_score"],
                "failures": json.dumps(side["failures"], ensure_ascii=False),
            })
    return {"name": name, "n": len(scores), "avg_score": round(sum(scores)/max(1, len(scores)), 4), "csv": str(csv_path)}


def extract_best_candidate(result: Any) -> str:
    if hasattr(result, "best_candidate"):
        return normalize_candidate(getattr(result, "best_candidate"))
    if isinstance(result, dict):
        for k in ["best_candidate", "candidate", "best"]:
            if k in result:
                return normalize_candidate(result[k])
    if hasattr(result, "candidate"):
        return normalize_candidate(getattr(result, "candidate"))
    raise RuntimeError("Could not extract GEPA best candidate. Inspect gepa_result_repr.txt.")


def parse_router(args) -> List[Dict[str, Any]]:
    if args.router_json:
        providers = json.loads(args.router_json)
        if not isinstance(providers, list):
            raise ValueError("--router-json must be a JSON list")
        return providers

    models = [m.strip() for m in args.reflection_models.split(",") if m.strip()]
    envs = [e.strip() for e in args.api_key_envs.split(",") if e.strip()]
    if len(envs) == 1 and len(models) > 1:
        envs = envs * len(models)
    if len(models) != len(envs):
        raise ValueError("Number of --reflection-models must match --api-key-envs, unless only one env is provided.")

    return [{
        "model": m,
        "api_key_env": e,
        "max_calls": max(1, args.reflection_max_calls_per_provider),
        "cooldown_sec": args.reflection_cooldown_sec,
    } for m, e in zip(models, envs)]


def run_gepa(args, model: LocalGGUFModel, rows: List[dict], out_dir: Path) -> str:
    train, val, test = split_rows(rows)
    print(f"[GEPA] train={len(train)} val={len(val)} test={len(test)}")

    cfg = RunCfg(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    summaries = []

    print("[GEPA] Baseline validation evaluation...")
    summaries.append(run_eval_csv("baseline_val", BASE_SYSTEM_PROMPT, model, val, cfg, out_dir))

    if args.no_gepa:
        (out_dir / "optimized_system_prompt.txt").write_text(BASE_SYSTEM_PROMPT, encoding="utf-8")
        (out_dir / "gepa_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        return BASE_SYSTEM_PROMPT

    from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

    router = RotatingReflectionLM(
        providers=parse_router(args),
        cache_path=str(out_dir / "reflection_cache.json"),
        max_retries_per_provider=args.reflection_retries_per_provider,
        default_temperature=args.reflection_temperature,
        default_max_tokens=args.reflection_max_tokens,
    )

    # Force conservative serial execution to avoid llama.cpp CUDA memory-pool crashes.
    try:
        engine = EngineConfig(
            max_metric_calls=args.max_metric_calls,
            parallel=False,
            max_workers=1,
            cache_evaluation=True,
            capture_stdio=True,
        )
    except TypeError:
        try:
            engine = EngineConfig(
                max_metric_calls=args.max_metric_calls,
                parallel=False,
                max_workers=1,
                cache_evaluation=True,
            )
        except TypeError:
            try:
                engine = EngineConfig(
                    max_metric_calls=args.max_metric_calls,
                    parallel=False,
                    max_workers=1,
                )
            except TypeError:
                engine = EngineConfig(max_metric_calls=args.max_metric_calls)

    try:
        reflection_cfg = ReflectionConfig(
            reflection_lm=router,
            reflection_minibatch_size=3,
        )
    except TypeError:
        reflection_cfg = ReflectionConfig(reflection_lm=router)

    try:
        config = GEPAConfig(
            engine=engine,
            reflection=reflection_cfg,
        )
    except TypeError:
        # Compatibility fallback for older GEPA versions.
        config = GEPAConfig(
            reflection_lm=router,
            engine=engine,
        )

    print("[GEPA] Starting optimize_anything...")
    result = optimize_anything(
        seed_candidate={"system_prompt": BASE_SYSTEM_PROMPT},
        evaluator=build_evaluator(model, cfg),
        dataset=train,
        valset=val,
        objective=OBJECTIVE,
        background=BACKGROUND,
        config=config,
    )

    best_prompt = extract_best_candidate(result)
    (out_dir / "optimized_system_prompt.txt").write_text(best_prompt, encoding="utf-8")
    (out_dir / "reflection_usage.json").write_text(json.dumps(router.usage_summary(), ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        (out_dir / "gepa_result_repr.txt").write_text(repr(result), encoding="utf-8")
    except Exception:
        pass

    print("[GEPA] Optimized validation evaluation...")
    summaries.append(run_eval_csv("optimized_val", best_prompt, model, val, cfg, out_dir))
    if test:
        summaries.append(run_eval_csv("baseline_gepa_test", BASE_SYSTEM_PROMPT, model, test, cfg, out_dir))
        summaries.append(run_eval_csv("optimized_gepa_test", best_prompt, model, test, cfg, out_dir))

    (out_dir / "gepa_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ OPTIMIZED SYSTEM PROMPT ================\n")
    print(best_prompt)
    print("\n=========================================================\n")
    print("[Reflection usage]", json.dumps(router.usage_summary(), ensure_ascii=False, indent=2))
    return best_prompt


def load_milu_test(n: int, seed: int, split: str) -> List[dict]:
    from datasets import load_dataset

    print(f"[MILU] Loading ai4bharat/MILU Kannada {split}")
    ds = load_dataset("ai4bharat/MILU", "Kannada", split=split)
    rows = []
    for ex in ds:
        ex = dict(ex)
        q = clean_text(ex.get("question"))
        options = parse_options_from_row(ex)
        if not q or len(options) < 4:
            continue
        gold = normalize_option_label(ex.get("target"), options)
        if gold not in {"A", "B", "C", "D"}:
            continue
        rows.append({
            "id": f"milu_{len(rows)+1:05d}",
            "task_type": "mcq",
            "input": format_mcq(q, options[:4]),
            "gold_option": gold,
            "reference_answer": gold,
            "domain": clean_text(ex.get("domain")),
            "subject": clean_text(ex.get("subject")),
        })
    if n and n > 0 and len(rows) > n:
        rows = random.Random(seed).sample(rows, n)
    print(f"[MILU] Prepared {len(rows)} rows.")
    return rows


def run_milu_eval(name: str, system_prompt: str, model: LocalGGUFModel, rows: List[dict], args, out_dir: Path) -> Dict[str, Any]:
    csv_path = out_dir / f"{name}_milu_outputs.csv"
    total = correct = 0
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id","gold","prediction","correct","output","domain","subject","input"])
        writer.writeheader()
        for i, ex in enumerate(rows, 1):
            output = model.generate(render_prompt(system_prompt, ex["input"]), args.milu_max_new_tokens, args.milu_temperature)
            pred = extract_option(output)
            gold = normalize_option_label(ex["gold_option"])
            ok = int(pred == gold)
            correct += ok
            total += 1
            if i % 100 == 0:
                print(f"[{name}] {i}/{len(rows)} acc={correct/total:.4f}")
            writer.writerow({
                "id": ex.get("id"),
                "gold": gold,
                "prediction": pred or "",
                "correct": ok,
                "output": output,
                "domain": ex.get("domain",""),
                "subject": ex.get("subject",""),
                "input": ex.get("input",""),
            })
    return {"name": name, "rows": total, "correct": correct, "accuracy": round(correct/max(1,total), 6), "csv": str(csv_path)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset-path", type=str, required=True)
    ap.add_argument("--gguf-path", type=str, default=None)
    ap.add_argument("--hf-repo-id", type=str, default=None)
    ap.add_argument("--hf-filename", type=str, default=None)
    ap.add_argument("--hf-revision", type=str, default=None)
    ap.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    ap.add_argument("--hf-local-dir", type=str, default="/kaggle/working/hf_model")

    ap.add_argument("--router-json", type=str, default=None)
    ap.add_argument("--reflection-models", type=str, default="gemini/gemini-2.5-flash")
    ap.add_argument("--api-key-envs", type=str, default="GEMINI_API_KEY")
    ap.add_argument("--reflection-max-calls-per-provider", type=int, default=25)
    ap.add_argument("--reflection-cooldown-sec", type=float, default=5.0)
    ap.add_argument("--reflection-retries-per-provider", type=int, default=2)
    ap.add_argument("--reflection-temperature", type=float, default=1.0)
    ap.add_argument("--reflection-max-tokens", type=int, default=4096)

    ap.add_argument("--max-metric-calls", type=int, default=40)
    ap.add_argument("--no-gepa", action="store_true")

    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--n-threads", type=int, default=0)
    ap.add_argument("--n-batch", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--run-milu-eval", action="store_true")
    ap.add_argument("--milu-eval-n", type=int, default=0)
    ap.add_argument("--milu-split", type=str, default="test")
    ap.add_argument("--milu-max-new-tokens", type=int, default=64)
    ap.add_argument("--milu-temperature", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="/kaggle/working/gepa_kannada_run")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Config] Serial-safe GEPA runner enabled.", flush=True)
    print(f"[Config] n_ctx={args.n_ctx}, n_batch={args.n_batch}, max_new_tokens={args.max_new_tokens}, n_gpu_layers={args.n_gpu_layers}", flush=True)
    print(f"[Config] max_metric_calls={args.max_metric_calls}", flush=True)

    rows = read_jsonl(args.dataset_path)
    write_jsonl(rows, str(out_dir / "gepa_dataset_used.jsonl"))
    print(f"[Dataset] Loaded {len(rows)} rows from {args.dataset_path}")

    if args.gguf_path:
        gguf_path = args.gguf_path
    else:
        if not args.hf_repo_id:
            raise SystemExit("Provide either --gguf-path or --hf-repo-id.")
        gguf_path = resolve_hf_gguf(args.hf_repo_id, args.hf_filename, args.hf_revision, args.hf_token_env, args.hf_local_dir)

    print("[Model] Loading GGUF...")
    model = LocalGGUFModel(
        gguf_path=gguf_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads or None,
        n_batch=args.n_batch,
    )

    best_prompt = run_gepa(args, model, rows, out_dir)

    if args.run_milu_eval:
        milu_rows = load_milu_test(args.milu_eval_n, args.seed, args.milu_split)
        base = run_milu_eval("baseline", BASE_SYSTEM_PROMPT, model, milu_rows, args, out_dir)
        opt = run_milu_eval("optimized", best_prompt, model, milu_rows, args, out_dir)
        summary = {
            "note": "MILU was not used for GEPA optimization by this script.",
            "baseline": base,
            "optimized": opt,
        }
        (out_dir / "milu_final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[MILU summary]", json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n[DONE]")
    print(f"Run directory: {out_dir}")
    print(f"Optimized prompt: {out_dir / 'optimized_system_prompt.txt'}")
    print(f"GEPA summary: {out_dir / 'gepa_summary.json'}")


if __name__ == "__main__":
    main()
