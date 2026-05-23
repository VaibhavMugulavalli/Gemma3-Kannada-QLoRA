#!/usr/bin/env python3
"""
Curriculum-aware continued pretraining (CPT) for Gemma on Kannada data.

Optimized for:
- 21GB VRAM GPU (e.g. RTX 3090 / A5000 / A100 20GB)
- gemma-3-4b-pt (base/pretrained checkpoint) with QLoRA 4-bit
- Two sources: textbook JSONL + IndicCorp JSONL/JSONL.GZ
- Stage-wise curriculum over textbook difficulty buckets
- QLoRA + bf16 + FlashAttention 2 (with SDPA fallback)
- Crash-friendly checkpointing + automatic resume across phases

Key changes from original for 21GB VRAM:
- model_name   : google/gemma-3-4b-pt  (4B base, not 12B)
- micro_batch  : 1  (safe for 21GB)
- grad_accum   : 16 (effective batch = 16)
- max_length   : 1024 (reduced from 2048 to save activation memory)
- lora_r       : 16  (reduced from 32 to save adapter memory)
- lora_alpha   : 32  (2x lora_r as convention)
- total_steps  : 1800 (unchanged)
- attn_impl    : sdpa (flash_attention_2 needs specific install; sdpa is safe default)

Key output folders:
  output_dir/checkpoints/phase{N}/checkpoint-XXXX   -> resumable Trainer checkpoints
  output_dir/phase_adapters/phase{N}                -> compact PEFT adapter after each phase
  output_dir/final_adapter                          -> final CPT adapter + tokenizer + metadata
  output_dir/run_state.json                         -> phase progress manifest

Example:
python gemma_kannada_cpt_curriculum.py \
  --textbook_jsonl /path/to/clean_dataset.jsonl \
  --indiccorp_jsonl /path/to/indiccorp_kn.jsonl.gz \
  --output_dir /path/to/output \
  --hf_token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset

from datasets import disable_caching
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


disable_caching()


# ─────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # ── Model ──────────────────────────────────────────────────────────────
    # gemma-3-4b-pt is the base (pretrained) checkpoint — ideal for CPT.
    # Using 4B instead of 12B because 21 GB VRAM is tight for training;
    # 4B + QLoRA leaves ~6-7 GB headroom for activations / optimizer state.
    p.add_argument("--model_name", type=str, default="google/gemma-3-4b-pt")

    # ── Data ───────────────────────────────────────────────────────────────
    p.add_argument("--textbook_jsonl",  type=str, required=True)
    p.add_argument("--indiccorp_jsonl", type=str, nargs="+", required=True)
    p.add_argument("--output_dir",      type=str, required=True)
    p.add_argument("--hf_token",        type=str, default=None)

    # ── Sequence / training sizes ──────────────────────────────────────────
    # max_length reduced to 1024 (from 2048) to keep activation memory
    # within 21 GB during backward pass with gradient checkpointing.
    p.add_argument("--max_length",                  type=int,   default=2048)
    p.add_argument("--micro_batch_size",            type=int,   default=2)
    # effective batch = micro_batch_size * grad_accum = 1 * 16 = 16
    p.add_argument("--gradient_accumulation_steps", type=int,   default=16)
    p.add_argument("--learning_rate",               type=float, default=1.5e-4)
    p.add_argument("--weight_decay",                type=float, default=0.01)
    p.add_argument("--warmup_ratio",                type=float, default=0.03)
    p.add_argument("--logging_steps",               type=int,   default=10)
    p.add_argument("--save_steps",                  type=int,   default=250)
    p.add_argument("--save_total_limit",            type=int,   default=3)
    p.add_argument("--seed",                        type=int,   default=42)

    # ── Phase budget ───────────────────────────────────────────────────────
    p.add_argument("--total_steps",   type=int,   default=12000)
    p.add_argument("--phase1_frac",   type=float, default=0.15)
    p.add_argument("--phase2_frac",   type=float, default=0.70)
    p.add_argument("--phase3_frac",   type=float, default=0.15)

    # ── Source mixing per phase ────────────────────────────────────────────
    p.add_argument("--phase1_textbook_ratio", type=float, default=0.3)
    p.add_argument("--phase2_textbook_ratio", type=float, default=0.15)
    p.add_argument("--phase3_textbook_ratio", type=float, default=0.1)

    # ── Optional token caps (0 = unlimited) ───────────────────────────────
    p.add_argument("--phase1_token_cap", type=int, default=0)
    p.add_argument("--phase2_token_cap", type=int, default=0)
    p.add_argument("--phase3_token_cap", type=int, default=0)

    # ── LoRA / quantization ────────────────────────────────────────────────
    # lora_r=16, lora_alpha=32 keeps adapter small; use 32/64 if you have
    # headroom and want more expressive capacity.
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_4bit",     action="store_true", default=True)
    p.add_argument("--disable_4bit", action="store_true")

    # attn_impl: sdpa is the safe default for CUDA 11.8 without the
    # flash-attn wheel installed. Switch to flash_attention_2 only after:
    #   pip install flash-attn --no-build-isolation
    p.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    # ── Data cleanup and curriculum ────────────────────────────────────────
    p.add_argument("--min_chars",        type=int, default=40)
    p.add_argument("--max_chars",        type=int, default=12000)
    p.add_argument("--textbook_buckets", type=int, default=5)
    p.add_argument("--repeat_textbook",    action="store_true", default=True)
    p.add_argument("--no_repeat_textbook", action="store_true")
    p.add_argument("--num_workers",      type=int, default=4)

    # ── Resume / export ────────────────────────────────────────────────────
    p.add_argument("--resume_from_checkpoint",   type=str, default=None)
    p.add_argument("--reset_run_state",          action="store_true")
    p.add_argument("--save_merged_16bit_model",  action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────
# Text normalisation utilities
# ─────────────────────────────────────────

_SOFT_HYPHENS = "\u00ad\u200b\ufeff"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    for ch in _SOFT_HYPHENS:
        text = text.replace(ch, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


GRADE_PAT   = re.compile(r"(?:^|[_-])g?(\d{1,2})(?:[_-]|$)", flags=re.IGNORECASE)
SAMPLE_PAT  = re.compile(r"(?:^|[_-])s(\d{1,4})(?:[_-]|$)",  flags=re.IGNORECASE)
CHAPTER_PAT = re.compile(
    r"(?:chapter|chap|ch|lesson|p[aà]tha|paatha|unit|adhyaya|ಪಾಠ|ಅಧ್ಯಾಯ)[_ :\-]*(\d{1,3})",
    flags=re.IGNORECASE,
)


def try_int(x, default=None):
    try:
        return int(x) if x is not None else default
    except Exception:
        return default


@dataclass
class TextbookRecord:
    text:       str
    meta:       Dict
    difficulty: float
    bucket:     int
    uid:        str


@dataclass
class IndicRecord:
    text: str
    meta: Dict


# ─────────────────────────────────────────
# Textbook loading + difficulty
# ─────────────────────────────────────────

def infer_grade(meta: Dict) -> int:
    for key in ["grade", "class", "std", "standard", "level"]:
        if key in meta:
            v = try_int(meta.get(key))
            if v is not None:
                return v
    raw_id = str(meta.get("id", ""))
    m = GRADE_PAT.search(raw_id)
    if m:
        return int(m.group(1))
    return 99


def infer_sample_order(meta: Dict) -> int:
    for key in ["sample_index", "sample_id", "order", "sequence", "seq", "position",
                "chapter_index", "chapter_order", "lesson_index"]:
        if key in meta:
            v = try_int(meta.get(key))
            if v is not None:
                return v
    raw_id = str(meta.get("id", ""))
    for pat in (SAMPLE_PAT, CHAPTER_PAT):
        m = pat.search(raw_id)
        if m:
            return int(m.group(1))
    return 999999


def infer_content_type(meta: Dict, text: str) -> str:
    for key in ["content_type", "type", "section_type", "genre"]:
        if key in meta and meta[key]:
            return str(meta[key]).lower()
    head = text[:160].lower()
    if any(k in head for k in ["ಪದ್ಯ", "poem"]):
        return "poem"
    if any(k in head for k in ["ವ್ಯಾಕರಣ", "grammar"]):
        return "grammar"
    if any(k in head for k in ["ಪ್ರಶ್ನೆ", "exercise", "ಅಭ್ಯಾಸ"]):
        return "exercise"
    return "prose"


def lexical_complexity_proxy(text: str) -> float:
    words = re.findall(r"\S+", text)
    if not words:
        return 0.0
    avg_word_len    = sum(len(w) for w in words) / max(len(words), 1)
    newline_density = text.count("\n") / max(len(text), 1)
    punct_density   = len(re.findall(r"[,:;?!।]", text)) / max(len(words), 1)
    return avg_word_len + 2.5 * punct_density + 5.0 * newline_density


def content_type_penalty(content_type: str) -> float:
    c = content_type.lower()
    if c in {"exercise", "worksheet", "qa", "question_answer"}:
        return 0.8
    if c in {"grammar", "technical", "essay"}:
        return 0.6
    if c in {"poem", "poetry"}:
        return 0.4
    return 0.0


def compute_textbook_difficulty(meta: Dict, text: str) -> float:
    grade        = infer_grade(meta)
    order        = infer_sample_order(meta)
    content_type = infer_content_type(meta, text)
    length_score = min(len(text) / 2000.0, 2.0)
    lex_score    = min(lexical_complexity_proxy(text) / 8.0, 2.0)
    order_score  = min(order / 200.0, 2.0)
    return (
        (1.4 * grade)
        + (0.8 * order_score)
        + (0.8 * length_score)
        + (0.7 * lex_score)
        + content_type_penalty(content_type)
    )


def load_jsonl(path: str) -> Iterator[Dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_textbook_records(
    path: str, min_chars: int, max_chars: int, n_buckets: int
) -> List[TextbookRecord]:
    temp: List[Tuple[str, Dict, float]] = []
    seen = set()
    for row in load_jsonl(path):
        text = normalize_text(str(row.get("text", "")))
        if len(text) < min_chars or len(text) > max_chars:
            continue
        if text in seen:
            continue
        seen.add(text)
        meta       = {k: v for k, v in row.items() if k != "text"}
        difficulty = compute_textbook_difficulty(meta, text)
        temp.append((text, meta, difficulty))

    temp.sort(key=lambda x: x[2])
    n = len(temp)
    if n == 0:
        raise ValueError("No textbook records survived preprocessing.")

    records: List[TextbookRecord] = []
    for i, (text, meta, difficulty) in enumerate(temp):
        bucket = min((i * n_buckets) // n, n_buckets - 1)
        records.append(
            TextbookRecord(
                text=text,
                meta=meta,
                difficulty=difficulty,
                bucket=bucket,
                uid=str(meta.get("id", i)),
            )
        )
    return records


# ─────────────────────────────────────────
# IndicCorp streaming
# ─────────────────────────────────────────

def indic_stream(
    paths: Sequence[str], min_chars: int, max_chars: int, seed: int = 42
) -> Iterator[IndicRecord]:
    rng = random.Random(seed)
    shuffled_paths = list(paths)
    while True:
        rng.shuffle(shuffled_paths)
        for path in shuffled_paths:
            for row in load_jsonl(path):
                text = normalize_text(str(row.get("text", "")))
                if len(text) < min_chars or len(text) > max_chars:
                    continue
                yield IndicRecord(
                    text=text,
                    meta={k: v for k, v in row.items() if k != "text"},
                )


# ─────────────────────────────────────────
# Curriculum-aware packed dataset
# ─────────────────────────────────────────

def default_bucket_probs(active_buckets: List[int]) -> List[float]:
    if len(active_buckets) == 1:
        return [1.0]
    weights = [1.0 / (rank + 1.0) for rank, _ in enumerate(active_buckets)]
    s = sum(weights)
    return [w / s for w in weights]


class MixedPackedDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        textbook_records:   List[TextbookRecord],
        indic_paths:        Sequence[str],
        max_length:         int,
        textbook_ratio:     float,
        active_buckets:     List[int],
        seed:               int,
        min_chars:          int,
        max_chars:          int,
        token_cap:          int  = 0,
        repeat_textbook:    bool = True,
    ):
        super().__init__()
        self.tokenizer        = tokenizer
        self.textbook_records = textbook_records
        self.indic_paths      = list(indic_paths)
        self.max_length       = max_length
        self.textbook_ratio   = textbook_ratio
        self.active_buckets   = sorted(active_buckets)
        self.seed             = seed
        self.min_chars        = min_chars
        self.max_chars        = max_chars
        self.token_cap        = token_cap
        self.repeat_textbook  = repeat_textbook

        self.bucket_to_records: Dict[int, List[TextbookRecord]] = {}
        for r in self.textbook_records:
            self.bucket_to_records.setdefault(r.bucket, []).append(r)

        self.bucket_probs = default_bucket_probs(self.active_buckets)
        self.eos = tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Tokenizer must have an EOS token.")

    def _textbook_iterator(self, rng: random.Random) -> Iterator[TextbookRecord]:
        pools:   Dict[int, List[TextbookRecord]] = {}
        cursors: Dict[int, int]                  = {}
        current_buckets = list(self.active_buckets)
        current_probs   = list(self.bucket_probs)

        for b in current_buckets:
            items = list(self.bucket_to_records.get(b, []))
            if not items:
                continue
            rng.shuffle(items)
            pools[b]   = items
            cursors[b] = 0

        if not pools:
            raise ValueError(f"No textbook records for active_buckets={current_buckets}")

        while True:
            bucket = rng.choices(current_buckets, weights=current_probs, k=1)[0]
            if bucket not in pools or not pools[bucket]:
                continue
            idx = cursors[bucket]
            if idx >= len(pools[bucket]):
                if not self.repeat_textbook:
                    del pools[bucket]
                    current_buckets = [b for b in current_buckets if b in pools]
                    if not current_buckets:
                        return
                    current_probs = default_bucket_probs(current_buckets)
                    continue
                rng.shuffle(pools[bucket])
                cursors[bucket] = 0
                idx = 0
            rec = pools[bucket][idx]
            cursors[bucket] += 1
            yield rec

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info is not None else 0
        rng         = random.Random(self.seed + worker_id)
        indic_iter    = indic_stream(self.indic_paths, self.min_chars, self.max_chars, seed=self.seed + worker_id)
        textbook_iter = self._textbook_iterator(rng)

        buffer:          List[int] = []
        produced_tokens: int       = 0

        while True:
            if self.token_cap > 0 and produced_tokens >= self.token_cap:
                return

            use_textbook = rng.random() < self.textbook_ratio
            if use_textbook:
                try:
                    text = next(textbook_iter).text
                except StopIteration:
                    text = next(indic_iter).text
            else:
                text = next(indic_iter).text

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            ids = ids + [self.eos]
            buffer.extend(ids)

            while len(buffer) >= self.max_length:
                chunk  = buffer[: self.max_length]
                buffer = buffer[self.max_length :]
                produced_tokens += len(chunk)
                yield {
                    "input_ids":      chunk,
                    "attention_mask": [1] * len(chunk),
                    "labels":         chunk.copy(),
                    "token_type_ids":  [0] * len(chunk),
                }


# ─────────────────────────────────────────
# Model / LoRA utilities
# ─────────────────────────────────────────

def find_target_modules(model) -> List[str]:
    target = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if any(x in last for x in
               ["q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "gate_proj", "down_proj"]):
            target.add(last)
    if not target:
        target = {"q_proj", "k_proj", "v_proj", "o_proj",
                  "up_proj", "gate_proj", "down_proj"}
    return sorted(target)


class PhaseBoundaryCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\n[TRAIN START] max_steps={args.max_steps}, output_dir={args.output_dir}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            used_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"[step={state.global_step}] VRAM={used_gb:.1f}GB  {logs}")


def make_quant_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if not (args.use_4bit and not args.disable_4bit):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(args: argparse.Namespace, attn_impl: str):
    quant_config = make_quant_config(args)
    return AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=args.hf_token,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )


def attach_new_lora(model, args: argparse.Namespace):
    model.config.use_cache = False
    # use_reentrant=False avoids a known memory leak in some torch versions
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if args.use_4bit and not args.disable_4bit:
        model = prepare_model_for_kbit_training(model)
    target_modules = find_target_modules(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    return model


def load_model_and_tokenizer(
    args: argparse.Namespace, adapter_dir: Optional[str] = None
):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=args.hf_token, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    chosen_attn = args.attn_impl
    try:
        model = load_base_model(args, chosen_attn)
    except Exception as e:
        print(f"Falling back to sdpa from {chosen_attn}: {e}")
        chosen_attn = "sdpa"
        model = load_base_model(args, chosen_attn)

    if adapter_dir:
        print(f"Loading existing PEFT adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model = attach_new_lora(model, args)

    model.print_trainable_parameters()
    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM after model load: {used_gb:.1f} / {total_gb:.1f} GB")
    print(f"Attention backend: {chosen_attn}")
    return model, tokenizer


# ─────────────────────────────────────────
# Run-state / checkpoint helpers
# ─────────────────────────────────────────

def run_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, "run_state.json")


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_run_state(output_dir: str) -> Dict:
    path = run_state_path(output_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed_phases":      [],
        "last_completed_phase":  None,
        "active_phase":          None,
        "active_checkpoint":     None,
        "last_adapter_dir":      None,
        "base_model_name":       None,
    }


def update_run_state(output_dir: str, **updates) -> Dict:
    state = load_run_state(output_dir)
    state.update(updates)
    save_json(run_state_path(output_dir), state)
    return state


def append_completed_phase(output_dir: str, phase_name: str, adapter_dir: str) -> Dict:
    state     = load_run_state(output_dir)
    completed = list(state.get("completed_phases", []))
    if phase_name not in completed:
        completed.append(phase_name)
    state.update(
        {
            "completed_phases":     completed,
            "last_completed_phase": phase_name,
            "active_phase":         None,
            "active_checkpoint":    None,
            "last_adapter_dir":     adapter_dir,
        }
    )
    save_json(run_state_path(output_dir), state)
    return state


def write_training_recipe(
    args: argparse.Namespace, tokenizer, final_dir: str
) -> None:
    recipe = {
        "base_model_name":    args.model_name,
        "adapter_type":       "PEFT QLoRA (4-bit NF4)",
        "vram_target":        "21 GB",
        "max_length":         args.max_length,
        "lora_r":             args.lora_r,
        "lora_alpha":         args.lora_alpha,
        "intended_reuse":     (
            "Load base model again with the same 4-bit config, "
            "then PeftModel.from_pretrained(base, final_adapter_dir, is_trainable=True) "
            "for further CPT or SFT."
        ),
        "recommended_next_step": (
            "For Kannada QA SFT: load gemma-3-4b-pt, attach this adapter, "
            "then continue LoRA/QLoRA training on your Kannada QA dataset."
        ),
        "notes": [
            "Do not convert to GGUF if you want to continue training.",
            "Tokenizer saved here matches the base model tokenizer.",
            "attn_impl=sdpa is safe for CUDA 11.8 without flash-attn wheel.",
            "To use flash_attention_2: pip install flash-attn --no-build-isolation",
        ],
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
        },
    }
    save_json(os.path.join(final_dir, "training_recipe.json"), recipe)


def maybe_export_merged_model(
    args: argparse.Namespace, final_adapter_dir: str
) -> None:
    if not args.save_merged_16bit_model:
        return
    print("Exporting merged 16-bit model for inference / archival...")
    merged_dir = os.path.join(args.output_dir, "merged_model_16bit")
    os.makedirs(merged_dir, exist_ok=True)
    quant_free_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=args.hf_token,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(quant_free_base, final_adapter_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(final_adapter_dir, use_fast=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged 16-bit model saved to: {merged_dir}")


# ─────────────────────────────────────────
# Phase definitions
# ─────────────────────────────────────────

def build_phase_steps(args: argparse.Namespace) -> List[int]:
    fracs = [args.phase1_frac, args.phase2_frac, args.phase3_frac]
    if not math.isclose(sum(fracs), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"phase fractions must sum to 1.0, got {fracs} -> {sum(fracs)}")
    p1 = int(round(args.total_steps * args.phase1_frac))
    p2 = int(round(args.total_steps * args.phase2_frac))
    p3 = args.total_steps - p1 - p2
    return [p1, p2, p3]


def phase_catalog(args: argparse.Namespace) -> List[Dict]:
    return [
        {
            "name":            "phase1",
            "textbook_ratio":  args.phase1_textbook_ratio,
            "active_buckets":  [0, 1],
            "seed_offset":     11,
            "token_cap":       args.phase1_token_cap,
        },
        {
            "name":            "phase2",
            "textbook_ratio":  args.phase2_textbook_ratio,
            "active_buckets":  [0, 1, 2, 3],
            "seed_offset":     22,
            "token_cap":       args.phase2_token_cap,
        },
        {
            "name":            "phase3",
            "textbook_ratio":  args.phase3_textbook_ratio,
            "active_buckets":  list(range(args.textbook_buckets)),
            "seed_offset":     33,
            "token_cap":       args.phase3_token_cap,
        },
    ]


def build_phase_dataset(
    args:              argparse.Namespace,
    tokenizer,
    textbook_records:  List[TextbookRecord],
    phase_def:         Dict,
) -> MixedPackedDataset:
    return MixedPackedDataset(
        tokenizer=tokenizer,
        textbook_records=textbook_records,
        indic_paths=args.indiccorp_jsonl,
        max_length=args.max_length,
        textbook_ratio=phase_def["textbook_ratio"],
        active_buckets=phase_def["active_buckets"],
        seed=args.seed + phase_def["seed_offset"],
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        token_cap=phase_def["token_cap"],
        repeat_textbook=args.repeat_textbook,
    )


def build_trainer(
    args:          argparse.Namespace,
    model,
    train_dataset,
    output_dir:    str,
    max_steps:     int,
) -> Trainer:
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        # paged_adamw_8bit keeps optimizer states in 8-bit — saves ~2 GB VRAM
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        seed=args.seed,
    )
    return Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[PhaseBoundaryCallback()],
    )


def save_phase_adapter(
    trainer,
    tokenizer,
    phase_adapter_dir: str,
    phase_name:        str,
    args:              argparse.Namespace,
) -> None:
    os.makedirs(phase_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(phase_adapter_dir)
    tokenizer.save_pretrained(phase_adapter_dir)
    save_json(
        os.path.join(phase_adapter_dir, "phase_metadata.json"),
        {
            "phase_name":       phase_name,
            "base_model_name":  args.model_name,
            "adapter_type":     "PEFT QLoRA 4-bit",
        },
    )


def pick_resume_checkpoint(
    args: argparse.Namespace, phase_ckpt_dir: str
) -> Optional[str]:
    if args.resume_from_checkpoint:
        return args.resume_from_checkpoint
    last_ckpt = get_last_checkpoint(phase_ckpt_dir) if os.path.isdir(phase_ckpt_dir) else None
    return last_ckpt


def phase_already_completed(state: Dict, phase_name: str) -> bool:
    return phase_name in set(state.get("completed_phases", []))


def choose_adapter_to_load(
    args: argparse.Namespace, state: Dict
) -> Optional[str]:
    last_adapter_dir = state.get("last_adapter_dir")
    if last_adapter_dir and os.path.isdir(last_adapter_dir):
        return last_adapter_dir
    final_adapter = os.path.join(args.output_dir, "final_adapter")
    if os.path.isdir(final_adapter):
        return final_adapter
    return None


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_repeat_textbook:
        args.repeat_textbook = False

    if args.reset_run_state:
        state_file = run_state_path(args.output_dir)
        if os.path.exists(state_file):
            os.remove(state_file)
        print("Removed existing run_state.json (--reset_run_state).")

    # ── Print VRAM summary before anything loads ───────────────────────────
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  Total VRAM: {total_gb:.1f} GB")

    # ── Load textbook data ─────────────────────────────────────────────────
    print("Loading textbook dataset...")
    textbook_records = load_textbook_records(
        path=args.textbook_jsonl,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        n_buckets=args.textbook_buckets,
    )
    bucket_counts: Dict[int, int] = {}
    for r in textbook_records:
        bucket_counts[r.bucket] = bucket_counts.get(r.bucket, 0) + 1
    print(f"Loaded {len(textbook_records)} textbook records")
    print(f"Bucket counts: {bucket_counts}")

    # ── Build phase plan ───────────────────────────────────────────────────
    phase_steps = build_phase_steps(args)
    phases      = phase_catalog(args)
    for phase_def, steps in zip(phases, phase_steps):
        phase_def["max_steps"] = steps
    print(f"Phase steps: {phase_steps}  (total={sum(phase_steps)})")

    # ── Load model ─────────────────────────────────────────────────────────
    state          = load_run_state(args.output_dir)
    adapter_to_load = choose_adapter_to_load(args, state)
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args, adapter_dir=adapter_to_load)
    update_run_state(
        args.output_dir,
        base_model_name=args.model_name,
        last_adapter_dir=adapter_to_load,
    )

    checkpoints_root  = os.path.join(args.output_dir, "checkpoints")
    phase_adapters_root = os.path.join(args.output_dir, "phase_adapters")
    os.makedirs(checkpoints_root,   exist_ok=True)
    os.makedirs(phase_adapters_root, exist_ok=True)

    # ── Phase loop ─────────────────────────────────────────────────────────
    for phase_def in phases:
        phase_name = phase_def["name"]
        max_steps  = phase_def["max_steps"]
        if max_steps <= 0:
            print(f"Skipping {phase_name} (max_steps=0)")
            continue

        state = load_run_state(args.output_dir)
        if phase_already_completed(state, phase_name):
            print(f"Skipping {phase_name} — already completed.")
            continue

        print(f"\n{'='*60}")
        print(f"  STARTING {phase_name.upper()}")
        print(f"  textbook_ratio={phase_def['textbook_ratio']} | "
              f"buckets={phase_def['active_buckets']} | steps={max_steps}")
        print(f"{'='*60}\n")

        train_dataset    = build_phase_dataset(args, tokenizer, textbook_records, phase_def)
        phase_ckpt_dir   = os.path.join(checkpoints_root,   phase_name)
        phase_adapter_dir = os.path.join(phase_adapters_root, phase_name)
        os.makedirs(phase_ckpt_dir,    exist_ok=True)
        os.makedirs(phase_adapter_dir, exist_ok=True)

        trainer = build_trainer(
            args=args,
            model=model,
            train_dataset=train_dataset,
            output_dir=phase_ckpt_dir,
            max_steps=max_steps,
        )

        resume_ckpt = pick_resume_checkpoint(args, phase_ckpt_dir)
        update_run_state(
            args.output_dir,
            active_phase=phase_name,
            active_checkpoint=resume_ckpt,
        )
        if resume_ckpt:
            print(f"Resuming {phase_name} from: {resume_ckpt}")
        else:
            print(f"Starting {phase_name} fresh.")

        trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_state()
        trainer.save_model(phase_ckpt_dir)
        save_phase_adapter(trainer, tokenizer, phase_adapter_dir, phase_name, args)
        append_completed_phase(args.output_dir, phase_name, phase_adapter_dir)
        print(f"✅  {phase_name} complete — adapter saved to: {phase_adapter_dir}")

        model = trainer.model
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            used_gb = torch.cuda.memory_allocated() / 1e9
            print(f"VRAM after phase cleanup: {used_gb:.1f} GB")

    # ── Save final adapter ─────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    write_training_recipe(args, tokenizer, final_dir)
    update_run_state(
        args.output_dir,
        active_phase=None,
        active_checkpoint=None,
        last_adapter_dir=final_dir,
    )
    print(f"\n✅  Final adapter saved to: {final_dir}")

    maybe_export_merged_model(args, final_dir)


if __name__ == "__main__":
    main()