# ============================================================
# Gemma 3 4B Kannada QGen fine-tuning
# Flow:
#   1) Load Google Gemma 3 base
#   2) Load CPT LoRA adapter explicitly
#   3) Merge CPT adapter into base
#   4) Save merged model with 1GB shards (low CPU RAM safe)
#   5) Reload merged model in 4-bit
#   6) Add fresh LoRA for QGen
#   7) Manually tokenize dataset with token_type_ids
#   8) Train with SFTTrainer + custom collator
#
# Intended stack:
#   torch          2.5.1+cu118
#   transformers   4.50.3+
#   peft           0.18.0+
#   trl            0.17.0
#   bitsandbytes   0.45.5
#   accelerate     1.6.0
#   datasets       3.5.0
# ============================================================

import os
import gc
import torch
from dataclasses import dataclass
from typing import Dict, List, Any

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# -----------------------------
# 0. Environment
# -----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# 1. Config
# -----------------------------
BASE_MODEL_ID = "google/gemma-3-4b-pt"
CPT_ADAPTER_ID = "vaibhav-mugulavalli-2004/gemma-3-4b-kannada-CPT-LoRA"

MERGED_MODEL_DIR = "./gemma3-4b-kannada-cpt-merged"
OUTPUT_DIR = "./gemma3-kn-qgen"

MAX_SEQ_LEN = 512

# New QGen LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_RATIO = 0.03
SEED = 42

# Merge control
FORCE_REMERGE = False   # set True if you want to rebuild merged model
MERGE_SAVE_SHARD_SIZE = "1GB"

# -----------------------------
# 2. Utility
# -----------------------------
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------
# 3. Tokenizer
# -----------------------------
print(f"Loading tokenizer from base model: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# 4. Merge CPT adapter into base
# -----------------------------
merged_config_path = os.path.join(MERGED_MODEL_DIR, "config.json")

if FORCE_REMERGE or not os.path.exists(merged_config_path):
    print("\n[Stage 1] Building merged CPT model...")
    cleanup()

    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    cpt_model = PeftModel.from_pretrained(
        base_model,
        CPT_ADAPTER_ID,
        is_trainable=False,
    )

    print("Merging CPT adapter into base weights...")
    merged_model = cpt_model.merge_and_unload()

    del base_model
    del cpt_model
    cleanup()

    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

    print(f"Saving merged model with max_shard_size={MERGE_SAVE_SHARD_SIZE} ...")
    merged_model.save_pretrained(
        MERGED_MODEL_DIR,
        safe_serialization=True,
        max_shard_size=MERGE_SAVE_SHARD_SIZE,
    )
    tokenizer.save_pretrained(MERGED_MODEL_DIR)

    print(f"Merged model saved to: {MERGED_MODEL_DIR}")

    del merged_model
    cleanup()
else:
    print(f"\n[Stage 1] Reusing existing merged model at: {MERGED_MODEL_DIR}")

# -----------------------------
# 5. Reload merged model in 4-bit for QLoRA
# -----------------------------
print("\n[Stage 2] Loading merged model in 4-bit for QGen LoRA training...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Gemma3ForConditionalGeneration.from_pretrained(
    MERGED_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

qgen_lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, qgen_lora_config)
model.print_trainable_parameters()

# -----------------------------
# 6. Dataset (direct HF loader -> direct tokenized dataset)
# -----------------------------
print("\n[Stage 3] Loading dataset...")
dataset = load_dataset("ai4bharat/IndicQuestionGeneration", "kn")

# Keep this moderate on Kaggle free tier
MAX_SEQ_LEN = 512

# Gemma-turn style prompt for question generation:
# user gives context + answer, model generates question
def build_user_prompt(context: str, answer: str) -> str:
    context = (context or "").strip()
    answer = (answer or "").strip()

    return (
        "<start_of_turn>user\n"
        "ಕೆಳಗಿನ ಪಠ್ಯ ಮತ್ತು ಉತ್ತರದ ಆಧಾರದಲ್ಲಿ ಸಹಜವಾದ ಕನ್ನಡ ಪ್ರಶ್ನೆಯನ್ನು ರಚಿಸು.\n\n"
        f"ಪಠ್ಯ:\n{context}\n\n"
        f"ಉತ್ತರ:\n{answer}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

def build_model_response(question: str) -> str:
    question = (question or "").strip()
    return question + "\n<end_of_turn>"

def is_valid_example(example):
    context = (example.get("context") or "").strip()
    answer = (example.get("answer") or "").strip()
    question = (example.get("question") or "").strip()

    if not context or not answer or not question:
        return False

    return True

train_raw = dataset["train"].filter(is_valid_example, num_proc=4)
eval_raw = dataset["validation"].filter(is_valid_example, num_proc=4)

print(f"Filtered train samples: {len(train_raw)}")
print(f"Filtered eval  samples: {len(eval_raw)}")

# -----------------------------
# 7. Direct tokenization from raw dataset
#    - no intermediate JSONL/text dataset
#    - labels only for assistant response
# -----------------------------
def tokenize_qgen_example(example):
    user_prompt = build_user_prompt(example["context"], example["answer"])
    model_response = build_model_response(example["question"])

    prompt_enc = tokenizer(
        user_prompt,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    response_enc = tokenizer(
        model_response,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    prompt_ids = prompt_enc["input_ids"]
    response_ids = response_enc["input_ids"]

    input_ids = prompt_ids + response_ids
    labels = ([-100] * len(prompt_ids)) + response_ids

    # Hard truncate to max length
    input_ids = input_ids[:MAX_SEQ_LEN]
    labels = labels[:MAX_SEQ_LEN]

    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }

train_dataset = train_raw.map(
    tokenize_qgen_example,
    remove_columns=train_raw.column_names,
    num_proc=4,
)

eval_dataset = eval_raw.map(
    tokenize_qgen_example,
    remove_columns=eval_raw.column_names,
    num_proc=4,
)

print("\nTokenized sample keys:", train_dataset.column_names)
print("Tokenized sample length:", len(train_dataset[0]["input_ids"]))

# Debug sample
debug_idx = 0
debug_user = build_user_prompt(train_raw[debug_idx]["context"], train_raw[debug_idx]["answer"])
debug_resp = build_model_response(train_raw[debug_idx]["question"])
print("\n=== Debug formatted sample ===")
print((debug_user + debug_resp)[:1200])

# -----------------------------
# 8. Custom collator that preserves token_type_ids + labels
# -----------------------------
@dataclass
class Gemma3QGenCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Extract labels before padding
        labels = [f["labels"] for f in features]

        # Remove labels temporarily
        features_no_labels = [
            {k: v for k, v in f.items() if k != "labels"}
            for f in features
        ]

        # Pad everything else
        batch = self.tokenizer.pad(
            features_no_labels,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Manually pad labels to same length
        max_len = batch["input_ids"].shape[1]

        padded_labels = []
        for l in labels:
            if len(l) > max_len:
                l = l[:max_len]
            else:
                l = l + [-100] * (max_len - len(l))
            padded_labels.append(l)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        # Safety fallback
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

        return batch

data_collator = Gemma3QGenCollator(tokenizer=tokenizer)

# -----------------------------
# 9. Trainer config
# -----------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    run_name="gemma3-kn-qgen-merged-cpt-base",

    # already tokenized
    max_length=MAX_SEQ_LEN,
    packing=False,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,

    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    optim="paged_adamw_8bit",

    bf16=True,
    fp16=False,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    dataloader_num_workers=4,
    report_to="none",
    seed=SEED,

    remove_unused_columns=False,
)

# -----------------------------
# 10. Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    args=sft_config,
)

# -----------------------------
# 11. Train
# -----------------------------
print("\n[Stage 4] Starting training...")
trainer.train()

# -----------------------------
# 12. Save new QGen LoRA adapter
# -----------------------------
adapter_out = os.path.join(OUTPUT_DIR, "qgen-lora-adapter")
os.makedirs(adapter_out, exist_ok=True)

model.save_pretrained(adapter_out)
tokenizer.save_pretrained(adapter_out)

print(f"\nQGen LoRA adapter saved to: {adapter_out}")

# -----------------------------
# 13. Quick inference check
# -----------------------------
print("\n[Stage 5] Quick inference check...")
model.eval()

sample_context = (
    "1860ರ ದಶಕದಲ್ಲಿ ಕಾರ್ಲ್ ಹೆನ್ರಿಚ್ ಉಲ್ರಿಚ್ ಅವರು ಖಾಸಗಿಯಾಗಿ ಪ್ರಕಟಿಸಿದ "
    "ಕರಪತ್ರಗಳ ಸರಣಿಯಲ್ಲಿ ಅತ್ಯಂತ ಮುಂಚಿನ ಲೈಂಗಿಕ ದೃಷ್ಟಿಕೋನದ ವರ್ಗೀಕರಣದ "
    "ಯೋಜನೆಗಳನ್ನು ಪ್ರಸ್ತಾಪಿಸಿದರು."
)
sample_answer = "1860 ರಲ್ಲಿ"

inference_prompt = build_user_prompt(sample_context, sample_answer)

inputs = tokenizer(
    inference_prompt,
    return_tensors="pt",
    add_special_tokens=False,
    return_token_type_ids=True,
).to(model.device)

if "token_type_ids" not in inputs:
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=False,
)

print("\n=== Inference check ===")
print("Raw generated:", repr(generated))

# Optional cleanup for display
generated_clean = generated.split("<end_of_turn>")[0].strip()
print("Generated question:", generated_clean)