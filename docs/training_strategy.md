# Training Strategy

This document explains the training strategy used for the Kannada-first Mayura-4B(Gemma 3 4B base) fine-tuning project. The project follows a staged post-training pipeline instead of training a model from scratch.

The main training path is:

```text
Base Gemma 3 4B
        ↓
Kannada Continued Pretraining (CPT)
        ↓
Kannada Supervised Fine-Tuning (SFT)
        ↓
Merged / exportable checkpoint
        ↓
GGUF conversion and local inference testing
        ↓
Kannada evaluation benchmarks
```

The goal is to adapt a capable general-purpose base model into a Kannada-first model using curated Kannada data while keeping the training practical on limited GPU resources.

---

## 1. Training Objectives

The training pipeline was designed around the following objectives:

- Improve Kannada generation quality on a small open model.
- Make the model more comfortable with Kannada script, vocabulary, grammar, and educational prose.
- Improve Kannada question-answering behavior through supervised instruction tuning.
- Preserve the base model's general reasoning ability as much as possible.
- Avoid unnecessary multilingual or code-mixed behavior.
- Keep the pipeline reproducible across college GPUs, Kaggle, and Colab where possible.
- Produce checkpoints that can be merged and exported to GGUF for local testing in tools such as LM Studio.

The project does not attempt to train a Kannada LLM from scratch. Instead, it performs domain and language adaptation on top of an existing pretrained instruction model.

---

## 2. Base Model

The base model used for adaptation was:

```text
google/gemma-3-4b-pt
```

This model was selected because it is small enough to fine-tune with parameter-efficient methods while still being large enough to retain useful reasoning and instruction-following ability.

Important base-model decisions:

- Use the original Gemma 3 tokenizer for the stable training path.
- Do not change tokenizer files between CPT, SFT, merge, and inference.
- Treat tokenizer extension as an experimental path, not the stable public pipeline.
- Use adapter-based fine-tuning to reduce GPU memory requirements.

---

## 3. Why CPT Followed by SFT

The training strategy separates language adaptation from instruction adaptation.

### 3.1 Continued Pretraining

CPT teaches the model to better model Kannada text distribution. It improves exposure to Kannada grammar, phrasing, textbook style, and native-script language patterns.

CPT is useful because the base model already knows some Kannada but may not have enough Kannada token exposure to behave reliably as a Kannada-first assistant.

### 3.2 Supervised Fine-Tuning

SFT teaches the model how to respond to user instructions and questions in Kannada.

SFT is performed after CPT because instruction tuning alone may not be enough if the model's underlying Kannada distribution is weak. CPT improves the model's Kannada language familiarity, while SFT improves answer behavior.

The intended separation is:

| Stage | Objective | Dataset Type | Loss Target |
|---|---|---|---|
| CPT | Kannada language adaptation | Raw Kannada text | Next-token prediction over text |
| SFT | Kannada instruction following | Kannada QA / instruction data | Assistant response tokens |

---

## 4. Stage 0: Data Preparation

Before training, datasets are cleaned and converted into model-ready formats.

### 4.1 CPT Data Preparation

CPT uses raw Kannada text from:

- KSEEB textbook content from Grade 3 to Grade 12 / 2nd PUC.
- IndicCorp v2 Kannada slice.

The processed CPT rows follow a simple text format:

```json
{"text": "...clean Kannada text block..."}
```

Important preprocessing steps:

- Unicode normalization.
- Removal of empty or very short rows.
- Filtering of noisy OCR output.
- Filtering of post-chapter exercises and activity pages.
- Removal of dotted answer-line regions.
- Deduplication where possible.
- Kannada character-ratio checks.
- Token count estimation.
- Train/validation split.

### 4.2 SFT Data Preparation

SFT uses the Kannada slice of:

```text
FoundryAILabs/k12-indian-curriculum-4.9m
```

The dataset is converted into Gemma-style chat/instruction format.

Preferred model-visible formatting:

```text
<start_of_turn>user
ಪ್ರಶ್ನೆ: ...
<end_of_turn>
<start_of_turn>model
...
<end_of_turn>
```

The training data should use Kannada-native labels such as `ಪ್ರಶ್ನೆ:` and `ಉತ್ತರ:` instead of English labels such as `question`, `answer`, or `prompt`.

Important SFT preprocessing steps:

- Keep Kannada-only or Kannada-majority rows.
- Remove malformed examples.
- Remove empty instructions or responses.
- Replace English prompt labels with Kannada equivalents.
- Keep the assistant response clean and direct.
- Apply assistant-only loss masking where possible.
- Run label sanity checks before training.

---

## 5. Stage 1: Kannada Continued Pretraining

### 5.1 Purpose

The CPT stage adapts the base model to Kannada text distribution. This stage is not intended to make the model a polished assistant. Its main purpose is to improve Kannada language modeling before instruction tuning.

Expected improvements from CPT:

- Better Kannada fluency.
- Better handling of Kannada Unicode script.
- Better school-textbook-style explanations.
- Lower tendency to switch into other languages.
- Better Kannada vocabulary coverage.

### 5.2 CPT Dataset Mixture

The intended CPT mixture is:

| Source | Approximate Role | Intended Weight |
|---|---|---:|
| KSEEB textbook corpus | High-quality educational Kannada | 80% |
| IndicCorp v2 Kannada | Broader general-domain Kannada | 20% |

The preferred implementation is token-level mixing rather than file-count mixing, because source files can differ greatly in length.

**TODO:** Confirm whether the final run used exact token-level 80:20 weighting or approximate source sampling.

### 5.3 Curriculum-Style Ordering

The CPT data order follows a curriculum-like idea:

1. Start with cleaner textbook-style Kannada.
2. Mix in broader IndicCorp v2 Kannada.
3. Pack examples into training sequences.
4. Train with periodic checkpointing.

The reasoning is that high-quality textbook data gives the model a clean Kannada signal early, while IndicCorp adds scale and variety.

### 5.4 CPT Training Objective

CPT uses standard causal language modeling.

For each token sequence, the model predicts the next token:

```text
x1, x2, x3, ..., xn
```

The model is trained to predict:

```text
x2, x3, x4, ..., x(n+1)
```

Loss is applied across the packed Kannada text sequence, except padding tokens.

### 5.5 CPT Fine-Tuning Method

The practical training path uses parameter-efficient fine-tuning rather than full fine-tuning.

Recommended adapter method:

```text
LoRA / QLoRA
```

Common target modules for Gemma-family transformer blocks:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```


### 5.6 CPT Checkpointing

Checkpointing is important because training was performed across constrained environments where session resets, GPU limits, or runtime limits may interrupt training.

Recommended checkpointing policy:

- Save checkpoints periodically.
- Keep only a limited number of recent checkpoints to control disk usage.
- Support resume-from-checkpoint.
- Log training loss and validation loss.
- Push important checkpoints to Hugging Face when local disk is unreliable.

Observed project note:

- CPT loss was reported around approximately `1.1` at around `1.2k` steps in one run.
- This should be treated as an intermediate training signal, not a final quality guarantee.

---

## 6. Stage 2: Kannada Supervised Fine-Tuning

### 6.1 Purpose

The SFT stage teaches the CPT-adapted model how to answer questions and follow instructions in Kannada.

Expected improvements from SFT:

- Better Kannada QA behavior.
- More direct answers.
- Better instruction following.
- Better school-level explanation style.
- Reduced prompt/template leakage.
- Better Kannada-only response compliance.

### 6.2 SFT Dataset

The final SFT dataset used was:

```text
FoundryAILabs/k12-indian-curriculum-4.9m
```

Only the Kannada slice was used for this project.

Approximate Kannada subset size:

```text
~408K rows
```

AI4Bharat IndicQuestionGeneration was explored but not used in the final SFT run.

### 6.3 SFT Format

The SFT examples are converted into chat format compatible with Gemma-style instruction training.

Example:

```text
<start_of_turn>user
ಪ್ರಶ್ನೆ: ಜಲ ಸಂರಕ್ಷಣೆ ಎಂದರೇನು?
<end_of_turn>
<start_of_turn>model
ಜಲ ಸಂರಕ್ಷಣೆ ಎಂದರೆ ನೀರನ್ನು ವ್ಯರ್ಥ ಮಾಡದೆ ಉಳಿಸಿ, ಸರಿಯಾಗಿ ಬಳಸುವ ಪ್ರಕ್ರಿಯೆ.
<end_of_turn>
```

Important formatting decisions:

- Use Kannada-visible labels.
- Keep the user prompt clear and short.
- Keep assistant responses in Kannada.
- Do not train the model to emit metadata fields.
- Do not expose JSON keys to the model unless intentionally training structured output.
- Avoid English labels such as `instruction`, `response`, `question`, and `answer` inside the prompt text.

### 6.4 Assistant-Only Loss

For SFT, the preferred objective is assistant-only loss.

This means:

- User prompt tokens are included in the input.
- Loss is masked on user/system/template tokens.
- Loss is applied only to the assistant response tokens.

This prevents the model from wasting training signal on learning to reproduce the prompt and helps it learn the answer behavior directly.

Sanity check before training:

```text
input shape: (...)
labels shape: (...)
PAD in labels: expected low / controlled
-100 in labels: should exist for masked prompt tokens
supervised token count: should be greater than zero
```

If supervised token count is zero or unexpectedly low, the SFT mask/template is likely broken.

### 6.5 SFT Training Budget

The SFT stage should be long enough to expose the model to the Kannada QA distribution, but not so long that it overfits or degrades the base model's general capabilities.

For constrained GPUs, a practical target is step-based training rather than full unlimited epochs.

Project target discussed:

```text
~10k–15k total SFT steps
```

The exact number should be chosen based on:

- Dataset size.
- Effective batch size.
- Sequence length.
- GPU memory.
- Training speed.
- Validation loss trend.
- Manual output quality.

**TODO:** Add final SFT step count and whether full dataset coverage was achieved.

---

## 7. Hardware and Environment Strategy

The project was trained and tested across multiple constrained environments.

Known environments:

| Environment | Role |
|---|---|
| College GPU notebooks / Kubeflow | Main training experiments |
| A100 MIG 3g.20gb | Higher-memory training runs |
| Kaggle GPUs | Verification, experiments, constrained training |
| Google Colab | Evaluation, conversion tests, debugging |
| Local machine / LM Studio | Manual GGUF inference testing |

Known constraints:

- College system limited to CUDA 11.8.
- FlashAttention support was limited by driver/CUDA compatibility.
- Some runs had low VRAM utilization or slow throughput.
- Kaggle/Colab runtimes required checkpoint-resume support.
- Disk limits required careful checkpoint retention.
- CPU RAM constraints affected model merge/conversion workflows.

Recommended environment approach:

- Avoid requiring FlashAttention in the public training script.
- Use `bf16` where supported, otherwise use `fp16`.
- Use gradient checkpointing to reduce VRAM usage.
- Use LoRA/QLoRA instead of full fine-tuning.
- Keep checkpoint saving predictable.
- Make all paths configurable for Kaggle, Colab, and local runs.

---

## 8. Precision and Memory Strategy

Because the model is 4B parameters, memory strategy matters.

Recommended setup:

| Component | Recommended Choice |
|---|---|
| Training method | LoRA / QLoRA |
| Compute dtype | `bf16` where supported, else `fp16` |
| Quantization during training | 4-bit QLoRA if VRAM constrained |
| Gradient checkpointing | Enabled |
| FlashAttention | Optional, not required |
| Sequence packing | Enabled where stable |
| Checkpoint retention | Limited |

For A100-class hardware, bf16 is preferred. For older GPUs such as P100/T4, fp16 or 4-bit QLoRA may be more practical depending on library compatibility.

---

## 9. Tokenizer Strategy

The stable training path uses the original Gemma 3 tokenizer.

The project explored Kannada tokenizer extension, but this was not used in the stable final path due to tokenizer merge and GGUF conversion issues.

Tokenizer rules:

- Use the same tokenizer for CPT, SFT, merging, and inference.
- Do not train with an extended tokenizer and infer with the base tokenizer.
- Do not overwrite tokenizer files after adapter training.
- During GGUF conversion, ensure the tokenizer files match the base Gemma tokenizer expected by the converter.

Reason:

Tokenizer mismatch can cause broken Kannada output, random multilingual leakage, and GGUF conversion failures.

Known issue:

```text
BPE pre-tokenizer was not recognized
unrecognized BPE tokenizer
No canonical BPE pre-tokenizer
```

---

## 10. Checkpoint and Merge Strategy

The training process produces adapter checkpoints first.

Recommended checkpoint stages:

```text
base model
  └── CPT LoRA adapter
        └── merged CPT checkpoint or CPT adapter path
              └── SFT LoRA adapter
                    └── merged final checkpoint
                          └── GGUF export
```

There are two practical ways to run SFT after CPT:

### Option A: Merge CPT, Then Train SFT

```text
Base model + CPT LoRA → merged CPT model → SFT LoRA → final merged model
```

Advantages:

- Cleaner separation between CPT and SFT adapters.
- Easier to inspect CPT checkpoint independently.
- Final SFT adapter is trained on the CPT-adapted model.

Disadvantages:

- Requires enough CPU RAM/disk for merging.
- Can be slower in constrained environments.

### Option B: Stack / Continue from CPT Adapter

```text
Base model + CPT LoRA → continue SFT from adapter/checkpoint → final adapter/merge
```

Advantages:

- Avoids intermediate full-model merge.
- More practical in constrained environments.

Disadvantages:

- Adapter composition and export can become harder.
- Requires careful tracking of which adapter was trained on which base.

**TODO:** Confirm which path was used for the final public checkpoint.

---

## 11. Evaluation During Training

Training loss alone is not enough to judge model quality.

Evaluation should happen at three levels:

1. Validation loss during CPT/SFT.
2. Automatic Kannada benchmarks after training.
3. Manual generation checks in LM Studio or a local inference script.

### 11.1 CPT Evaluation

CPT should be monitored using:

- Training loss.
- Validation loss.
- Kannada text continuation quality.
- Script consistency.
- Manual prompts before and after CPT.

CPT loss improvement does not guarantee good instruction-following. It only indicates improved next-token prediction on Kannada-like text.

### 11.2 SFT Evaluation

SFT should be monitored using:

- Training loss.
- Validation loss.
- Assistant-only supervised token count.
- Short Kannada QA prompts.
- MCQ-style prompts.
- Refusal/unknown-answer behavior.
- Kannada-only output compliance.

### 11.3 Benchmark Evaluation

The project uses the following evaluation datasets/benchmarks:

| Benchmark | Purpose | Metric / Signal |
|---|---|---|
| IndicQA Kannada | QA overlap and answer correctness | Exact Match, Contains Match, Token-F1, Char-F1 |
| MILU Kannada | Multitask Indic MCQ understanding | Accuracy / option-match accuracy |
| IndicGenBench | Indic generation quality | Task-specific generation metrics |
| GEPA Kannada evaluation set | Prompt optimization and behavior checks | Custom score / prompt-optimization feedback |
| Manual qualitative prompts | Fluency and Kannada-only behavior | Human inspection |

Known IndicQA Kannada full-run result:

| Metric | Value |
|---|---:|
| Rows | 1,138 |
| Exact Match | 0.1221 |
| Contains Match | 0.4271 |
| Token-F1 | 0.3086 |
| Char-F1 | 0.4990 |

**TODO:** Add final MILU Kannada score.

**TODO:** Add final IndicGenBench task-wise scores.

---

## 12. Inference Template Strategy

The model is expected to behave as a Kannada-first assistant. Inference templates should match the training format.

Recommended system behavior:

```text
ನೀನು ಕೇವಲ ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸುವ ನಿಖರ ಮತ್ತು ಸ್ಪಷ್ಟವಾದ AI ಸಹಾಯಕ.
ಬಳಕೆದಾರರ ಪ್ರಶ್ನೆಗಳಿಗೆ ನೇರವಾಗಿ, ಸಂಕ್ಷಿಪ್ತವಾಗಿ ಮತ್ತು ಕನ್ನಡದಲ್ಲೇ ಉತ್ತರಿಸು.
ಪ್ರಶ್ನೆಗೆ ಸರಿಯಾದ ಉತ್ತರ ತಿಳಿದಿಲ್ಲದಿದ್ದರೆ, ಪ್ರಾಮಾಣಿಕವಾಗಿ ಗೊತ್ತಿಲ್ಲ ಎಂದು ತಿಳಿಸು.
```

The inference chat template should:

- Use Gemma-compatible turn markers.
- Avoid adding English instruction wrappers.
- Avoid duplicating system prompts inside user messages.
- Keep the model-visible format consistent with SFT.
- Be tested in LM Studio after GGUF export.

If the model leaks other languages, the issue may come from:

- Dataset contamination.
- Prompt/template mismatch.
- Tokenizer mismatch.
- Too little Kannada CPT.
- Insufficient Kannada-only SFT behavior examples.
- Decoding temperature set too high.

---

## 13. Decoding Strategy for Testing

For manual testing, decoding should be controlled to reduce randomness and multilingual drift.

Recommended starting settings:

| Setting | Suggested Value |
|---|---:|
| Temperature | 0.2–0.5 |
| Top-p | 0.8–0.95 |
| Top-k | 40–100 |
| Repetition penalty | 1.05–1.15 |
| Max new tokens | Task-dependent |

For benchmark evaluation, use deterministic or near-deterministic decoding where possible.

Recommended benchmark setting:

```text
temperature = 0.0 or very low
do_sample = false where supported
```

This makes metrics more reproducible.

---

## 14. Failure Modes Observed

The project observed several common failure modes during training and inference.

### 14.1 Multilingual Leakage

The model sometimes produced Hindi, English, Tamil, Telugu, Chinese, or mixed-script output.

Possible causes:

- Base model multilingual priors.
- Inconsistent SFT formatting.
- Prompt template mismatch.
- Insufficient Kannada-only behavior tuning.
- Tokenizer or merge mismatch.
- High decoding randomness.

Mitigation:

- Use Kannada-only system prompt.
- Use Kannada labels in SFT.
- Filter SFT rows by Kannada ratio.
- Add small corrective Kannada-only examples.
- Keep tokenizer consistent.
- Use low-temperature decoding.

### 14.2 Template Leakage

The model may repeat prompt labels or template tokens.

Possible causes:

- Incorrect chat template.
- Loss applied to user/template tokens.
- Training examples containing raw metadata.

Mitigation:

- Use assistant-only loss masking.
- Run label sanity checks.
- Keep model-visible text clean.
- Test generated outputs after every major checkpoint.

### 14.3 GGUF Conversion Issues

Conversion issues were observed when exporting merged checkpoints.

Known errors:

```text
No canonical BPE pre-tokenizer
BPE pre-tokenizer was not recognized
unrecognized BPE tokenizer
```

Mitigation:

- Use the original base Gemma tokenizer.
- Keep tokenizer files aligned during merge and export.
- Track the llama.cpp version used for conversion.
- Document conversion steps separately.

### 14.4 Slow Training

Training was sometimes slower than expected.

Possible causes:

- Running on CPU after GPU quota exhaustion.
- No FlashAttention support.
- Too much dynamic padding.
- Sequence packing disabled.
- Small batch size.
- Data preprocessing bottleneck.
- Gradient checkpointing overhead.

Mitigation:

- Verify GPU availability.
- Use sequence packing where stable.
- Use grouped batching by length if supported.
- Use pre-tokenized datasets when practical.
- Use checkpointing carefully.
- Avoid unsupported acceleration libraries.

---

---

## 15. Summary

The training strategy uses a practical staged adaptation pipeline:

```text
Mayura-4B(Gemma 3 4B) instruction model
→ Kannada CPT on KSEEB + IndicCorp v2
→ Kannada SFT on FoundryAILabs K-12 Kannada QA
→ benchmark evaluation
→ merged checkpoint / GGUF export
```

This approach is suitable for limited compute because it avoids training from scratch and uses parameter-efficient fine-tuning. CPT improves Kannada language familiarity, while SFT improves user-facing Kannada question-answering behavior. The result is a Kannada-first model that can be evaluated using IndicQA Kannada, MILU Kannada, IndicGenBench, custom GEPA evaluation, and manual qualitative testing.

