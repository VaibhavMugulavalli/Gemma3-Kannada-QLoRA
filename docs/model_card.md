# Model Card: Mayura-4B

## Model Summary

**Mayura-4B** is a Kannada-first language model adapted from `google/gemma-3-4b-pt`. The model was developed to improve Kannada question answering, instruction following, and generation on a compact open model.

The model was trained using a two-stage post-training pipeline:

```text
google/gemma-3-4b-it
→ Kannada Continued Pretraining (CPT)
→ Kannada Supervised Fine-Tuning (SFT)
→ Evaluation and GGUF export

The project focuses on Kannada-centric adaptation rather than bilingual translation-style optimization or Kannada-English code-mixed tuning. The intended behavior is to answer in Kannada as consistently as possible, especially for educational and school-level question-answering tasks.

---

## Model Details

| Field | Value |
|---|---|
| Model name | Mayura-4B |
| Base model | `google/gemma-3-4b-pt` |
| Model size | 4B parameters |
| Primary language | Kannada (`kn`) |
| Architecture family | Gemma 3 |
| Training type | Continued Pretraining + Supervised Fine-Tuning |
| Fine-tuning method | TODO: LoRA / QLoRA |
| Release format | TODO: Hugging Face checkpoint / adapter / merged model / GGUF |
| Quantized release | TODO: GGUF quantization type, if released |
| License | TODO |
| Developer | TODO |

---

## Intended Use

This model is intended for Kannada-first language generation and educational QA use cases.

Suitable use cases include:

- Kannada question answering.
- School-level Kannada educational assistance.
- Kannada explanations of textbook-style concepts.
- Kannada instruction following.
- Kannada text generation.
- Research on low-resource Indic language adaptation.
- Local/offline experimentation with Kannada LLMs.

The model is especially intended for experimentation with compact Kannada-first LLMs rather than high-stakes production deployment.

---

## Out-of-Scope Use

This model is not intended for:

- Medical, legal, financial, or safety-critical decision-making.
- Factual answers without verification from reliable sources.
- Guaranteed Kannada translation quality.
- Production use without additional safety, bias, and factuality testing.
- Tasks requiring perfect multilingual translation or code-mixed Kannada-English handling.
- Sensitive personal-data processing without proper privacy controls.

The model may hallucinate, produce incorrect answers, or occasionally switch languages. Outputs should be reviewed before being used in important contexts.

---

## Training Data

The model uses a two-stage dataset strategy.

### Stage 1: Continued Pretraining Data

CPT was performed on Kannada-only text sources.

Primary CPT sources:

| Source | Purpose |
|---|---|
| KSEEB textbook corpus, Grade 3 to Grade 12 / 2nd PUC | High-quality educational Kannada text |
| IndicCorp v2 Kannada slice | Broader general-domain Kannada exposure |

The intended CPT mix was approximately:

| Source | Intended Weight |
|---|---:|
| KSEEB / textbook Kannada | 80% |
| IndicCorp v2 Kannada | 20% |

**TODO:** Confirm whether the final run used exact token-level 80:20 mixing or approximate sampling.

### Stage 2: Supervised Fine-Tuning Data

SFT was performed using the Kannada slice of:

```text
FoundryAILabs/k12-indian-curriculum-4.9m
```

This dataset provides K-12 question-answer and instruction-style examples across Indian languages. For this model, only the Kannada portion was used.

Approximate Kannada subset size:

```text
~408K rows
```

AI4Bharat IndicQuestionGeneration was explored earlier but was not used in the final SFT run.

---

## Data Processing

### CPT Processing

CPT data was cleaned and prepared with emphasis on Kannada text quality.

Processing included:

- Kannada Unicode normalization.
- Removal of empty or very short rows.
- Filtering of noisy OCR output.
- Filtering of textbook exercises, activity pages, and dotted answer-line regions.
- Kannada character-ratio checks.
- Basic deduplication where possible.
- Token count estimation.
- Train/validation splitting.

### SFT Processing

SFT data was converted into a Gemma-compatible chat/instruction format.

Model-visible prompt style used Kannada labels such as:

```text
ಪ್ರಶ್ನೆ:
ಉತ್ತರ:
```

Important formatting choices:

- Kannada-native prompt labels were preferred over English field names.
- Responses were kept Kannada-first wherever possible.
- Malformed or empty examples were removed.
- Assistant-only loss masking was preferred so that the model learned to generate the answer rather than reproduce the full prompt.

---

## Training Procedure

The model was adapted in two main stages.

### 1. Continued Pretraining

CPT was used to improve Kannada language modeling ability. This stage targeted Kannada fluency, vocabulary exposure, script consistency, and familiarity with educational Kannada.

### 2. Supervised Fine-Tuning

SFT was used to improve question-answering and instruction-following behavior. This stage trained the model to provide more direct Kannada responses for educational and QA-style prompts.


## Hardware and Environment

Training and testing were performed across constrained GPU environments.

Known environments used during the project:

| Environment | Role |
|---|---|
| College GPU notebooks / Kubeflow | Main training experiments |
| A100 MIG 3g.20gb | Higher-memory training runs |
| Kaggle GPUs | Verification and constrained training experiments |
| Google Colab | Evaluation, conversion tests, and debugging |
| Local machine / LM Studio | GGUF inference testing |

Known constraints:

- CUDA 11.8 limitation on some college GPUs.
- FlashAttention was not consistently available because of CUDA/driver constraints.
- Kaggle and Colab runtime limits required checkpoint-resume support.
- Some merge and conversion steps were constrained by CPU RAM and disk availability.

---

## Evaluation

The model was evaluated on Kannada QA, multiple-choice understanding, and Indic generation benchmarks.

### Evaluation Summary

| Benchmark | Task | Metric | Result |
|---|---|---:|---:|
| IndicQA Kannada | QA | Token-F1 | 30.86 |
| IndicQA Kannada | QA | Char-F1 | 49.90 |
| IndicQA Kannada | QA | Contains Match | 42.71 |
| IndicQA Kannada | QA | Exact Match | 12.21 |
| MILU Kannada | MCQ understanding | Accuracy | 40.55 |
| IndicGenBench FLORES-IN en→kn | Generation / translation transfer | chrF | 31.53 |
| IndicGenBench FLORES-IN en→kn | Script consistency | Kannada-char ratio | 72.26% |
| IndicGenBench XQuAD-IN Kannada | QA transfer | Token-F1 | 22.54 |
| IndicGenBench XQuAD-IN Kannada | QA transfer | Char-F1 | 54.40 |
| IndicGenBench XQuAD-IN Kannada | QA transfer | chrF | 40.75 |
| IndicGenBench XQuAD-IN Kannada | Script consistency | Kannada-char ratio | 63.94% |
| IndicGenBench CrossSum-IN Kannada | Summarization transfer | chrF | 9.79 |
| IndicGenBench CrossSum-IN Kannada | Summarization transfer | Char-F1 | 23.88 |
| IndicGenBench CrossSum-IN Kannada | Script consistency | Kannada-char ratio | 37.57% |

### IndicQA Kannada

The model achieved **30.86 Token-F1** on IndicQA Kannada. This outperformed several reported multilingual baselines in the project comparison, including Bloom, Gemma 7B, Llama-3 8B, and Llama-3.1 8B. It remained below the stronger Gemma-2-class baseline.

Internal full-run metrics:

| Metric | Value |
|---|---:|
| Rows | 1,138 |
| Exact Match | 0.1221 |
| Contains Match | 0.4271 |
| Token-F1 | 0.3086 |
| Char-F1 | 0.4990 |

### MILU Kannada

The model achieved **40.55% accuracy** on MILU Kannada.

This was one of the strongest results in the project. The model outperformed several listed open and Kannada-specific baselines despite being only a 4B model. It also slightly exceeded Navarasa-2.0 7B in the project comparison.

### IndicGenBench

IndicGenBench results show mixed performance.

Strongest result:

- FLORES-IN en→kn: **31.53 chrF**, **72.26% Kannada-character ratio**.

Moderate result:

- XQuAD-IN Kannada: **22.54 Token-F1**, **54.40 Char-F1**.

Weakest result:

- CrossSum-IN Kannada: **9.79 chrF**, **37.57% Kannada-character ratio**.

This suggests that the model improved Kannada generation and QA transfer but still requires task-specific training for cross-lingual summarization.

---

## Qualitative Evaluation

Manual qualitative testing focused on:

- Kannada-only response consistency.
- Avoiding English, Hindi, Tamil, Telugu, Chinese, or mixed-script leakage.
- Directness and clarity of answers.
- School-level explanation quality.
- MCQ answer formatting.
- Whether the model admits uncertainty when it does not know the answer.
- LM Studio prompt-template compatibility.

Manual testing showed that prompt format and system instruction strongly affect the model's Kannada-only behavior.

---

## Prompting

Recommended system prompt:

```text
ನೀನು ಕೇವಲ ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸುವ ಅತ್ಯಂತ ನಿಖರ ಮತ್ತು ಸ್ಪಷ್ಟವಾದ AI ಸಹಾಯಕ.
ಬಳಕೆದಾರರ ಪ್ರಶ್ನೆಗಳಿಗೆ ನೇರವಾಗಿ, ಸಂಕ್ಷಿಪ್ತವಾಗಿ ಮತ್ತು ಶಾಲಾ ಮಟ್ಟದ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸು.
ನಿನ್ನ ಉತ್ತರವು ಕೇವಲ ಕನ್ನಡದಲ್ಲೇ ಇರಬೇಕು. ಇಂಗ್ಲಿಷ್, ಹಿಂದಿ, ತಮಿಳು, ತೆಲುಗು, ಮಲಯಾಳಂ, ಗುಜರಾತಿ, ಬೆಂಗಾಲಿ, ಚೈನೀಸ್ ಅಥವಾ ಮಿಶ್ರ-ಲಿಪಿ ಬಳಕೆಯನ್ನು ತಪ್ಪಿಸು.
ಪ್ರಶ್ನೆಗೆ ಸರಿಯಾದ ಉತ್ತರ ತಿಳಿದಿಲ್ಲದಿದ್ದರೆ, "ಕ್ಷಮಿಸಿ, ನನಗೆ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ಗೊತ್ತಿಲ್ಲ" ಎಂದು ಪ್ರಾಮಾಣಿಕವಾಗಿ ತಿಳಿಸು.
```

Recommended decoding for evaluation:

| Setting | Suggested Value |
|---|---:|
| Temperature | 0.0–0.2 |
| Top-p | 0.8–0.95 |
| Top-k | 40–100 |
| Repetition penalty | 1.05–1.15 |
| Max new tokens | Task-dependent |

For benchmark evaluation, deterministic decoding is preferred where supported.

---

## Strengths

The model shows the strongest performance in:

- Kannada QA.
- Kannada MCQ-style understanding.
- Educational Kannada responses.
- Kannada generation compared to some larger generic baselines.
- Compact deployment experiments through GGUF.

The strongest benchmark result is MILU Kannada, where the model reached **40.55% accuracy** and outperformed several listed multilingual and Kannada-specific models.

---

## Limitations

Known limitations:

- The model may still occasionally generate non-Kannada or mixed-script text.
- It may hallucinate facts or give overconfident answers.
- Exact benchmark comparisons may not be directly equivalent unless prompt format, decoding settings, and evaluation scripts match the original papers.
- Cross-lingual summarization performance is weak.
- The model was not optimized for code-mixed Kannada-English usage.
- The model was not trained as a general translation model.
- Evaluation is still ongoing and should be expanded with saved raw predictions.
- The model should not be used for high-stakes decisions without verification.

---

## Bias, Risks, and Safety

Because the model is adapted from a general pretrained LLM and trained on educational and web-scale Kannada sources, it may inherit biases from:

- The base model.
- Public web text in IndicCorp v2.
- Educational curriculum content.
- SFT question-answer distributions.

Potential risks:

- Incorrect or hallucinated answers.
- Overconfident explanations.
- Uneven performance across Kannada dialects, domains, and writing styles.
- Language leakage into English or other Indic languages.
- Poor performance on topics not covered by the training data.

Users should verify important outputs against trusted sources.

