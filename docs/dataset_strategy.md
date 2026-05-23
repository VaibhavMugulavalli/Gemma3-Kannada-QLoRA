# Dataset Strategy

This document explains the dataset design used for the Kannada-first Gemma 3 4B fine-tuning project. The goal of the dataset strategy was to adapt a general-purpose pretrained model into a Kannada-centric model through a two-stage post-training pipeline:

1. **Continued Pretraining (CPT)** on Kannada-only raw text.
2. **Supervised Fine-Tuning (SFT)** on Kannada question-answer and instruction-style data.

The strategy intentionally separates raw language adaptation from instruction-following adaptation. CPT is used to improve Kannada fluency, script consistency, vocabulary exposure, and domain coverage. SFT is used to improve user-facing answer behavior, question-answering, and Kannada-only response formatting.

---

## 1. Objectives

The dataset pipeline was built around the following objectives:

- Improve Kannada generation quality on a small open model.
- Preserve the base model's general reasoning ability while adapting it to Kannada.
- Increase exposure to native Kannada educational and general-domain text.
- Avoid unnecessary multilingual mixing during Kannada adaptation.
- Keep CPT and SFT datasets separate because they serve different training objectives.
- Build a reproducible dataset pipeline suitable for GitHub documentation and future retraining.

The model was not trained as a translation model or a code-mixed Kannada-English assistant. The intended behavior is Kannada-first generation, with strict Kannada-only output wherever possible.

---

## 2. Training Stages and Dataset Roles

| Stage | Dataset Type | Main Purpose | Format |
|---|---|---|---|
| CPT | Raw Kannada text | Language adaptation, fluency, vocabulary exposure | Plain text / JSONL text blocks |
| SFT | Kannada QA / instruction data | Instruction following and answer alignment | Chat/instruction JSONL |
| Evaluation | Kannada QA and task benchmarks | Measure factual QA, answer overlap, script consistency, and task behavior | Benchmark-specific JSONL/CSV |

---

## 3. Dataset Buckets

The project separates data into two major buckets.

### Bucket A: Continued Pretraining Data

Bucket A contains plain Kannada text for causal language modeling. This bucket is used only for CPT.

Examples:

```json
{"text": "...Kannada paragraph or cleaned textbook block..."}
```

A richer intermediate format may be used before final packing:

```json
{
  "id": "textbook_grade_08_science_chapter_03_block_001",
  "text": "...Kannada text...",
  "source": "textbook",
  "language": "kn",
  "board": "KSEEB",
  "grade": "8",
  "subject": "Science",
  "book_id": "TODO",
  "chapter": "TODO",
  "page": 12
}
```

For training, the final packed file can be simplified to `{"text": ...}` rows. Metadata is still useful during cleaning, auditing, deduplication, and documentation.

### Bucket B: Supervised Fine-Tuning Data

Bucket B contains instruction-style Kannada data used for SFT.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "ಪ್ರಶ್ನೆ: ..."
    },
    {
      "role": "assistant",
      "content": "ಉತ್ತರ: ..."
    }
  ]
}
```

The SFT stage should not be mixed casually with raw CPT text. The model should first learn Kannada distribution through CPT, then learn response behavior through SFT.

---

## 4. CPT Data Sources

### 4.1 KSEEB Kannada Textbook Corpus

The primary curated corpus for CPT was extracted from Karnataka school textbook material.

Final CPT textbook coverage included:

- KSEEB textbook content.
- Grade 3–10 textbook material.
- Grade 11 and Grade 12 material through 1st PUC and 2nd PUC sources.
- Subject-wise and grade-wise metadata derived from filenames.

The textbook corpus was selected because it provides structured, native Kannada educational prose. It is useful for improving Kannada fluency, school-level explanations, and domain coverage across science, social science, language, and general academic content.

Textbook data was treated as high-quality Kannada prose but required substantial filtering because many textbook pages contain exercises, fill-in-the-blanks, activities, diagrams, and answer-line regions that are not ideal for raw language modeling.

### 4.2 IndicCorp v2 Kannada Slice

IndicCorp v2 Kannada text was used as the broader general-domain CPT source.

Purpose:

- Increase Kannada token exposure beyond textbook style.
- Add more varied writing styles.
- Improve general Kannada language modeling.
- Reduce overfitting to school-textbook phrasing.

IndicCorp data was converted into JSONL-style text rows and combined with the textbook corpus for CPT.

---

## 5. CPT Mixing Strategy

The CPT mixture was designed to balance quality and scale.

The intended weighting was approximately:

| Source | Role | Approximate Weight |
|---|---|---:|
| KSEEB textbook corpus | High-quality educational Kannada | 80% |
| IndicCorp v2 Kannada | General-domain Kannada scale | 20% |

This weighting prioritizes curated textbook Kannada while still exposing the model to broader Kannada usage from IndicCorp.

A curriculum-style order was preferred:

1. Start with cleaner textbook content.
2. Add broader IndicCorp v2 Kannada text.
3. Pack into causal-LM chunks.
4. Train with checkpointing and resume support.

The exact implementation may use sampling by rows, documents, or tokens. Token-level weighting is preferred because raw dataset sizes can differ significantly.

**TODO:** Confirm whether the final training run used exact 80:20 weighting by tokens, approximate sampling by file count, or another effective ratio.

---

## 6. Textbook Cleaning and Filtering

Textbook extraction required cleaning because many PDFs contained non-training content such as activity pages, post-chapter questions, dotted answer lines, and OCR artifacts.

The filtering strategy was:

- Keep prose-like chapter content.
- Reject pages dominated by questions, activities, fill-in-the-blanks, or worksheets.
- Reject pages with high dotted-line or dashed-line density.
- Reject or separate sections around post-chapter exercise markers such as `ಅಭ್ಯಾಸ`.
- Preserve chapter-level content where possible.
- Keep rejected question/activity pages separately only if they may later be useful for SFT or evaluation.

The goal was to prevent the CPT dataset from becoming dominated by incomplete question fragments or answer-line templates.

Example rejection patterns:

- Many dotted lines: `..............`
- Many blank answer regions.
- Repeated exercise/question numbering.
- Pages containing mostly short question prompts.
- Pages after explicit exercise sections.

---

## 7. OCR and Text Extraction Considerations

Some textbook PDFs did not provide clean native Unicode text. Issues observed during extraction included:

- Text embedded as images.
- Legacy Kannada font encodings.
- Nudi font artifacts.
- Garbled Unicode sequences.
- Poor native PDF extraction quality.
- OCR noise from scanned pages.

Where native extraction failed, OCR-based extraction was explored. The cleaned dataset should prefer reliable Unicode Kannada text over noisy OCR output.

Recommended quality checks:

- Kannada Unicode character ratio.
- Minimum text length per block.
- Repetition detection.
- Removal of headers, footers, page numbers, and copyright noise.
- Manual spot checks per source type.
- Deduplication across repeated textbook sections.

---

## 8. SFT Data Sources

### 8.1 FoundryAILabs K-12 Indian Curriculum Dataset

The SFT dataset used for instruction tuning was:

```text
FoundryAILabs/k12-indian-curriculum-4.9m
```

This dataset is described as a multilingual K-12 curriculum question-answer dataset covering CBSE/NCERT-style content across 12 Indian languages. The Kannada subset contains approximately 408K entries.

Known dataset details:

- Dataset ID: `FoundryAILabs/k12-indian-curriculum-4.9m`
- Task type: question answering and text generation.
- Format: JSON.
- License: Apache-2.0.
- Full dataset size: approximately 4.9M question-answer pairs.
- Kannada subset size: approximately 408K rows.
- Relevant fields:
  - `instruction`
  - `response`
  - `language`
  - `subject`
  - `class`

The Kannada slice was selected because it directly matches the model's intended use case: Kannada educational question answering and school-level assistance.

### 8.2 Datasets Not Used in Final SFT

AI4Bharat IndicQuestionGeneration Kannada was explored earlier as a possible Kannada QA-style dataset, but it was not used in the final SFT run.

Earlier observed usage:

```python
load_dataset("ai4bharat/IndicQuestionGeneration", "kn")
```

It remains a possible future dataset for context-based QA, question generation, or reading-comprehension-style adaptation, but it should not be listed as part of the final used SFT data for this repo version.

---

## 9. SFT Formatting Strategy

SFT examples were formatted to match the target chat behavior of the Gemma-family model.

A Kannada-native prompt style was preferred over English labels such as `question`, `answer`, or `prompt`.

Preferred Kannada labels:

- `ಪ್ರಶ್ನೆ:` for question.
- `ಉತ್ತರ:` for answer.
- Kannada-only system instruction where applicable.

Example training format:

```text
<start_of_turn>user
ಪ್ರಶ್ನೆ: ...
<end_of_turn>
<start_of_turn>model
...
<end_of_turn>
```

The model should be trained to output only the assistant answer, not to repeat the full prompt template.

Important formatting rules:

- Avoid English field labels inside the model-visible prompt.
- Avoid mixed-language boilerplate unless intentionally training refusal or language-control behavior.
- Keep the answer in Kannada wherever possible.
- Mask prompt tokens during SFT if using a trainer setup that supports assistant-only loss.
- Run label sanity checks before training.

---

## 10. Kannada-Only Output Strategy

The project target is Kannada-first behavior. Therefore, the dataset pipeline should minimize unwanted multilingual drift.

Recommended controls:

- Use only Kannada slice data for SFT.
- Avoid mixing Hindi, Telugu, Tamil, English, or code-mixed data unless explicitly planned.
- Filter rows with low Kannada character ratio.
- Remove examples where the answer is mostly English.
- Replace English template labels with Kannada equivalents.
- Use a Kannada-only system prompt during inference and evaluation.
- Add small amounts of corrective SFT data if the model repeatedly leaks English or other scripts.

Example corrective behavior:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain photosynthesis in English."
    },
    {
      "role": "assistant",
      "content": "ಕ್ಷಮಿಸಿ, ನಾನು ಕನ್ನಡದಲ್ಲೇ ಉತ್ತರಿಸುತ್ತೇನೆ. ಪ್ರಕಾಶ ಸಂಶ್ಲೇಷಣೆ ಎಂದರೆ ಸಸ್ಯಗಳು ಸೂರ್ಯನ ಬೆಳಕಿನ ಸಹಾಯದಿಂದ ಆಹಾರ ತಯಾರಿಸುವ ಪ್ರಕ್ರಿಯೆ."
    }
  ]
}
```

Use such examples carefully so that the model does not over-refuse normal questions.

---

## 11. Tokenizer Strategy

The project explored Kannada tokenizer extension, but the trained extended tokenizer was not used in the final stable pipeline due to merge and conversion issues.

Final practical strategy:

- Use the base Gemma tokenizer consistently across CPT, SFT, merging, and inference.
- Do not change tokenizer files between CPT and SFT.
- Keep tokenizer files aligned with the base model during GGUF conversion.
- Avoid training adapters with one tokenizer and converting/inferencing with another.

Reason:

Tokenizer mismatch can cause degraded Kannada output, broken generation, or GGUF conversion failures. Keeping the tokenizer stable improves reproducibility, especially when exporting merged checkpoints and GGUF files.

Known issue encountered:

- `llama.cpp` GGUF conversion produced errors such as unrecognized BPE tokenizer / pre-tokenizer hash issues.

This should be documented separately in a conversion or deployment guide.

---

## 12. Train/Validation Split Strategy

Recommended split policy:

- Split by document/source where possible, not by random chunks only.
- Avoid leakage between train and validation from the same chapter or repeated document section.
- Keep validation data small but representative for CPT loss tracking.
- Use separate held-out Kannada QA benchmarks for real evaluation.

Suggested splits:

| Dataset | Suggested Split |
|---|---:|
| CPT raw text | 98–99% train, 1–2% validation |
| SFT QA data | 95–98% train, 2–5% validation |
| Evaluation benchmarks | Fully held out |

**TODO:** Confirm the exact split used in the final CPT and SFT runs.

---

## 13. Evaluation Dataset Strategy

Evaluation should not rely only on training loss. The model should be tested on held-out Kannada tasks that measure answer correctness, answer overlap, task-following, multitask understanding, generation quality, and Kannada-only behavior.

### 13.1 Evaluation Datasets and Benchmarks Used

The evaluation workflow used the following Kannada-focused evaluation sets and benchmarks:

| Evaluation Dataset / Benchmark | Used For | Main Metric / Signal | Notes |
|---|---|---|---|
| IndicQA Kannada | Extractive / generative Kannada QA evaluation | Exact Match, Contains Match, Token-F1, Char-F1 | Used for full Kannada QA benchmarking with 1,138 rows. |
| MILU Kannada | Multi-task Indic language understanding | Accuracy / option-match accuracy | Kannada subset used to test subject-level MCQ reasoning and India-centric knowledge. |
| IndicGenBench | Indic text-generation evaluation | Task-specific generation metrics | Used to evaluate open-ended generation behavior across Indic generation tasks. |
| Kannada GEPA evaluation set | Prompt optimization and response-quality testing | Custom QA score / prompt-optimization score | Smaller curated set used during GEPA prompt optimization. |
| Manual Kannada qualitative prompts | Fluency, script consistency, refusal behavior, and practical answer testing | Human inspection | Used to catch issues not visible from automatic metrics. |

### 13.2 IndicQA Kannada

IndicQA Kannada was used as the main benchmark-style QA evaluation dataset.

The evaluation treated the model as a generative QA model and compared model outputs against reference answers using overlap-style metrics.

Observed full-run result:

| Metric | Value |
|---|---:|
| Rows | 1,138 |
| Exact Match | 0.1221 |
| Contains Match | 0.4271 |
| Token-F1 | 0.3086 |
| Char-F1 | 0.4990 |

Interpretation:

- Exact Match is expected to be low for generative QA because Kannada answers may be phrased differently from the reference answer.
- Contains Match is higher, suggesting that the model often includes at least part of the expected answer.
- Token-F1 around 0.31 shows useful partial answer overlap, but also indicates room for improvement in benchmark-style factual precision.
- Char-F1 around 0.50 suggests that character-level overlap is meaningfully better than strict token-level matching, which is useful for Kannada where tokenization can be sensitive.

### 13.3 MILU Kannada

MILU Kannada was used as a multitask understanding benchmark.

MILU stands for Multi-task Indic Language Understanding. It is useful because it evaluates more than simple answer overlap. The Kannada subset tests the model on multiple-choice questions across different domains and subjects, including India-centric and culturally relevant knowledge.

Evaluation signal:

- Primary metric: accuracy / option-match accuracy.
- Task style: multiple-choice question answering.
- Kannada subset: used for Kannada-specific multitask understanding.
- Purpose: measure whether the model can select the correct answer option under a structured MCQ setup.

Recommended reporting format:

| Benchmark | Language | Setting | Metric | Score |
|---|---|---|---|---:|
| MILU | Kannada | TODO: zero-shot / few-shot | Accuracy | TODO |

### 13.4 IndicGenBench

IndicGenBench was used as a generation-quality benchmark for Indic language tasks.

It is useful because the project is not only about MCQ accuracy or extractive QA. The model also needs to generate usable Kannada responses. IndicGenBench helps evaluate user-facing generation tasks such as question answering, summarization, and translation-style generation.

Evaluation signal:

- Primary metric: task-specific generation quality.
- Useful checks: answer relevance, generation fluency, semantic overlap, and Kannada output consistency.
- Purpose: evaluate whether the model can produce meaningful Kannada text in open-ended generation settings.

Recommended reporting format:

| Benchmark | Language | Task | Metric | Score |
|---|---|---|---|---:|
| IndicGenBench | Kannada | TODO | TODO | TODO |

### 13.5 GEPA Kannada Evaluation Set

A smaller Kannada QA-style evaluation set was used during GEPA prompt optimization.

Known setup:

- Dataset file used during GEPA experiments: `gepa_dataset_kn.jsonl`.
- Observed row count: 195 rows.
- Used for prompt optimization rather than final standalone benchmark reporting.
- Useful for evaluating Kannada-only compliance, short-answer behavior, and answer format control.

This set should be documented as an internal/custom evaluation set unless it is later published with the repo.

### 13.6 Manual Qualitative Evaluation

Manual evaluation was also used because automatic metrics do not fully capture Kannada fluency, script purity, and practical answer quality.

Manual checks focused on:

- Whether the model replies only in Kannada.
- Whether the model leaks English, Hindi, Tamil, Telugu, Chinese, or other scripts.
- Whether the model gives concise school-level answers.
- Whether the model follows MCQ instructions.
- Whether the model avoids hallucinating when it does not know the answer.
- Whether the LM Studio chat template produces stable Kannada responses.

### 13.7 Future Evaluation Candidates

The following datasets or benchmark categories can be added later if they are used in future runs:

| Candidate | Why It May Help |
|---|---|
| IndicMMLU / IndicMMLU-Pro Kannada subsets | Broader subject-wise MCQ-style evaluation. |
| L3Cube IndicQuest Kannada | General Indian-context QA evaluation. |
| Custom KSEEB/PUC held-out QA set | Stronger alignment with the model's training domain. |
| Kannada script-ratio benchmark | Measures language leakage and script consistency. |

---

## 14. Data Quality Checks

Before training, each dataset stage should pass quality checks.

### CPT Checks

- Kannada Unicode ratio.
- Empty or very short text removal.
- Duplicate block removal.
- Repeated punctuation removal.
- Header/footer cleanup.
- OCR artifact detection.
- Exercise/activity page filtering.
- Token count estimation.

### SFT Checks

- Non-empty instruction and response.
- Kannada-majority response.
- Prompt/answer separation.
- No accidental English field labels in model-visible text.
- No malformed chat templates.
- No answer leakage in the user prompt unless intentionally doing context-answer-question generation.
- Label masking sanity check.

### Evaluation Checks

- Ensure evaluation rows are not copied from SFT training data.
- Preserve answer/reference fields.
- Track Kannada script ratio of model outputs.
- Store raw predictions for manual inspection.

---

## 15. Known Limitations

Current known limitations of the dataset pipeline:

- Total Kannada token exposure is still much smaller than large-scale language adaptation runs.
- Some textbook PDFs require OCR or legacy-font cleanup.
- Extended tokenizer experiments were not used in the final stable pipeline due to merge/conversion issues.
- Kannada QA data availability is limited compared to English.
- The model may still leak English or other scripts under some prompts.
- CPT improves language familiarity, but factual QA behavior depends heavily on SFT quality.
- MCQ-style benchmark behavior may require targeted SFT or evaluation-specific prompt calibration.
- GGUF conversion may require tokenizer compatibility fixes in `llama.cpp`.

---

## 16. Recommended Repository Documentation

The repository should include:

| File | Purpose |
|---|---|
| `dataset_strategy.md` | Explains this strategy and source choices |
| `training_strategy.md` | Explains CPT → SFT training setup |
| `evaluation.md` | Reports IndicQA and other benchmark results |
| `model_card.md` | Model details, intended use, limitations, and benchmark results |
---


## 17. Summary

The dataset strategy uses a staged Kannada adaptation pipeline:

- **CPT:** KSEEB textbook Kannada + IndicCorp v2 Kannada for language adaptation.
- **SFT:** Kannada K-12 QA/instruction data for answer behavior.
- **Evaluation:** Kannada QA and benchmark-style tasks to measure factuality, answer overlap, and Kannada-only behavior.

This separation keeps the training objective clean: first adapt the model to Kannada distribution, then teach it how to answer Kannada questions. The approach is practical for a small open model and suitable for iterative improvement as more Kannada-native datasets become available.

