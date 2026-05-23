# Evaluation

This document reports the evaluation results for the Kannada-adapted Gemma 3 4B model.

The evaluated model is referred to here as:

```text
Mayura-4B(Gemma 3 4B fine tuned)
```

Training setup:

```text
google/gemma-3-4b-it
→ Kannada Continued Pretraining (CPT)
→ Kannada Supervised Fine-Tuning (SFT)
→ Kannada-first evaluation
```

The goal of evaluation is to measure whether Kannada-specific post-training improves the model across Kannada question answering, multiple-choice understanding, and open-ended generation tasks.

---

## 1. Evaluation Goals

The evaluation process focuses on five questions:

1. Does the model answer Kannada questions better than generic multilingual baselines?
2. Does the model perform competitively on Kannada multiple-choice understanding tasks?
3. Does the model generate Kannada text with acceptable script consistency?
4. Which tasks improved the most after Kannada CPT + SFT?
5. Which tasks still need additional training or prompt optimization?

The evaluation is not limited to loss values from training. It uses external benchmarks, automatic metrics, and manual qualitative checks.

---

## 2. Benchmarks Used

The project uses the following evaluation benchmarks and test sets:

| Benchmark / Evaluation Set | Task Type | Main Metrics |
|---|---|---|
| IndicQA Kannada | Kannada extractive / generative question answering | Token-F1, Char-F1, Exact Match, Contains Match |
| MILU Kannada | Kannada multiple-choice understanding | Accuracy |
| IndicGenBench | Indic generation and transfer tasks | chrF, Token-F1, Char-F1, Kannada character ratio |
| Kannada GEPA evaluation set | Prompt/system-policy optimization | Custom prompt-optimization score |
| Manual qualitative prompts | Practical Kannada assistant behavior | Human inspection |

---

## 3. Metric Definitions

### 3.1 Token-F1

Token-F1 measures token-level overlap between the predicted answer and the reference answer. It is useful for QA tasks where the model may produce a partially correct answer rather than an exact string match.

### 3.2 Char-F1

Char-F1 measures character-level overlap between prediction and reference. This is especially useful for Kannada because tokenization can be sensitive to morphology, spacing, and script handling.

### 3.3 Exact Match

Exact Match checks whether the model output exactly matches the reference answer. This is strict and often low for generative models, especially when answers can be phrased in multiple valid ways.

### 3.4 Contains Match

Contains Match checks whether the generated output contains the expected answer string or a key reference span. This is more forgiving than Exact Match and useful for generative QA.

### 3.5 Accuracy

Accuracy is used for multiple-choice tasks such as MILU Kannada. The model is counted as correct when it selects the correct option.

### 3.6 chrF

chrF is a character n-gram F-score commonly used for generation and translation-style tasks. It is useful for evaluating Kannada generation because it captures character-level similarity between predicted and reference text.

### 3.7 Kannada Character Ratio

Kannada character ratio measures the percentage of output characters that are Kannada-script characters. This is used as a script-consistency signal.

---

## 4. IndicQA Kannada

### 4.1 Task Description

IndicQA Kannada evaluates question answering in Kannada. The model is given a Kannada question and is expected to produce the correct answer.

This project evaluates the Kannada Mayura-4B(Gemma 3 4B) model as a generative QA model and compares the output against reference answers using overlap-style metrics.

### 4.2 Reported Result

| Model / System | Size / Setting | Kannada IndicQA F1 |
|---|---|---:|
| Bloom | Paper baseline | 15.63 direct / 17.86 translate-test |
| Gemma | 7B | 24.47 direct / 16.31 translate-test |
| Llama-3 | 8B | 20.39 direct / 25.84 translate-test |
| Llama-3.1 | 8B | 21.31 direct / 26.22 translate-test |
| Gemma-2 | Larger Gemma-2-class baseline | 35.31 direct / 37.98 translate-test |
| **Mayura-4B** | **4B, CPT + SFT** | **30.86 Token-F1** |

### 4.3 Interpretation

The Kannada Gemma 3 4B model reaches **30.86 Token-F1** on IndicQA Kannada.

This is a strong result for the project because the 4B Kannada-adapted model outperforms the reported Kannada F1 scores of Bloom, Gemma 7B, Llama-3 8B, and Llama-3.1 8B in the comparison table.

The model still trails the stronger Gemma-2-class baseline. This should be interpreted with model scale in mind because the stronger Gemma-2 comparison comes from a larger model family. The result suggests that Kannada CPT + SFT substantially improved Kannada QA behavior beyond generic multilingual ability.

### 4.4 Internal Full-Run Metrics

The full IndicQA Kannada run also produced the following internal metrics:

| Metric | Value |
|---|---:|
| Rows | 1,138 |
| Exact Match | 0.1221 |
| Contains Match | 0.4271 |
| Token-F1 | 0.3086 |
| Char-F1 | 0.4990 |

These metrics show that strict exact match is low, which is expected for generative Kannada QA. The higher Contains Match and Char-F1 indicate that the model frequently includes partially correct answer spans even when its wording differs from the reference.

---

## 5. MILU Kannada

### 5.1 Task Description

MILU Kannada evaluates multitask Indic-language understanding through multiple-choice questions. It is useful for measuring Kannada reasoning, subject knowledge, and option-selection behavior.

The main metric is accuracy.

### 5.2 General Model Comparison

| Model | Size / Type | Kannada MILU Accuracy |
|---|---|---:|
| Gemma-2-2B | 2B base | 29.77 |
| Gemma-2-2B-IT | 2B instruct | 28.52 |
| Llama-3.2-3B | 3B base | 31.75 |
| Llama-3.2-3B-Instruct | 3B instruct | 31.83 |
| Nemotron-4-Mini-Hindi | 4B Hindi-focused | 27.40 |
| Pangea | 7B | 33.83 |
| Aya-23 | 8B | 27.53 |
| Llama-3.1 | 8B | 37.09 |
| Llama-3.1-Instruct | 8B instruct | 36.04 |
| Navarasa-2.0 | 7B Indic model | 40.01 |
| **Mayura-4B** | **4B, CPT + SFT** | **40.55** |
| Gemma-2-9B-IT | 9B instruct | 43.47 |
| Gemma-2-9B | 9B base | 53.99 |
| Gemma-2-27B-IT | 27B instruct | 58.23 |
| Gemma-2-27B | 27B base | 63.43 |

### 5.3 Interpretation

MILU Kannada is the strongest model-comparison result for the project so far.

The Mayura-4B model reaches **40.55% accuracy**, outperforming several published open models in the comparison table, including:

- Gemma-2-2B
- Gemma-2-2B-IT
- Llama-3.2-3B
- Llama-3.2-3B-Instruct
- Nemotron-4-Mini-Hindi-4B
- Pangea-7B
- Aya-23-8B
- Llama-3.1-8B
- Llama-3.1-8B-Instruct

It also slightly outperforms Navarasa-2.0 7B. The model remains below Gemma-2 9B and 27B models, which is expected because those models are significantly larger.

### 5.4 Kannada-Specific Model Comparison

| Kannada-Specific Model | Size | MILU Kannada Accuracy |
|---|---:|---:|
| Kan-LLaMA-7B-base | 7B | 28.67 |
| Ambari-7B-base | 7B | 28.84 |
| Ambari-7B-Instruct | 7B | 26.83 |
| **Mayura-4B** | **4B** | **40.55** |

### 5.5 Kannada-Specific Interpretation

This is one of the most important results for the project.

The current 4B model significantly outperforms the listed Kannada-specific 7B baselines on MILU Kannada. This suggests that the Kannada CPT + SFT pipeline was more effective for this benchmark than several prior Kannada-specific 7B adaptations.

---

## 6. IndicGenBench

### 6.1 Task Description

IndicGenBench evaluates Indic-language generation and cross-lingual transfer behavior. For this project, the relevant Kannada tasks include translation-style generation, QA transfer, and summarization.

The reported tasks are:

- FLORES-IN en→kn
- XQuAD-IN Kannada
- CrossSum-IN Kannada

### 6.2 Results

| Task | Current Mayura-4B Result | Paper-Reported Comparison | Interpretation |
|---|---|---|---|
| **FLORES-IN en→kn** | **31.53 chrF**, 64.78 Char-F1, 72.26% Kannada-char ratio | IndicGenBench reports Kannada en→kn chrF of 11.1 for LLaMA-65B, 15.6 for Gemma-7B-IT, 29.9 for GPT-4, and 58.0 for PaLM-2-L. | Strong Kannada generation result; above Gemma-7B-IT and slightly above GPT-4 in the paper's Kannada FLORES table, but not a strict leaderboard claim due to possible setup differences. |
| **XQuAD-IN Kannada** | **22.54 Token-F1**, 54.40 Char-F1, 40.75 chrF, 63.94% Kannada-char ratio | IndicGenBench reports one-shot XQuAD-IN Kannada scores such as 41.1 for Gemma-7B-IT, 52.0 for BLOOMZ-7B, 55.1 for GPT-4, and 74.4 for PaLM-2-L. | Okay/moderate performance. The model shows partial answer overlap but is not yet strong against large XQuAD baselines. |
| **CrossSum-IN Kannada** | **9.79 chrF**, 23.88 Char-F1, 37.57% Kannada-char ratio | IndicGenBench reports CrossSum-IN Kannada chrF of 13.6 for Gemma-7B-IT, 15.1 for GPT-4, and 25.4 for PaLM-2-L. | Weakest result; cross-lingual summarization needs task-specific training. |

### 6.3 IndicGenBench Interpretation

The FLORES-IN en→kn result is promising. The model reaches **31.53 chrF**, which is higher than the listed Gemma-7B-IT comparison and slightly above the listed GPT-4 Kannada FLORES value in the project comparison table.

This should not be treated as a strict leaderboard claim unless the evaluation setup is confirmed to exactly match the paper setup. However, it is still a useful signal that Kannada CPT + SFT improved Kannada generation quality.

The XQuAD-IN Kannada result is moderate. The model shows partial answer overlap but is still behind the larger reported baselines.

The weakest result is CrossSum-IN Kannada. The low chrF and Kannada-character ratio suggest that cross-lingual summarization requires additional task-specific SFT data, better prompting, or bilingual summarization examples.

---

## 7. GEPA Prompt Optimization Evaluation

### 7.1 Purpose

GEPA is used as a post-training prompt/system-policy optimization step. It is not model-weight training. Instead, it searches for better system prompts and task instructions using feedback from model trajectories.

The goal is to improve:

- Kannada-only response consistency.
- Benchmark-specific answer formatting.
- MCQ option-following behavior.
- Conciseness and directness.
- Refusal behavior when the answer is unknown.

### 7.2 GEPA Dataset

Known setup:

```text
gepa_dataset_kn.jsonl
```

Observed size:

```text
195 rows
```

This set is used for prompt optimization and behavior testing, not as a replacement for full benchmark evaluation.

---

## 8. Manual Qualitative Evaluation

Manual testing is used because automatic metrics do not fully capture whether the model behaves like a usable Kannada assistant.

Manual checks focus on:

- Kannada-only output.
- Avoiding English, Hindi, Tamil, Telugu, Chinese, or mixed-script leakage.
- Concise school-level explanations.
- MCQ answer formatting.
- Whether the model admits uncertainty when it does not know the answer.
- Whether the LM Studio prompt template produces stable Kannada responses.
- Whether the model avoids repeating prompt labels or chat-template artifacts.

---

## 9. Overall Findings

The current Mayura-4B model shows measurable improvement in Kannada-specific QA, MCQ reasoning, and Kannada generation.

The strongest evidence comes from:

1. **IndicQA Kannada**, where the model reaches **30.86 Token-F1** and outperforms several reported multilingual 7B/8B baselines.
2. **MILU Kannada**, where the model reaches **40.55% accuracy**, outperforming several published open multilingual and Kannada-specific baselines despite being only 4B parameters.
3. **FLORES-IN en→kn**, where the model shows strong Kannada generation quality with **31.53 chrF** and **72.26% Kannada-character ratio**.

The clearest limitation is summarization. The **CrossSum-IN Kannada** result remains weak and likely requires task-specific summarization training, stronger prompting, and possibly bilingual summarization examples.

---

## 10. Limitations

The current evaluation has the following limitations:

- Some comparisons are based on paper-reported baselines and may not use identical prompts, decoding settings, or evaluation harnesses.
- FLORES-IN comparisons should not be treated as strict leaderboard claims unless the setup is exactly reproduced.
- Exact decoding settings should be documented for each benchmark.
- GEPA is still an optimization process and should be reported separately from model-weight improvements.
- Manual evaluation is subjective and should be supplemented with saved prediction files.
- Kannada character ratio measures script consistency, not factual correctness.
- Strong benchmark performance does not guarantee safe or reliable behavior in all real-world use cases.

---

## 12. Current Result Summary

| Benchmark | Task | Main Metric | Mayura-4B Result |
|---|---|---:|---:|
| IndicQA Kannada | QA | Token-F1 | 30.86 |
| IndicQA Kannada | QA | Char-F1 | 49.90 |
| IndicQA Kannada | QA | Contains Match | 42.71 |
| MILU Kannada | MCQ understanding | Accuracy | 40.55 |
| IndicGenBench FLORES-IN en→kn | Generation / translation transfer | chrF | 31.53 |
| IndicGenBench FLORES-IN en→kn | Script consistency | Kannada-char ratio | 72.26% |
| IndicGenBench XQuAD-IN Kannada | QA transfer | Token-F1 | 22.54 |
| IndicGenBench XQuAD-IN Kannada | QA transfer | Char-F1 | 54.40 |
| IndicGenBench CrossSum-IN Kannada | Summarization transfer | chrF | 9.79 |
| IndicGenBench CrossSum-IN Kannada | Script consistency | Kannada-char ratio | 37.57% |

---

## 13. Next Evaluation Steps

Recommended next steps:

1. Save raw predictions for all benchmarks.
2. Document exact prompts and decoding settings.
3. Re-run MILU Kannada with deterministic decoding.
4. Re-run IndicGenBench with the finalized GEPA-optimized system prompt.
5. Add a Kannada script-ratio report across all benchmark generations.
6. Build a held-out KSEEB/PUC Kannada QA test set.
7. Add summarization-specific evaluation after any future summarization SFT.
8. Compare the merged HF checkpoint and GGUF checkpoint to ensure quantization does not significantly degrade Kannada performance.

---

## 14. Conclusion

The evaluation results show that Kannada-specific CPT followed by SFT can make a compact 4B model competitive on Kannada tasks.

The strongest result is MILU Kannada, where the model reaches **40.55% accuracy** and outperforms multiple larger or Kannada-specific baselines in the project comparison. IndicQA Kannada also shows strong QA improvement with **30.86 Token-F1**. IndicGenBench shows that generation quality improved most clearly for FLORES-IN en→kn, while summarization remains the weakest area.

Overall, the model is a promising Kannada-first adaptation of Gemma 3 4B, with clear strengths in Kannada QA and MCQ-style understanding, and clear next steps around summarization, prompt optimization, and stricter reproducibility reporting.

