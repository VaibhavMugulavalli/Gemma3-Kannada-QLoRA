# Mayura-4B

**Mayura-4B** is a Kannada-first language-model adaptation project built around Gemma 3 4B. The goal of the project is to test whether a compact open model can be meaningfully improved for Kannada question answering, instruction following, and generation through Kannada-specific post-training.

The repository contains documentation, experiment scripts, evaluation notebooks, and result files for the Kannada CPT + SFT pipeline.

```text
Gemma 3 4B base
        ↓
Kannada Continued Pretraining (CPT)
        ↓
Kannada Supervised Fine-Tuning (SFT)
        ↓
GGUF/local inference testing
        ↓
Kannada benchmark evaluation
```

---

## Why this project exists

Most multilingual LLMs technically support Kannada, but their Kannada behavior can be inconsistent in practice. Common issues include mixed-script output, Kannada-English or Kannada-Hindi leakage, weak Kannada educational QA, and poor performance on Kannada-specific benchmarks.

Mayura-4B explores a focused alternative: adapt a compact open model specifically for Kannada using curated Kannada corpora, instead of relying only on generic multilingual capability.

The model is not designed as a code-mixed Kannada-English model and was not optimized primarily as a translation model. The intended behavior is Kannada-first generation, especially for educational and question-answering use cases.

---

## Current status

| Item | Status |
|---|---|
| Base model | `google/gemma-3-4b-pt` |
| Main training approach | CPT followed by SFT |
| Primary language | Kannada (`kn`) |
| CPT data | KSEEB textbook corpus + IndicCorp v2 Kannada |
| SFT data | Kannada slice of `FoundryAILabs/k12-indian-curriculum-4.9m` |
| Evaluation | IndicQA Kannada, MILU Kannada, IndicGenBench, GEPA set, manual prompts |
| Local inference target | GGUF / LM Studio / llama.cpp-style testing |
| Model weights | TODO: add Hugging Face model link |
| License | TODO: add repository license |

---

## Repository structure

```text
Mayura-4B/
├── docs/
│   ├── dataset_strategy.md
│   ├── evaluation.md
│   ├── model_card.md
│   └── training_strategy.md
│
├── evaluation runs/
│   ├── IndicGenBench.ipynb
│   ├── IndicQA_eval.ipynb
│   └── MILU_Kannada_eval.ipynb
│
├── experiments/
│   ├── gemma_kannada_cpt_curriculum.py
│   ├── gemma_kannada_qa.py
│   ├── gepa_executor.ipynb
│   └── gepa_kn_serial_safe.py
│
└── results/
    ├── crosssum_kn_metrics.json
    ├── crosssum_kn_predictions.csv
    ├── flores_en_to_kn_metrics.json
    ├── flores_en_to_kn_predictions.csv
    ├── indicgenbench_kannada_summary.csv
    ├── indicqa_kannada_gguf_results.csv
    ├── milu_kannada_gguf_loglik_letter_plus_option.csv
    ├── xquad_kn_streaming_metrics.json
    └── xquad_kn_streaming_predictions.csv
```

---

## Documentation

The main project documentation is available under [`docs/`](docs/):

| File | Purpose |
|---|---|
| [`docs/dataset_strategy.md`](docs/dataset_strategy.md) | Explains the CPT, SFT, and evaluation dataset strategy. |
| [`docs/training_strategy.md`](docs/training_strategy.md) | Describes the CPT → SFT training pipeline, tokenizer strategy, checkpointing, and merge/export flow. |
| [`docs/evaluation.md`](docs/evaluation.md) | Reports benchmark results on IndicQA Kannada, MILU Kannada, and IndicGenBench. |
| [`docs/model_card.md`](docs/model_card.md) | Model-card style documentation covering intended use, training data, evaluation, limitations, and risks. |

---

## Dataset strategy

Mayura-4B uses a two-stage dataset design.

### 1. Continued Pretraining data

CPT uses Kannada-only raw text to improve Kannada fluency, script handling, vocabulary coverage, and educational-domain familiarity.

Primary CPT sources:

| Source | Role |
|---|---|
| KSEEB textbook corpus, Grade 3 to Grade 12 / 2nd PUC | High-quality educational Kannada text |
| IndicCorp v2 Kannada slice | Broader Kannada language exposure |

The intended CPT mixture prioritizes curated textbook Kannada while still including general-domain Kannada from IndicCorp v2.

### 2. Supervised Fine-Tuning data

SFT uses the Kannada slice of:

```text
FoundryAILabs/k12-indian-curriculum-4.9m
```

This dataset provides K-12 style question-answer and instruction examples across Indian languages. For Mayura-4B, the Kannada subset is used to improve Kannada educational QA and instruction-following behavior.

AI4Bharat IndicQuestionGeneration was explored during earlier experiments, but it is not documented as the final SFT dataset for the main pipeline.

---

## Training strategy

The main training strategy is:

```text
1. Start from Gemma 3 4B base.
2. Run Kannada continued pretraining on raw Kannada text.
3. Run Kannada supervised fine-tuning on QA/instruction data.
4. Merge/export checkpoints where needed.
5. Convert or test through GGUF for local inference.
6. Evaluate using Kannada-focused benchmarks.
```

The project uses parameter-efficient training methods such as LoRA/QLoRA to keep training practical on constrained GPUs.

Key training choices:

- Base tokenizer is kept stable across CPT, SFT, merge, and inference.
- Kannada-only or Kannada-majority data is preferred.
- English prompt labels are avoided in model-visible SFT prompts where possible.
- Checkpoint-resume support is important because training was designed for constrained environments such as college GPUs, Kaggle, and Colab.
- GGUF testing is used to validate local inference behavior.

---

## Experiment scripts

### `experiments/gemma_kannada_cpt_curriculum.py`

Curriculum-aware CPT script for Gemma 3 4B on Kannada data.

Main features:

- Uses `google/gemma-3-4b-pt` by default.
- Supports textbook JSONL and IndicCorp JSONL/JSONL.GZ inputs.
- Uses QLoRA 4-bit training.
- Uses LoRA target modules such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Includes phase-wise curriculum training.
- Supports crash-friendly checkpointing and resume.
- Saves phase adapters and a final CPT adapter.

Example command:

```bash
python experiments/gemma_kannada_cpt_curriculum.py \
  --textbook_jsonl /path/to/kannada_textbook_cpt.jsonl \
  --indiccorp_jsonl /path/to/indiccorp_kn.jsonl.gz \
  --output_dir ./outputs/cpt \
  --hf_token "$HF_TOKEN"
```

### `experiments/gemma_kannada_qa.py`

Experimental QA/question-generation fine-tuning script.

This script:

- Loads the Gemma 3 4B base model.
- Loads a CPT LoRA adapter.
- Merges the CPT adapter into the base model.
- Saves a merged checkpoint with 1 GB shards for lower CPU-RAM pressure.
- Reloads the merged model in 4-bit.
- Adds a fresh LoRA adapter for QA/question-generation training.
- Uses a custom collator for Gemma 3 tokenization behavior.

Note: this script currently uses `ai4bharat/IndicQuestionGeneration` in the code path. Treat it as an experimental script unless it is updated to match the final documented SFT dataset.

### `experiments/gepa_kn_serial_safe.py`

Serial-safe GEPA runner for Kannada GGUF model prompt optimization.

Main purpose:

- Optimize Kannada system prompts without changing model weights.
- Improve Kannada-only behavior.
- Improve MCQ answer formatting.
- Reduce leakage into English, Hindi, Tamil, Telugu, Malayalam, Gujarati, Bengali, Chinese, or mixed scripts.
- Run safely on Kaggle with local GGUF inference and provider-call limits.

### `experiments/gepa_executor.ipynb`

Notebook wrapper for running GEPA experiments interactively.

---

## Evaluation notebooks

The [`evaluation runs/`](evaluation%20runs/) directory contains notebooks for benchmark execution:

| Notebook | Purpose |
|---|---|
| [`IndicQA_eval.ipynb`](evaluation%20runs/IndicQA_eval.ipynb) | IndicQA Kannada evaluation. |
| [`MILU_Kannada_eval.ipynb`](evaluation%20runs/MILU_Kannada_eval.ipynb) | MILU Kannada multiple-choice evaluation. |
| [`IndicGenBench.ipynb`](evaluation%20runs/IndicGenBench.ipynb) | IndicGenBench Kannada generation/transfer evaluation. |

---

## Results

### Summary

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

### Interpretation

The strongest current result is **MILU Kannada**, where Mayura-4B reaches **40.55% accuracy**. This is a strong signal for Kannada multiple-choice understanding relative to several listed multilingual and Kannada-specific baselines in the evaluation documentation.

The model also performs strongly on **IndicQA Kannada**, reaching **30.86 Token-F1**. This suggests that Kannada CPT + SFT improved QA behavior beyond generic multilingual capability.

The **FLORES-IN en→kn** result shows promising Kannada generation quality with **31.53 chrF** and **72.26% Kannada-character ratio**.

The weakest area is **CrossSum-IN Kannada**, where the result suggests that cross-lingual summarization requires task-specific training or better prompting.

---

## Result files

The [`results/`](results/) directory contains metric JSONs and prediction CSVs.

| File | Description |
|---|---|
| [`results/indicqa_kannada_gguf_results.csv`](results/indicqa_kannada_gguf_results.csv) | IndicQA Kannada GGUF predictions/results. |
| [`results/milu_kannada_gguf_loglik_letter_plus_option.csv`](results/milu_kannada_gguf_loglik_letter_plus_option.csv) | MILU Kannada GGUF log-likelihood evaluation results. |
| [`results/flores_en_to_kn_metrics.json`](results/flores_en_to_kn_metrics.json) | FLORES-IN English → Kannada metric summary. |
| [`results/flores_en_to_kn_predictions.csv`](results/flores_en_to_kn_predictions.csv) | FLORES-IN predictions. |
| [`results/xquad_kn_streaming_metrics.json`](results/xquad_kn_streaming_metrics.json) | XQuAD-IN Kannada metric summary. |
| [`results/xquad_kn_streaming_predictions.csv`](results/xquad_kn_streaming_predictions.csv) | XQuAD-IN Kannada predictions. |
| [`results/crosssum_kn_metrics.json`](results/crosssum_kn_metrics.json) | CrossSum-IN Kannada metric summary. |
| [`results/crosssum_kn_predictions.csv`](results/crosssum_kn_predictions.csv) | CrossSum-IN Kannada predictions. |
| [`results/indicgenbench_kannada_summary.csv`](results/indicgenbench_kannada_summary.csv) | Compact IndicGenBench Kannada summary. |

---

## Setup

This repository does not currently provide a single locked environment file. A practical Python environment should include the usual LLM fine-tuning and evaluation stack.

Recommended base environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

Recommended training/evaluation packages:

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes
pip install pandas numpy tqdm scikit-learn evaluate sacrebleu rouge-score rapidfuzz
pip install huggingface_hub
```

For GGUF/local inference experiments:

```bash
pip install llama-cpp-python
```

For GEPA prompt optimization:

```bash
pip install litellm gepa rapidfuzz tqdm pandas huggingface_hub
```

Exact package versions may need to be adjusted depending on GPU, CUDA version, and whether the run is on Kaggle, Colab, local Linux, or a college GPU environment.

---

## Hugging Face access

Gemma models may require Hugging Face access approval and authentication.

Set your token before running training scripts:

```bash
export HF_TOKEN="your_huggingface_token"
```

Then pass it to scripts where required:

```bash
--hf_token "$HF_TOKEN"
```

---

## Recommended inference behavior

The model is intended to answer in Kannada. A stricter Kannada-only system prompt is recommended during testing.

Example system prompt:

```text
ನೀನು ಕೇವಲ ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸುವ ಅತ್ಯಂತ ನಿಖರ ಮತ್ತು ಸ್ಪಷ್ಟವಾದ AI ಸಹಾಯಕ.
ಬಳಕೆದಾರರ ಪ್ರಶ್ನೆಗಳಿಗೆ ನೇರವಾಗಿ, ಸಂಕ್ಷಿಪ್ತವಾಗಿ ಮತ್ತು ಶಾಲಾ ಮಟ್ಟದ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸು.
ನಿನ್ನ ಉತ್ತರವು ಕೇವಲ ಕನ್ನಡದಲ್ಲೇ ಇರಬೇಕು. ಇಂಗ್ಲಿಷ್, ಹಿಂದಿ, ತಮಿಳು, ತೆಲುಗು, ಮಲಯಾಳಂ, ಗುಜರಾತಿ, ಬೆಂಗಾಲಿ, ಚೈನೀಸ್ ಅಥವಾ ಮಿಶ್ರ-ಲಿಪಿ ಬಳಕೆಯನ್ನು ತಪ್ಪಿಸು.
ಪ್ರಶ್ನೆಗೆ ಸರಿಯಾದ ಉತ್ತರ ತಿಳಿದಿಲ್ಲದಿದ್ದರೆ, "ಕ್ಷಮಿಸಿ, ನನಗೆ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ಗೊತ್ತಿಲ್ಲ" ಎಂದು ಪ್ರಾಮಾಣಿಕವಾಗಿ ತಿಳಿಸು.
```

Suggested benchmark decoding:

| Setting | Suggested Value |
|---|---:|
| Temperature | 0.0–0.2 |
| Top-p | 0.8–0.95 |
| Top-k | 40–100 |
| Repetition penalty | 1.05–1.15 |
| Max new tokens | Task-dependent |

For benchmark reporting, deterministic or near-deterministic decoding is preferred.

---

## Known limitations

- The model may still occasionally produce non-Kannada or mixed-script output.
- Cross-lingual summarization is currently weak relative to QA and MCQ performance.
- Benchmark comparisons may not be directly equivalent unless prompt format, decoding settings, and evaluation harnesses match exactly.
- The GitHub repository contains scripts, docs, notebooks, and results, but model-weight release details should be added separately.
- Some scripts are experimental and may not exactly match the final documented training path.
- The model should not be used for high-stakes medical, legal, financial, or safety-critical tasks.
- Important answers should be verified against reliable sources.

---

## Roadmap

Recommended next improvements:

- Add a `requirements.txt` or `environment.yml` file.
- Add exact model-weight links and GGUF links.
- Add final license information.
- Add exact CPT and SFT hyperparameter tables.
- Add raw evaluation command examples for each benchmark notebook.
- Add a reproducible `scripts/` directory separate from exploratory `experiments/`.
- Add a dedicated GGUF conversion guide.
- Add summarization-specific SFT data and rerun CrossSum-IN Kannada.
- Add a small Kannada script-ratio regression test for future checkpoints.

---

## Citation

If you use this repository or model, cite the project as:

```bibtex
@misc{mayura4b,
  title        = {Mayura-4B: Kannada-first adaptation of Gemma 3 4B},
  author       = {Vaibhav Mugulavalli},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/VaibhavMugulavalli/Mayura-4B}
}
```

---

## Contact

For issues, suggestions, or collaboration, open an issue in this repository.

