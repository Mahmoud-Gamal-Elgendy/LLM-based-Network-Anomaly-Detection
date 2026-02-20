# LLM-based Network Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This project investigates **parameter-efficient fine-tuning of a Large Language Model (LLM) for network intrusion detection**, framing the classification task as a natural language reasoning problem. We convert structured NSL-KDD network traffic features into natural language descriptions (*textification*), then fine-tune **Microsoft phi-2 (2.7B parameters)** using **QLoRA (4-bit quantization + LoRA adapters)** to classify traffic into five categories: Normal, DoS, Probe, R2L, and U2R.

The system generates human-readable classification outputs with natural language justifications, offering a step toward more interpretable network security decisions. This is an **experimental/research prototype** — results are preliminary and subject to the limitations discussed below.

### Research Questions

1. Can an LLM fine-tuned via QLoRA learn to classify network traffic from textified feature descriptions?
2. Does the natural language generation paradigm produce outputs that are more interpretable than traditional classifier outputs?
3. What are the failure modes when applying LLM-based classification to heavily imbalanced network security datasets?

---

## Methodology

### Pipeline Overview

```
Raw NSL-KDD Logs → Preprocessing → Textification (NLG) → QLoRA Fine-Tuning → Inference & Evaluation
```

### 4-Module Architecture

| Module | Script | Purpose | Key Technique |
|--------|--------|---------|---------------|
| **1. Preprocessing** | `1_data_preprocessing.py` | Clean NSL-KDD, map 39 attack types → 5 categories | Label mapping, validation, missing value handling |
| **2. Textification** | `2_data_textification.py` | Convert tabular features → natural language | Template-based NLG with conditional feature inclusion |
| **3. Fine-Tuning** | `3_train_model.py` | Train phi-2 with QLoRA | 4-bit NF4 quantization, LoRA r=16, causal LM objective |
| **4. Evaluation** | `4_evaluate_model.py` | Classification metrics, confusion matrix, output analysis | Greedy decoding, keyword extraction for label parsing |

### Textification Strategy

Each network connection record (41 numerical/categorical features) is converted to a natural language paragraph using template-based generation. Three verbosity levels are supported:

- **Basic:** Protocol, service, flag, byte counts only
- **Detailed:** Adds duration, connection counts, error rates (conditional inclusion)
- **Rich:** All features including authentication events, privilege access, host statistics

**Example (detailed verbosity):**
```
Input:  [tcp, private, S0, 0, 0, count=283, serror_rate=1.00, ...]
Output: "A tcp connection to private service with connection status S0.
         was instantaneous. transferred 0 bytes from source and 0 bytes
         from destination. was part of 283 connections to the same host
         in the past 2 seconds. had high SYN error rate (1.00)."
```

> **Note:** The conditional inclusion of features (e.g., `serror_rate` only reported when > 0.5) introduces **information loss** for subtle attack patterns. This is a known limitation — see [Limitations](#limitations).

### Fine-Tuning Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | microsoft/phi-2 (2.7B) | Small enough for consumer GPU; strong reasoning capabilities |
| Quantization | NF4, double quantization | Reduces memory to ~2GB; enables 8GB GPU training |
| LoRA rank (r) | 16 | Balance between capacity and parameter efficiency |
| LoRA alpha | 32 | Standard 2× rank scaling |
| LoRA dropout | 0.05 | Light regularization |
| Target modules | q_proj, k_proj, v_proj, dense, fc1, fc2 | Full attention + FFN coverage |
| Trainable parameters | ~16M / 2.7B (**0.6%**) | Parameter-efficient fine-tuning |
| Epochs | 1 | Training loss plateaus early; more epochs risk overfitting |
| Effective batch size | 16 (4 × 4 accumulation) | Memory-constrained choice |
| Learning rate | 2e-4 (linear decay, 50-step warmup) | Standard for LoRA fine-tuning |
| Optimizer | Paged AdamW 8-bit | Memory-efficient variant |
| Max sequence length | 256 tokens | Covers >99% of samples; reduces padding waste |
| Training objective | Causal LM (next-token prediction) | Standard for instruction tuning |

**Training Dynamics (1 epoch, 1,575 steps):**
- Loss: 3.37 → 0.12 (rapid convergence in first 200 steps, plateau thereafter)
- Gradient norm: stabilized ~0.15 after step 200
- Eval loss at step 1500: 0.1997

---

## Dataset: NSL-KDD

### Overview

The **NSL-KDD** dataset is a refined version of KDD Cup '99 with redundant records removed. It contains 41 features per network connection and 39 specific attack types grouped into 5 categories.

### Class Distribution

| Split | Total | Normal | DoS | Probe | R2L | U2R |
|-------|-------|--------|-----|-------|-----|-----|
| **Train** | 25,192 | 53.4% | 36.7% | 9.1% | 0.8% | **0.04%** |
| **Test** | 22,544 | 43.1% | 33.1% | 10.7% | **12.8%** | 0.3% |

> **Critical imbalance:** U2R has only ~10 training samples. R2L has ~200 training samples but 12.8% of the test set — a significant distribution shift between train and test.

### Dataset Limitations

- **Temporal relevance:** NSL-KDD is derived from 1998 DARPA data. Attack patterns do not represent modern network threats, and results should not be extrapolated to production environments.
- **Synthetic origin:** Traffic was generated in a controlled testbed, not captured from real networks.
- **Feature engineering:** The 41 features were pre-engineered by domain experts — the LLM operates on textified versions of these features, not raw packet data.
- **Standard benchmark only:** We use NSL-KDD as a well-understood benchmark for comparing approaches, not as evidence of real-world viability.

### Download

[NSL-KDD on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd) — place `KDDTrain.txt` and `KDDTest.txt` in `Data/`.

---

## Results

### Experimental Setup

- **Training:** 1 epoch on full training set (25,192 samples), NVIDIA GPU with CUDA
- **Evaluation:** Stratified subsample of **199 test samples** (due to CPU inference constraints)
- **Decoding:** Greedy (temperature=0, no sampling) for reproducibility
- **Label extraction:** Keyword matching on generated text

> **Important:** All metrics below are from a 199-sample subsample, not the full 22,544-sample test set. Per-class metrics for minority classes (especially U2R with n=1) are **not statistically reliable**. Full test set evaluation on GPU is needed for publication-quality results.

### Observed Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 74.37% |
| **Macro Precision** | 0.4449 |
| **Macro Recall** | 0.4902 |
| **Macro F1** | 0.4637 |
| **Weighted Precision** | 0.6593 |
| **Weighted Recall** | 0.7437 |
| **Weighted F1** | 0.6937 |

### Per-Class Breakdown

| Category | Precision | Recall | F1-Score | Support (n) |
|----------|-----------|--------|----------|-------------|
| **DoS** | 0.885 | 0.818 | 0.850 | 66 |
| **Normal** | 0.687 | 0.919 | 0.786 | 86 |
| **Probe** | 0.652 | 0.714 | 0.682 | 21 |
| **R2L** | 0.000 | 0.000 | 0.000 | 25 |
| **U2R** | 0.000 | 0.000 | 0.000 | 1 |

### Confusion Matrix

|  | Pred: DoS | Pred: Normal | Pred: Probe | Pred: R2L | Pred: U2R |
|---|---|---|---|---|---|
| **True: DoS** | **54** | 10 | 2 | 0 | 0 |
| **True: Normal** | 4 | **79** | 3 | 0 | 0 |
| **True: Probe** | 3 | 3 | **15** | 0 | 0 |
| **True: R2L** | 0 | 22 | 3 | **0** | 0 |
| **True: U2R** | 0 | 1 | 0 | 0 | **0** |

### Analysis

**What works:**
- **DoS detection** (F1=0.85) — high connection counts and SYN error rates provide clear textual signals that the model learned
- **Normal traffic** (F1=0.79) — the model correctly identifies standard connection patterns

**What fails:**
- **R2L (F1=0.00)** — all 25 R2L samples were misclassified, primarily as "normal" (22/25). R2L attacks (e.g., `warezmaster`, `guess_passwd`) often resemble normal connections in their textified form, lacking distinctive traffic-level signatures
- **U2R (F1=0.00)** — only 1 test sample; completely undetectable given ~10 training examples
- **Normal bias** — the model's 91.9% recall on normal traffic comes at the cost of absorbing R2L and U2R samples as false negatives

### Sample Outputs

**Correct DoS detection:**
```
Input:  "A tcp connection to gopher service with connection status S0.
         was instantaneous. transferred 0 bytes... was part of 283
         connections to the same host... had high SYN error rate (1.00)."

Output: "This network traffic shows signs of denial of service attack
         (specifically: neptune). Classification: dos"
```

**Failure case (R2L → Normal):**
```
Input:  "A tcp connection to ftp_data service with connection status SF.
         lasted 280 seconds. transferred 283618 bytes from source and
         0 bytes from destination."

Output: "This network traffic represents normal network activity.
         Classification: normal."

True label: r2l (warezmaster)
```

> The R2L failure illustrates a fundamental limitation: the textified description of this `warezmaster` attack is indistinguishable from legitimate FTP activity when only traffic-level features are described. Content-based features (which are sparse for R2L in the training data) are needed but were not prominently represented in the textification.

---

## Limitations

### Methodological Limitations

1. **Evaluation sample size.** Metrics are from 199 subsampled test instances. Per-class statistics for R2L (n=25) and U2R (n=1) have no statistical power. Full test set evaluation is required.

2. **No held-out validation set.** The test JSON is used as the eval set during training, meaning checkpoint selection is influenced by test performance. This constitutes **mild information leakage** — a proper 3-way split (train/val/test) is needed.

3. **No baseline comparisons.** No traditional ML baselines (Random Forest, XGBoost, SVM) or other LLM configurations were trained on the same data. Without baselines, the results cannot be contextualized.

4. **No statistical rigor.** Single run, single seed (42), no confidence intervals, no significance testing. Results may not be reproducible across random seeds.

5. **Causal LM loss over full sequence.** The training objective computes loss over the entire prompt (instruction + input + output), wasting model capacity on predicting instruction tokens. Masking the instruction/input tokens in the loss would focus learning on the classification output.

### Technical Limitations

6. **Information loss in textification.** Conditional feature inclusion (thresholds like `serror_rate > 0.5`) discards low-magnitude signals. Continuous features are converted to text with limited precision. The ordering and phrasing of features may introduce unwanted inductive biases.

7. **Class imbalance unaddressed.** No oversampling, class weighting, or focal loss. U2R (~10 training samples) and R2L (~200 samples) are effectively unlearnable at this scale.

8. **Template-based output ≠ true XAI.** The model generates text from memorized training templates (e.g., "This network traffic shows signs of..."). It does not perform feature attribution, attention-based explanation, or counterfactual reasoning. The outputs are better described as **natural language classification** than explainable AI.

9. **Dataset age.** NSL-KDD (derived from 1998 data) does not reflect modern attack patterns, encrypted traffic, or contemporary protocols. Results do not generalize to production network security.

---

## Project Structure

```
LLM-based Network Anomaly Detection/
│
├── Data/
│   ├── KDDTrain.txt                                   # Raw NSL-KDD training data
│   ├── KDDTest.txt                                    # Raw NSL-KDD test data
│   ├── train_processed.csv                            # Preprocessed training data
│   ├── test_processed.csv                             # Preprocessed test data
│   ├── attack_mapping.json                            # 39 attacks → 5 categories
│   ├── train_textified_detailed_instruction.json      # Textified training set
│   └── test_textified_detailed_instruction.json       # Textified test set
│
├── Scripts/
│   ├── 1_data_preprocessing.py                        # Data loading, cleaning, mapping
│   ├── 2_data_textification.py                        # Feature → natural language conversion
│   ├── 3_train_model.py                               # QLoRA fine-tuning pipeline
│   └── 4_evaluate_model.py                            # Inference, metrics, analysis
│
├── Models/
│   └── phi2-kdd-lora/
│       └── final/                                     # Trained LoRA adapter weights
│           ├── adapter_config.json
│           ├── adapter_model.safetensors               # ~65 MB
│           ├── tokenizer.json
│           └── tokenizer_config.json
│
├── Results/
│   ├── metrics.json                                   # All evaluation metrics
│   ├── predictions.json                               # Per-sample predictions
│   ├── confusion_matrix.png                           # Confusion matrix heatmap
│   ├── per_class_metrics.png                          # Per-class bar chart
│   └── evaluation_report.txt                          # Summary report
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| CUDA | 11.8 | 12.1 |
| GPU VRAM | 8 GB | 16 GB |
| System RAM | 16 GB | 32 GB |
| Disk space | 20 GB | 50 GB |

> **CPU-only:** Training is not feasible on CPU. Evaluation can run on CPU but takes several hours for the full test set.

### Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd "LLM-based Network Anomaly Detection"

# 2. Install PyTorch with CUDA (choose your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# 3. Install project dependencies
pip install -r requirements.txt
```

---

## Usage

### Running the Full Pipeline

```bash
cd Scripts

# Module 1: Preprocessing (~2 min)
python 1_data_preprocessing.py

# Module 2: Textification (~5 min)
python 2_data_textification.py

# Module 3: Fine-Tuning (~2-3 hours on T4/RTX 3080)
python 3_train_model.py

# Module 4: Evaluation (~15 min GPU / several hours CPU)
python 4_evaluate_model.py
```

### Module Details

#### Module 1: Data Preprocessing

Loads raw NSL-KDD text files, cleans labels (strips trailing dots), maps 39 specific attack types to 5 broad categories using a predefined dictionary, validates data quality, and exports processed CSVs.

#### Module 2: Data Textification

Converts each of the 41 numerical/categorical features into natural language using template functions. Supports three verbosity levels (`basic`, `detailed`, `rich`) and two output formats (`instruction`, `prompt_completion`). Default: `detailed` + `instruction`.

**Configuration** (edit in script):
```python
VERBOSITY = 'detailed'      # 'basic', 'detailed', 'rich'
FORMAT_TYPE = 'instruction'  # 'instruction', 'prompt_completion'
```

#### Module 3: Fine-Tuning

Loads phi-2 with 4-bit NF4 quantization via bitsandbytes, applies LoRA adapters to all attention and FFN layers, and trains using HuggingFace Trainer with paged AdamW 8-bit. Includes gradient flow verification, gradient checkpointing with `use_reentrant=False`, and automatic checkpoint resumption.

**Monitoring:**
```bash
tensorboard --logdir Models/logs
```

#### Module 4: Evaluation

Loads the trained adapter, runs greedy inference on the test set (or a stratified subsample), extracts predicted labels via keyword matching, and computes accuracy, precision, recall, F1 (macro and weighted), confusion matrix, and per-class breakdowns. Exports all results to `Results/`.

**Configuration** (edit in script):
```python
max_samples = 200   # Set to 0 for full test set (requires GPU)
do_sample = False   # Greedy decoding
```

---
## Reproducibility

| Item | Value |
|------|-------|
| Random seed | 42 |
| Training epochs | 1 |
| Total training steps | 1,575 |
| Final training loss | 0.118 |
| Eval loss (step 1500) | 0.200 |
| PEFT version | 0.18.1 |
| Quantization | NF4, double quant, float16 compute |
| Evaluation subsample | 199 samples (stratified) |

To reproduce: run Modules 1–4 sequentially. The model checkpoint is included in `Models/phi2-kdd-lora/final/`. The base model (phi-2) will be downloaded automatically from HuggingFace Hub on first run (~5 GB).

---

## Acknowledgments

- **Microsoft Research** for the phi-2 model
- **HuggingFace** for transformers, PEFT, and datasets libraries
- **NSL-KDD** dataset creators (University of New Brunswick)
- **bitsandbytes** authors for quantization support

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

**Author:** Mahmoud Youssef
