# An Experimental Study of Machine Learning Classifiers for Abusive Language Detection

This repository contains the source code, dataset, and reproducible
experimental pipeline for a comparative study of machine learning classifiers
applied to the task of abusive language detection on short English social
media text. The study is organised into two complementary experiments,
evaluated separately:

1. **Individual classical classifiers** — Support Vector Machine (SVM),
   K-Nearest Neighbours (KNN), Logistic Regression (LR), Multinomial Naive
   Bayes (NB), Gradient Boosting (GB), and Random Forest (RF), implemented
   in `single_algorithms_experiment.py`.
2. **Ensemble approaches** — combinations of the above base learners
   (e.g. soft/hard voting and related ensemble strategies), implemented in
   `ensemble_experiment.py`.

Both experiments share a unified TF-IDF feature representation, stratified
5-fold cross-validation, and a common set of evaluation metrics, so that
observed differences in performance can be attributed to the classifier (or
ensemble) itself rather than to differences in preprocessing or evaluation
protocol.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Requirements](#requirements)
5. [How to Reproduce](#how-to-reproduce)
6. [License](#license)
7. [Citation](#citation)

---

## Overview

The automated detection of abusive language on social media platforms remains
a challenging classification problem due to class imbalance, informal
language, and the highly context-dependent nature of abuse. This study
contributes an end-to-end, reproducible pipeline that compares both
individual classical classifiers and ensemble methods on a curated corpus of
English tweets. The objective is to provide a controlled empirical baseline
against which future work — including deep learning and transformer-based
approaches — can be compared.

The two experiment scripts are run independently and each produces its own
set of aggregated results, allowing the single-model and ensemble regimes to
be analysed on equal footing.

### Feature Ablation

In addition to the two classifier-focused experiments, `feature_ablation_experiment.py`
runs a two-stage feature-ablation study. Stage 1 uses Random Forest as a fixed
probe to compare seven TF-IDF-based feature configurations (word, character,
hybrid word+character, and progressively enriched hybrid feature sets with
lexical, punctuation/emoji, lexicon, and sentiment signals) and selects the
best-performing configuration by macro-F1. Stage 2 then compares all six
classifiers (SVM, KNN, LR, NB, GB, RF) on the winning feature set under the
same 5-fold cross-validation protocol. Outputs are written to
`results/feature_ablation/` as `stage1_rf_feature_ablation_folds.csv`,
`stage1_rf_feature_ablation_summary.csv`,
`stage2_classifier_comparison_folds.csv`,
`stage2_classifier_comparison_summary.csv`, and `FINAL_RESULTS.csv`.

### Enhanced Experiment

`enhanced_experiment.py` extends the feature-ablation study into a four-stage
pipeline over the same corpus and cross-validation protocol. Stage 1 performs a
**preprocessing ablation** (raw, stopword removal, stemming, lemmatisation, and
their combinations) using Random Forest on the E2 hybrid feature set; the
winning preprocessing is fixed for later stages. Stage 2 repeats the
**feature ablation** on top of the selected preprocessing. Stage 3 evaluates
**scaling sensitivity** of numerical features for models known to be
scale-sensitive (KNN, RF). Stage 4 conducts a final **classifier comparison**
(SVM, KNN, LR, NB, GB, RF) on the winning preprocessing and feature
configuration. An optional Stage 0 reproduces the E2 Random Forest baseline
for direct comparison, and an optional annotation-agreement analysis is run
when the dataset contains independent annotator label columns.

Alongside the four stages, the script emits additive diagnostics: a
**dataset metadata audit** (`dataset_metadata_audit.csv`); a **majority-class
baseline** (`majority_baseline_*`); a **class-weight ablation** contrasting
balanced and unbalanced settings for RF, LR and SVM
(`class_weight_ablation_*`); a **Random Forest out-of-fold error analysis**
comprising per-record predictions, confusion matrix, per-metric summary,
false-positive and false-negative listings, and probability-bin calibration
(`rf_oof_*`, `rf_false_positives.csv`, `rf_false_negatives.csv`,
`rf_probability_bins.csv`); a **paired RF-vs-LR permutation test** with
bootstrap confidence intervals on the OOF predictions
(`rf_lr_paired_oof_predictions.csv`,
`rf_vs_lr_paired_permutation_test.csv`); and an optional
**RepeatedStratifiedKFold RF stability** analysis
(`rf_repeated5fold_folds.csv`, `rf_repeated5fold_summary.csv`). All outputs
are written to `results/enhanced/` alongside `FINAL_ENHANCED_RESULTS.csv`
and `experiment_manifest.csv`.

---

## Repository Structure

```
Abusive_Language_Detection/
├── single_algorithms_experiment.py   # Individual classical classifiers
├── ensemble_experiment.py            # Ensemble approaches
├── feature_ablation_experiment.py    # Two-stage feature-ablation study
├── enhanced_experiment.py            # Multi-stage preprocessing / feature / scaling / classifier ablation
├── Updated_dataset.csv                   # Labelled corpus
├── requirements.txt                      # Python dependencies
├── results/                              # Aggregated experimental results
├── LICENSE                               # MIT License
└── README.md
```

---

## Dataset

The dataset (`Updated_dataset.csv`) contains approximately **20,113** English-
language short text records with the following schema:

| Column       | Type    | Description                                   |
|--------------|---------|-----------------------------------------------|
| `record_id`  | string  | Unique record identifier                      |
| `text`       | string  | Raw short text (tweet)                        |
| `label`      | integer | Binary label: `0` non-abusive, `1` abusive    |
| `label_name` | string  | Human-readable label (`Non-Abusive` / `Abusive`) |
| `word_count` | integer | Word count of the text                        |
| `char_count` | integer | Character count of the text                   |

Labels are binary: `0` denotes **Non-Abusive** and `1` denotes **Abusive**.
The corpus exhibits a moderate class imbalance in favour of the non-abusive
class, which motivates the use of macro-averaged F1 as the primary
evaluation criterion.

The dataset is intended for academic research on abusive-language detection.
It contains text that some readers may find offensive; this is inherent to
the task and does not reflect the views of the authors.

---

## Requirements

- Python ≥ 3.9
- Dependencies listed in [`requirements.txt`](requirements.txt).

Install:

```bash
pip install -r requirements.txt
```

The feature-ablation experiment uses NLTK's VADER sentiment analyser and
POS tagger. After installing the Python packages, download the required
NLTK data (one-time):

```bash
python -m nltk.downloader vader_lexicon punkt punkt_tab \
    averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

---

## How to Reproduce

1. Clone the repository:

   ```bash
   git clone https://github.com/NamrataG7/Abusive_Language_Detection.git
   cd Abusive_Language_Detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run each experiment separately. Both scripts read `Updated_dataset.csv`
   from the current working directory:

   ```bash
   python single_algorithms_experiment.py
   python ensemble_experiment.py
   python feature_ablation_experiment.py
   python enhanced_experiment.py --run-stage0 --run-repeated-stability
   ```

   Additional CLI flags: omit `--run-stage0` to skip baseline reproduction,
   add `--skip-annotation` to skip annotator agreement, add
   `--skip-new-diagnostics` to skip the additive diagnostic outputs, and
   omit `--run-repeated-stability` to skip the repeated 5-fold RF stability
   analysis.

Each script performs dataset validation, TF-IDF feature extraction,
stratified 5-fold cross-validation, and writes its aggregated results into
the `results/` directory.

---

## Results

All numbers below are means over five stratified cross-validation folds on
`Updated_dataset.csv` (20,114 rows; 87.4% non-abusive, 12.6% abusive) with
`random_state=42`. Metric definitions are consistent across all four
experiments.

### Cross-experiment comparison (best model per experiment)

| Experiment                         | Best model / configuration                                              | Accuracy | Precision | Recall | Macro-F1 |
|------------------------------------|--------------------------------------------------------------------------|----------|-----------|--------|----------|
| Majority-class baseline            | Predict the majority class                                               | 0.8741   | –         | 0.0000 | 0.4664   |
| `single_algorithms_experiment.py`  | Random Forest on TF-IDF (word 1–2 gram)                                  | 0.9494   | 0.8180    | 0.7702 | 0.8821   |
| `ensemble_experiment.py`           | Ensemble B (soft voting)                                                 | 0.9450   | 0.7752    | 0.7931 | 0.8762   |
| `feature_ablation_experiment.py`   | Random Forest on E2 hybrid word + character TF-IDF                       | 0.9516   | 0.8432    | 0.7572 | 0.8852   |
| `enhanced_experiment.py` (final)   | Random Forest, lemmatisation + E2 hybrid word + character TF-IDF         | **0.9531** | **0.8437** | **0.7706** | **0.8893** |

The enhanced pipeline improves on the single-algorithm baseline by
`+0.0072` Macro-F1 and on the ensemble by `+0.0131` Macro-F1. All classical
classifiers substantially outperform the majority-class baseline
(`Macro-F1 = 0.4664`), confirming that the modelling signal is real and not
an artefact of class imbalance.

### Enhanced experiment: classifier comparison

Stage 4 of `enhanced_experiment.py` evaluates all six classifiers on the
selected preprocessing (`P3_Lemmatization`) and feature configuration
(`E2_Hybrid_Word_Character`):

| Classifier | Accuracy | Precision | Specificity | Recall | Macro-F1 |
|------------|----------|-----------|-------------|--------|----------|
| **RF**     | 0.9531   | 0.8437    | 0.9794      | 0.7706 | **0.8893** |
| LR         | 0.9448   | 0.7623    | 0.9633      | 0.8160 | 0.8782   |
| SVM        | 0.9421   | 0.7644    | 0.9653      | 0.7809 | 0.8697   |
| GB         | 0.9376   | 0.8370    | 0.9824      | 0.6269 | 0.8409   |
| NB         | 0.9315   | 0.8218    | 0.9818      | 0.5823 | 0.8216   |
| KNN        | 0.8748   | 0.6402    | 0.9984      | 0.0166 | 0.4826   |

### Statistical significance, calibration and stability

- **RF vs LR paired permutation test** (5,000 permutations on
  out-of-fold predictions,
  `results/enhanced/rf_vs_lr_paired_permutation_test.csv`):
  ΔMacro-F1 = **+0.0111** (95% bootstrap CI **[0.0063, 0.0159]**),
  two-sided **p = 0.00020**. Random Forest's advantage over Logistic
  Regression is statistically significant.
- **RF stability across 15 folds** (repeated 5-fold × 3 repeats,
  `results/enhanced/rf_repeated5fold_summary.csv`):
  Macro-F1 = **0.8883 ± 0.0064**, Recall = 0.7709 ± 0.0124. Variance is
  under one percentage point, so the reported result is not a fold-lucky
  outcome.
- **RF calibration on OOF predictions**
  (`results/enhanced/rf_probability_bins.csv`):
  observed abusive rate is 96.1% in the (0.9, 1.0] bin and 91.6% in the
  (0.8, 0.9] bin, decreasing monotonically towards low-confidence bins.
  RF is well-calibrated on high-confidence positives, which is useful for
  downstream thresholding.
- **Class-weight ablation**
  (`results/enhanced/class_weight_ablation_summary.csv`): `balanced` is
  critical for LR (Macro-F1 0.878 vs 0.864 unbalanced) and SVM, but RF is
  essentially insensitive (0.8893 vs 0.8889).
- **Scaling sensitivity (Stage 3,
  `results/enhanced/stage3_scaling_ablation_summary.csv`):** scaling
  handcrafted numeric features lifts KNN Macro-F1 from 0.5414 to 0.5990
  but has negligible effect on RF (0.8861 → 0.8866), consistent with
  ensemble tree insensitivity to feature scale.

### RF error analysis (out-of-fold, winning configuration)

Confusion matrix from `results/enhanced/rf_oof_confusion_matrix.csv`:

|                        | Predicted Non-Abusive | Predicted Abusive |
|------------------------|-----------------------|-------------------|
| **Actual Non-Abusive** | 17,218                | 363               |
| **Actual Abusive**     | 581                   | 1,952             |

Corresponding pooled OOF metrics
(`results/enhanced/rf_oof_metrics.csv`):
Accuracy = 0.9531, Macro-F1 = 0.8893, abusive-class Precision = 0.8432,
Recall = 0.7706, F1 = 0.8053. Full lists of the specific mis-classified
examples are in `results/enhanced/rf_false_positives.csv` and
`results/enhanced/rf_false_negatives.csv` for qualitative error review.

### Where to find each artefact

- `results/algorithm_results.csv`, `results/ensemble_results.csv` — 5-fold
  means for the single-algorithm and ensemble experiments; run logs in
  `results/single_run.log` and `results/ensemble_run.log`.
- `results/feature_ablation/` — Stage 1 and Stage 2 fold + summary CSVs,
  with `FINAL_RESULTS.csv` as the two-stage rollup.
- `results/enhanced/` — Stage 0 (opt-in E2 reproduction), Stage 1
  (preprocessing), Stage 2 (feature), Stage 3 (scaling), and Stage 4
  (classifier) fold and summary CSVs, `FINAL_ENHANCED_RESULTS.csv`,
  `experiment_manifest.csv`, plus the additive diagnostics
  (`majority_baseline_*`, `class_weight_ablation_*`, `rf_oof_*`,
  `rf_false_*`, `rf_probability_bins.csv`,
  `rf_lr_paired_oof_predictions.csv`,
  `rf_vs_lr_paired_permutation_test.csv`, `rf_repeated5fold_*`,
  `dataset_metadata_audit.csv`).

### Final takeaway

1. The **best classical pipeline on this corpus is
   lemmatisation + hybrid word/character TF-IDF + Random Forest**
   (Macro-F1 = 0.8893, stable at ± 0.0064 across 15 folds).
2. This pipeline **beats every single-algorithm, ensemble, and
   feature-ablation configuration tested** and its advantage over the
   nearest competitor (Logistic Regression) is statistically significant
   (paired permutation p = 0.00020).
3. RF is also the **cheapest strong model** — training in under two
   seconds per fold — while SVM (~52 s/fold) and GB (~72 s/fold) trail it
   in both Macro-F1 and runtime.
4. Recall on the abusive class is 0.7706 and RF is well-calibrated on
   high-confidence positives, so operationally the model is best suited
   to precision-oriented deployments (e.g. moderator triage) rather than
   full automation.
5. All model families materially outperform the trivial majority
   baseline (Macro-F1 = 0.4664), and the four-experiment progression
   (single-algorithms → ensemble → feature ablation → enhanced) shows a
   clear, monotone quality improvement as preprocessing and feature
   engineering are added — evidence that further gains are more likely to
   come from richer text representations than from further classical
   classifier swaps.

---

## License

The source code in this repository is released under the terms of the
[MIT License](LICENSE). The dataset is provided for academic research use
only. No formal copyright or other intellectual-property registration has
been filed for the contents of this repository at the time of publication.

---

## Citation

If you use this code or the accompanying dataset in your research, please
cite this repository:

```bibtex
@misc{aald2025,
  title        = {An Experimental Study of Machine Learning Classifiers for
                  Abusive Language Detection},
  author       = {{NamrataG7}},
  year         = {2025},
  howpublished = {\url{https://github.com/NamrataG7/Abusive_Language_Detection}},
  note         = {GitHub repository}
}
```
