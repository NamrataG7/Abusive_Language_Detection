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

- `results/algorithm_results.csv` and `results/ensemble_results.csv` were regenerated on `Updated_dataset.csv` on 2026-08-29. Corresponding run logs are in `results/single_run.log` and `results/ensemble_run.log`.
- `results/feature_ablation/FINAL_RESULTS.csv` reports the Stage 2 classifier comparison on the winning feature configuration selected in Stage 1; per-fold and summary files for both stages are alongside it in `results/feature_ablation/`.
- `results/enhanced/FINAL_ENHANCED_RESULTS.csv` and `results/enhanced/experiment_manifest.csv` summarise the four-stage enhanced experiment (preprocessing → feature → scaling → classifier ablation); per-stage fold and summary CSVs are alongside them in `results/enhanced/`.
- `results/enhanced/rf_oof_confusion_matrix.csv` and the companion `rf_oof_*` files record the Random Forest out-of-fold error analysis on the winning configuration.
- `results/enhanced/rf_vs_lr_paired_permutation_test.csv` reports the paired RF-vs-LR permutation test with bootstrap confidence intervals on OOF predictions.
- `results/enhanced/class_weight_ablation_summary.csv` and `results/enhanced/majority_baseline_summary.csv` contextualise the classifier results against class-weight variants and a majority-class baseline.
- `results/enhanced/rf_repeated5fold_summary.csv` reports the repeated 5-fold Random Forest stability check, and `results/enhanced/dataset_metadata_audit.csv` documents the corpus audit (duplicates, author/thread/time/language metadata availability).

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
