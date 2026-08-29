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

---

## Repository Structure

```
Abusive_Language_Detection/
├── single_algorithms_experiment.py   # Individual classical classifiers
├── ensemble_experiment.py            # Ensemble approaches
├── feature_ablation_experiment.py    # Two-stage feature-ablation study
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
   ```

Each script performs dataset validation, TF-IDF feature extraction,
stratified 5-fold cross-validation, and writes its aggregated results into
the `results/` directory.

---

## Results

- `results/algorithm_results.csv` and `results/ensemble_results.csv` were regenerated on `Updated_dataset.csv` on 2026-08-29. Corresponding run logs are in `results/single_run.log` and `results/ensemble_run.log`.
- `results/feature_ablation/FINAL_RESULTS.csv` reports the Stage 2 classifier comparison on the winning feature configuration selected in Stage 1; per-fold and summary files for both stages are alongside it in `results/feature_ablation/`.

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
