# An Experimental Study of Machine Learning Classifiers for Abusive Language Detection

This repository contains the source code, dataset, and reproducible experimental
pipeline for a comparative study of six classical machine learning classifiers
applied to the task of abusive language detection on short English social media
text. The experiment evaluates Support Vector Machine (SVM), K-Nearest
Neighbours (KNN), Logistic Regression (LR), Multinomial Naive Bayes (NB),
Gradient Boosting (GB), and Random Forest (RF) under a unified TF-IDF feature
representation, stratified 5-fold cross-validation, and a common set of
evaluation metrics.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Requirements](#requirements)
6. [How to Reproduce](#how-to-reproduce)
7. [Results](#results)
8. [Discussion](#discussion)
9. [Limitations](#limitations)
10. [License](#license)
11. [Citation](#citation)

---

## Overview

The automated detection of abusive language on social media platforms remains a
challenging classification problem due to class imbalance, informal language,
and the highly context-dependent nature of abuse. This study contributes an
end-to-end, reproducible pipeline that compares six widely used classical
classifiers on a curated corpus of 20,122 English tweets. The objective is to
provide a controlled empirical baseline against which future work — including
deep learning and transformer-based approaches — can be compared.

The experimental design fixes the feature representation, the cross-validation
splits, and the evaluation metrics across all classifiers, so that observed
differences in performance can be attributed to the classifier itself rather
than to differences in preprocessing or evaluation protocol.

---

## Repository Structure

```
Abusive_Language_Detection/
├── algorithms_against_abuse.py     # Main experimental script
├── dataset.csv                     # Labelled corpus (20,122 records)
├── requirements.txt                # Python dependencies
├── results/
│   └── AAA_algorithm_results.csv   # Per-classifier mean 5-fold results
├── LICENSE                         # MIT License
├── .gitignore
└── README.md
```

---

## Dataset

The dataset (`dataset.csv`) contains **20,122** English-language short text
records with the following schema:

| Column       | Type    | Description                                   |
|--------------|---------|-----------------------------------------------|
| `record_id`  | string  | Unique record identifier (`AAA-#####`)        |
| `text`       | string  | Raw short text (tweet)                        |
| `label`      | integer | Binary label: `0` non-abusive, `1` abusive    |
| `label_name` | string  | Human-readable label                          |
| `language`   | string  | Language of the text (English)                |
| `word_count` | integer | Word count of the text                        |
| `char_count` | integer | Character count of the text                   |

### Class Distribution

| Class        | Count  | Proportion |
|--------------|--------|------------|
| Non-abusive (0) | 17,589 | 87.41 % |
| Abusive (1)     |  2,533 | 12.59 % |
| **Total**       | **20,122** | **100.00 %** |

The corpus exhibits a moderate class imbalance (approximately 6.9 : 1 in favour
of the non-abusive class), which motivates the use of macro-averaged F1 as the
primary evaluation criterion and the application of class-weighted training for
selected classifiers.

The dataset is intended for academic research on abusive-language detection.
It contains text that some readers may find offensive; this is inherent to the
task and does not reflect the views of the authors.

---

## Methodology

### Feature Representation

All classifiers share a single TF-IDF feature representation configured as
follows:

- Lowercasing enabled
- Unicode accent stripping
- Sublinear term-frequency scaling
- Word n-grams of order `(1, 2)`
- Maximum vocabulary size: **10,000** terms
- Minimum document frequency: **2**
- Maximum document frequency: **0.95**

For Gradient Boosting, the sparse TF-IDF matrix is projected onto its top
**200** principal directions via Truncated Singular Value Decomposition (SVD),
which is a standard adaptation for tree-based learners on high-dimensional
sparse input.

### Classifiers

| Symbol | Model                        | Notes                                         |
|--------|------------------------------|-----------------------------------------------|
| SVM    | Linear Support Vector Machine | `class_weight="balanced"`                    |
| KNN    | K-Nearest Neighbours          | k = 5                                        |
| LR     | Logistic Regression           | `class_weight="balanced"`, max_iter = 2000   |
| NB     | Multinomial Naive Bayes       | α = 1.0                                      |
| GB     | Gradient Boosting             | 100 estimators, depth 3, on SVD features     |
| RF     | Random Forest                 | Configuration selected on internal validation set with tuned decision threshold |

### Random Forest Selection

The Random Forest classifier is selected from four candidate configurations
that vary in the number of trees, maximum depth, minimum leaf size, feature
subsampling strategy, and class-weighting scheme. For each configuration, an
internal 20 % validation split (stratified) is held out from the training
portion of each fold, and the decision threshold on the positive-class
probability is tuned on this validation set within the range
`{0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60}` using macro-F1 as the selection
criterion. The configuration with the highest mean validation macro-F1 across
the outer folds is retained, and its per-fold optimal threshold is applied to
the untouched outer test fold.

### Evaluation Protocol

- **Cross-validation:** Stratified 5-fold cross-validation with fixed random
  state (`42`) for full reproducibility.
- **Metrics reported (per fold and averaged):**
  - Accuracy
  - Precision (positive class)
  - Specificity (true-negative rate)
  - Recall (positive class)
  - Macro-averaged F1

Macro-averaged F1 is treated as the primary metric because it weights the two
classes equally and is therefore insensitive to the underlying class imbalance.

---

## Requirements

- Python ≥ 3.9
- Dependencies listed in [`requirements.txt`](requirements.txt):

```
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
```

Install:

```bash
python3 -m pip install -r requirements.txt
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
   python3 -m pip install -r requirements.txt
   ```

3. Run the experiment:

   ```bash
   python3 algorithms_against_abuse.py
   ```

The script performs dataset validation, TF-IDF feature extraction, SVD
reduction for Gradient Boosting, stratified 5-fold cross-validation for all
six classifiers, Random Forest model selection with threshold tuning, and
writes the final aggregated results to `results/AAA_algorithm_results.csv`.

The full run is single-machine and CPU-only; on a modern laptop it completes
in the order of a few minutes.

---

## Results

The following results are the mean of five stratified folds. They are
reproduced verbatim from `results/AAA_algorithm_results.csv`.

| Model | Accuracy | Precision | Specificity | Recall | Macro-F1 |
|-------|---------:|----------:|------------:|-------:|---------:|
| SVM   | 0.9370   | 0.7314    | 0.9580      | 0.7912 | 0.8618   |
| KNN   | 0.8765   | 0.7245    | 0.9985      | 0.0292 | 0.4949   |
| LR    | 0.9416   | 0.7508    | 0.9615      | 0.8034 | 0.8712   |
| NB    | 0.9007   | 0.9419    | 0.9980      | 0.2246 | 0.6544   |
| GB    | 0.9345   | 0.8472    | 0.9847      | 0.5863 | 0.8281   |
| **RF**| **0.9478** | **0.8016** | **0.9721** | **0.7789** | **0.8801** |

**Best model:** Random Forest, with a mean macro-F1 of **0.8801**, a mean
accuracy of **0.9478**, and a mean recall on the abusive class of **0.7789**.

---

## Discussion

Three observations emerge from the comparison.

1. **Macro-F1 is more informative than accuracy under class imbalance.** KNN
   attains an accuracy of 0.8765 despite recovering only 2.9 % of abusive
   examples. Its macro-F1 of 0.4949 exposes this behaviour and would not be
   visible from accuracy alone.

2. **Precision alone can be misleading.** Multinomial Naive Bayes attains the
   highest precision (0.9419) among all models, but its recall of 0.2246 shows
   that it recovers less than a quarter of the abusive class. Its macro-F1 of
   0.6544 places it well below the linear discriminative models.

3. **Random Forest with tuned threshold offers the best overall balance.**
   Random Forest combined with per-fold decision-threshold tuning attains the
   highest macro-F1 (0.8801) and the highest accuracy (0.9478), while
   maintaining a competitive recall on the abusive class (0.7789). Linear
   models (SVM, Logistic Regression) are close behind and remain attractive
   baselines when interpretability and training cost are priorities.

---

## Limitations

- The evaluation is restricted to a single English-language corpus; results
  may not transfer directly to other languages or platforms.
- The feature representation is bag-of-n-grams TF-IDF; contextual embeddings
  or transformer-based encoders are not evaluated here.
- Class imbalance is addressed via class weighting and threshold tuning; no
  resampling techniques are applied.
- Manual annotation of abusive language is inherently subjective; residual
  label noise may act as an upper bound on achievable performance.

---

## License

This project is released under the terms of the [MIT License](LICENSE).

---

## Citation

If you use this code or the accompanying dataset in your research, please cite
this repository:

```bibtex
@misc{aald2025,
  title        = {An Experimental Study of Machine Learning Classifiers for
                  Abusive Language Detection},
  author       = {Namrata G.},
  year         = {2025},
  howpublished = {\url{https://github.com/NamrataG7/Abusive_Language_Detection}}
}
```
