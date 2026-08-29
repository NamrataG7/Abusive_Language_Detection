import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

import os

DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset.csv"
)

RANDOM_STATE = 42
N_SPLITS = 5

MAX_FEATURES = 10000
MIN_DF = 2
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)

# Reduced representation for Gradient Boosting
GB_SVD_COMPONENTS = 200

# Probability thresholds evaluated during RF validation
RF_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATASET)

if "text" not in df.columns:
    raise ValueError("Dataset must contain a 'text' column.")

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

df["text"] = df["text"].fillna("").astype(str)
df["label"] = pd.to_numeric(
    df["label"],
    errors="raise"
).astype(int)


# ============================================================
# DATASET VALIDATION
# ============================================================

print("\nDataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

EXPECTED_TOTAL = 20122
EXPECTED_NON_ABUSIVE = 17589
EXPECTED_ABUSIVE = 2533

if len(df) != EXPECTED_TOTAL:
    raise ValueError(
        f"Expected {EXPECTED_TOTAL} records, found {len(df)}."
    )

if (df["label"] == 0).sum() != EXPECTED_NON_ABUSIVE:
    raise ValueError(
        f"Expected {EXPECTED_NON_ABUSIVE} non-abusive records."
    )

if (df["label"] == 1).sum() != EXPECTED_ABUSIVE:
    raise ValueError(
        f"Expected {EXPECTED_ABUSIVE} abusive records."
    )

print("\nDataset validation passed.")

X_text = df["text"]
y = df["label"].values

print("\nAbusive:", (y == 1).sum())
print("Non-abusive:", (y == 0).sum())
print("Abusive %:", round((y == 1).mean() * 100, 2))


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    ngram_range=NGRAM_RANGE
)

X = vectorizer.fit_transform(X_text)

print("TF-IDF matrix shape:", X.shape)


# ============================================================
# SVD FOR GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 70)
print("PREPARING FEATURES FOR GRADIENT BOOSTING")
print("=" * 70)

svd_components = min(
    GB_SVD_COMPONENTS,
    X.shape[1] - 1
)

svd = TruncatedSVD(
    n_components=svd_components,
    random_state=RANDOM_STATE
)

X_gb = svd.fit_transform(X)

print(
    "Original TF-IDF features:",
    X.shape[1]
)

print(
    "GB SVD features:",
    X_gb.shape[1]
)

print(
    "Explained variance:",
    round(
        svd.explained_variance_ratio_.sum(),
        4
    )
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Specificity": specificity,
        "Recall": recall,
        "Macro-F1": macro_f1
    }


# ============================================================
# MODELS
# ============================================================

models = {

    "SVM": SVC(
        kernel="linear",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),

    "LR": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "NB": MultinomialNB(
        alpha=1.0
    ),

    "GB": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE
    )
}


# ============================================================
# CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# RESULTS
# ============================================================

all_results = {}


# ============================================================
# RUN STANDARD MODELS
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X_text, y),
        start=1
    ):

        print(
            f"\nStarting Fold {fold}..."
        )

        if model_name == "GB":
            X_train = X_gb[train_idx]
            X_test = X_gb[test_idx]
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        fold_results.append(metrics)

        print(
            f"Fold {fold}: "
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"Precision={metrics['Precision']:.4f}, "
            f"Specificity={metrics['Specificity']:.4f}, "
            f"Recall={metrics['Recall']:.4f}, "
            f"Macro-F1={metrics['Macro-F1']:.4f}"
        )

    results_df = pd.DataFrame(
        fold_results
    )

    mean_results = results_df.mean()

    all_results[model_name] = mean_results

    print("\n" + "-" * 70)
    print(f"{model_name} MEAN RESULTS")
    print("-" * 70)

    for metric in [
        "Accuracy",
        "Precision",
        "Specificity",
        "Recall",
        "Macro-F1"
    ]:
        print(
            f"{metric:<12}: "
            f"{mean_results[metric]:.4f}"
        )


# ============================================================
# RANDOM FOREST VALIDATION
# ============================================================
#
# RF configuration is selected using the training portion of
# each outer fold. The outer test fold remains untouched.
#
# The final selected RF is reported simply as "RF".
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST VALIDATION")
print("=" * 70)

rf_configurations = [

    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": {
            0: 1,
            1: 2
        }
    },

    {
        "n_estimators": 300,
        "max_depth": 40,
        "min_samples_leaf": 2,
        "max_features": "log2",
        "class_weight": {
            0: 1,
            1: 2
        }
    }
]


# ------------------------------------------------------------
# First determine the best RF configuration.
# ------------------------------------------------------------

configuration_scores = []

for config_number, config in enumerate(
    rf_configurations,
    start=1
):

    validation_scores = []

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        skf.split(X_text, y),
        start=1
    ):

        inner_train_idx, validation_idx = train_test_split(
            train_idx,
            test_size=0.20,
            stratify=y[train_idx],
            random_state=RANDOM_STATE + fold
        )

        rf = RandomForestClassifier(
            **config,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        rf.fit(
            X[inner_train_idx],
            y[inner_train_idx]
        )

        validation_prob = rf.predict_proba(
            X[validation_idx]
        )[:, 1]

        best_score = -1

        for threshold in RF_THRESHOLDS:

            validation_pred = (
                validation_prob >= threshold
            ).astype(int)

            score = f1_score(
                y[validation_idx],
                validation_pred,
                average="macro",
                zero_division=0
            )

            if score > best_score:
                best_score = score

        validation_scores.append(
            best_score
        )

    mean_score = np.mean(
        validation_scores
    )

    configuration_scores.append(
        (
            mean_score,
            config_number
        )
    )


configuration_scores.sort(
    reverse=True
)

best_config_number = configuration_scores[0][1]

best_rf_config = rf_configurations[
    best_config_number - 1
]

print(
    "\nSelected Random Forest configuration:"
)

print(
    best_rf_config
)


# ============================================================
# FINAL RF CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RF")
print("=" * 70)

rf_fold_results = []

selected_thresholds = []

for fold, (
    train_idx,
    test_idx
) in enumerate(
    skf.split(X_text, y),
    start=1
):

    print(
        f"\nStarting Fold {fold}..."
    )

    # --------------------------------------------------------
    # Inner validation set for threshold selection
    # --------------------------------------------------------

    inner_train_idx, validation_idx = train_test_split(
        train_idx,
        test_size=0.20,
        stratify=y[train_idx],
        random_state=RANDOM_STATE + fold
    )

    rf_validation = RandomForestClassifier(
        **best_rf_config,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_validation.fit(
        X[inner_train_idx],
        y[inner_train_idx]
    )

    validation_probability = rf_validation.predict_proba(
        X[validation_idx]
    )[:, 1]

    best_threshold = 0.50
    best_validation_f1 = -1

    for threshold in RF_THRESHOLDS:

        validation_prediction = (
            validation_probability >= threshold
        ).astype(int)

        validation_f1 = f1_score(
            y[validation_idx],
            validation_prediction,
            average="macro",
            zero_division=0
        )

        if validation_f1 > best_validation_f1:

            best_validation_f1 = validation_f1
            best_threshold = threshold

    selected_thresholds.append(
        best_threshold
    )

    # --------------------------------------------------------
    # Train final RF on complete outer training fold
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        **best_rf_config,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(
        X[train_idx],
        y[train_idx]
    )

    test_probability = rf.predict_proba(
        X[test_idx]
    )[:, 1]

    y_pred = (
        test_probability >= best_threshold
    ).astype(int)

    metrics = calculate_metrics(
        y[test_idx],
        y_pred
    )

    rf_fold_results.append(
        metrics
    )

    print(
        f"Fold {fold}: "
        f"Accuracy={metrics['Accuracy']:.4f}, "
        f"Precision={metrics['Precision']:.4f}, "
        f"Specificity={metrics['Specificity']:.4f}, "
        f"Recall={metrics['Recall']:.4f}, "
        f"Macro-F1={metrics['Macro-F1']:.4f}"
    )


# ============================================================
# RF MEAN
# ============================================================

rf_mean = (
    pd.DataFrame(
        rf_fold_results
    )
    .mean()
)

all_results["RF"] = rf_mean


print("\n" + "-" * 70)
print("RF MEAN RESULTS")
print("-" * 70)

for metric in [
    "Accuracy",
    "Precision",
    "Specificity",
    "Recall",
    "Macro-F1"
]:

    print(
        f"{metric:<12}: "
        f"{rf_mean[metric]:.4f}"
    )

print(
    "\nSelected thresholds:",
    selected_thresholds
)

print(
    "Mean threshold:",
    round(
        np.mean(selected_thresholds),
        2
    )
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

print("\n\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

summary = pd.DataFrame(
    all_results
).T

summary = summary[
    [
        "Accuracy",
        "Precision",
        "Specificity",
        "Recall",
        "Macro-F1"
    ]
]

print(
    summary.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = (
    summary["Macro-F1"]
    .idxmax()
)

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(
    "Model:",
    best_model
)

print(
    "Macro-F1:",
    f"{summary.loc[best_model, 'Macro-F1']:.4f}"
)

print(
    "Accuracy:",
    f"{summary.loc[best_model, 'Accuracy']:.4f}"
)

print(
    "Precision:",
    f"{summary.loc[best_model, 'Precision']:.4f}"
)

print(
    "Specificity:",
    f"{summary.loc[best_model, 'Specificity']:.4f}"
)

print(
    "Recall:",
    f"{summary.loc[best_model, 'Recall']:.4f}"
)


# ============================================================
# SAVE ONLY ONE RESULTS FILE
# ============================================================

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "AAA_algorithm_results.csv"
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

summary.to_csv(
    OUTPUT_FILE
)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nExperiment completed.")