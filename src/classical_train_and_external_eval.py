# =========================
# CLASSICAL ML (TRAIN + SAVE) + EXTERNAL EVAL
# - Trains Logistic Regression + PCA95
# - Trains Random Forest + PCA95
# - Saves models (scaler, pca, classifiers)
# - Evaluates on internal test set AND external CSV set
# - Threshold tuning on external set (like CNN)
# =========================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# =========================
# CONFIG
# =========================
RANDOM_SEED = 42

PROJECT_ROOT = Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "reports" / "results"

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_raw"
PERSON_CSV = EXTERNAL_DIR / "person_4k.csv"
OBJECT_CSV = EXTERNAL_DIR / "object_5k.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Must match your dataset build
EXPECTED_WAVEFORM_LEN = 25000

# For classical ML:
# Keep this FALSE if you want to exactly match your earlier PCA results (321 comps for 95%).
# Turn TRUE if you want faster training (but PCA component counts will differ).
USE_DOWNSAMPLE = False
DOWNSAMPLE_FACTOR = 10  # 25000 -> 2500 if enabled

# External evaluation threshold settings
DEFAULT_THRESHOLD = 0.5
THRESHOLDS_TO_SCAN = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]


# =========================
# HELPERS
# =========================
def downsample_avg_pool(X: np.ndarray, factor: int) -> np.ndarray:
    """Average pool downsampling: (N,L)->(N,L/factor)."""
    N, L = X.shape
    if L % factor != 0:
        L_new = (L // factor) * factor
        X = X[:, :L_new]
        L = L_new
    return X.reshape(N, L // factor, factor).mean(axis=2).astype(np.float32)


def load_waveforms(csv_path: Path, expected_len: int = 25000) -> np.ndarray:
    """
    Reads CSV with no header assumption.
    Primary extraction: column R onward (Excel style) => index 17 onward.
    Fallback: last expected_len columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path, header=None)

    # Take from column R (index 17) if exists
    if df.shape[1] >= 18:
        X = df.iloc[:, 17:].to_numpy()
    else:
        X = df.to_numpy()

    # If too wide, take last expected_len columns
    if X.shape[1] > expected_len:
        X = X[:, -expected_len:]

    if X.shape[1] < expected_len:
        raise ValueError(
            f"{csv_path.name}: waveform too short ({X.shape[1]}), expected {expected_len}. "
            f"Check export/columns."
        )

    return X.astype(np.float32)


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return acc, prec, rec, f1, auc, cm


def print_report(title: str, y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    acc, prec, rec, f1, auc, cm = metrics_from_probs(y_true, y_prob, thr)
    print(f"\n=== {title} (thr={thr:.2f}) ===")
    print(f"acc  : {acc:.4f}")
    print(f"prec : {prec:.4f}")
    print(f"rec  : {rec:.4f}")
    print(f"f1   : {f1:.4f}")
    print(f"auc  : {auc:.4f}")
    print("Confusion Matrix [[TN FP],[FN TP]]:")
    print(cm)
    return {
        "threshold": float(thr),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }


def threshold_tuning(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Scan thresholds and return:
    - best by F1
    - best by Youden's J (TPR - FPR)
    """
    best_f1 = None
    best_j = None

    print("\n=== THRESHOLD TUNING (EXTERNAL DATA) ===")
    print("thr | acc  | prec | rec  | f1   | TN  FP  FN  TP")
    print("-" * 55)

    for thr in THRESHOLDS_TO_SCAN:
        acc, prec, rec, f1, auc, cm = metrics_from_probs(y_true, y_prob, thr)
        tn, fp, fn, tp = cm.ravel()

        # Youden's J = TPR - FPR = rec - fp/(fp+tn)
        fpr = fp / (fp + tn + 1e-12)
        j = rec - fpr

        print(f"{thr:>4.2f} | {acc:>4.4f} | {prec:>4.4f} | {rec:>4.4f} | {f1:>4.4f} | {tn:>4d} {fp:>3d} {fn:>3d} {tp:>3d}")

        if (best_f1 is None) or (f1 > best_f1["f1"]):
            best_f1 = {"thr": thr, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "j": j, "cm": cm.tolist()}

        if (best_j is None) or (j > best_j["j"]):
            best_j = {"thr": thr, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "j": j, "cm": cm.tolist()}

    print("\nBest threshold by F1:")
    print(best_f1)
    print("\nBest threshold by Youden's J (TPR-FPR):")
    print(best_j)

    return best_f1, best_j


# =========================
# MAIN
# =========================
def main():
    np.random.seed(RANDOM_SEED)

    # 1) Load internal dataset
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    X_test = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(int)
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(int)

    print("Loaded internal dataset:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Train counts:", {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())})
    print("Test  counts:", {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum())})

    # Optional downsample (classical)
    if USE_DOWNSAMPLE:
        X_train = downsample_avg_pool(X_train, DOWNSAMPLE_FACTOR)
        X_test = downsample_avg_pool(X_test, DOWNSAMPLE_FACTOR)
        print("Downsampled internal:", X_train.shape, X_test.shape)

    # 2) Build pipelines
    # NOTE: StandardScaler is important before PCA + Logistic
    # RandomForest doesn’t strictly need scaling, but we keep same PCA input for fair comparison.
    pca95 = PCA(n_components=0.95, random_state=RANDOM_SEED)

    logreg_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", pca95),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=None
        )),
    ])

    rf_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )),
    ])

    # 3) Train
    print("\nTraining Logistic Regression (PCA95)...")
    logreg_pipe.fit(X_train, y_train)

    print("Training Random Forest (PCA95)...")
    rf_pipe.fit(X_train, y_train)

    # 4) Save models
    logreg_path = MODELS_DIR / "logreg_pca95.joblib"
    rf_path = MODELS_DIR / "rf_pca95.joblib"
    joblib.dump(logreg_pipe, logreg_path)
    joblib.dump(rf_pipe, rf_path)

    print("\n✅ Saved:")
    print("-", logreg_path)
    print("-", rf_path)

    # 5) Internal evaluation
    logreg_prob_test = logreg_pipe.predict_proba(X_test)[:, 1]
    rf_prob_test = rf_pipe.predict_proba(X_test)[:, 1]

    internal_logreg = print_report("INTERNAL TEST - Logistic+PCA95", y_test, logreg_prob_test, DEFAULT_THRESHOLD)
    internal_rf = print_report("INTERNAL TEST - RF+PCA95", y_test, rf_prob_test, DEFAULT_THRESHOLD)

    # 6) Load external dataset
    X_person = load_waveforms(PERSON_CSV, expected_len=EXPECTED_WAVEFORM_LEN)
    X_obj = load_waveforms(OBJECT_CSV, expected_len=EXPECTED_WAVEFORM_LEN)
    y_person = np.ones((X_person.shape[0],), dtype=int)
    y_obj = np.zeros((X_obj.shape[0],), dtype=int)

    X_ext = np.vstack([X_obj, X_person]).astype(np.float32)
    y_ext = np.concatenate([y_obj, y_person]).astype(int)

    print("\nLoaded external dataset:")
    print("object :", X_obj.shape, "label=0")
    print("person :", X_person.shape, "label=1")
    print("TOTAL  :", X_ext.shape, y_ext.shape)
    print("Counts :", {0: int((y_ext == 0).sum()), 1: int((y_ext == 1).sum())})

    if USE_DOWNSAMPLE:
        X_ext = downsample_avg_pool(X_ext, DOWNSAMPLE_FACTOR)
        print("Downsampled external:", X_ext.shape)

    # 7) External evaluation
    logreg_prob_ext = logreg_pipe.predict_proba(X_ext)[:, 1]
    rf_prob_ext = rf_pipe.predict_proba(X_ext)[:, 1]

    external_logreg = print_report("EXTERNAL TEST - Logistic+PCA95", y_ext, logreg_prob_ext, DEFAULT_THRESHOLD)
    external_rf = print_report("EXTERNAL TEST - RF+PCA95", y_ext, rf_prob_ext, DEFAULT_THRESHOLD)

    # Threshold tuning on external (optional but recommended)
    print("\n--- Logistic external threshold tuning ---")
    logreg_best_f1, logreg_best_j = threshold_tuning(y_ext, logreg_prob_ext)

    print("\n--- RF external threshold tuning ---")
    rf_best_f1, rf_best_j = threshold_tuning(y_ext, rf_prob_ext)

    # 8) Save report JSON
    report = {
        "use_downsample": USE_DOWNSAMPLE,
        "downsample_factor": DOWNSAMPLE_FACTOR if USE_DOWNSAMPLE else None,
        "internal": {
            "logistic_pca95": internal_logreg,
            "rf_pca95": internal_rf,
        },
        "external": {
            "logistic_pca95": external_logreg,
            "rf_pca95": external_rf,
        },
        "external_threshold_tuning": {
            "logistic_best_by_f1": logreg_best_f1,
            "logistic_best_by_youdenJ": logreg_best_j,
            "rf_best_by_f1": rf_best_f1,
            "rf_best_by_youdenJ": rf_best_j,
        },
        "paths": {
            "logreg_model": str(logreg_path),
            "rf_model": str(rf_path),
            "person_csv": str(PERSON_CSV),
            "object_csv": str(OBJECT_CSV),
        }
    }

    out_path = OUT_DIR / "classical_internal_external_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Saved report JSON:", out_path)


if __name__ == "__main__":
    main()
