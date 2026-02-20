import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path.cwd()

MODEL_PATH = PROJECT_ROOT / "models" / "cnn1d_best.keras"

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_raw"
PERSON_CSV = EXTERNAL_DIR / "person_4k.csv"
OBJECT_CSV = EXTERNAL_DIR / "object_5k.csv"

OUT_DIR = PROJECT_ROOT / "reports" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# must match training
EXPECTED_WAVEFORM_LEN = 25000
USE_DOWNSAMPLE = True
DOWNSAMPLE_FACTOR = 10      # 25000 -> 2500
USE_PER_SAMPLE_ZNORM = True

# default threshold (we will tune and pick best)
THRESHOLD = 0.5

# =========================
# HELPERS
# =========================
def downsample_avg_pool(X: np.ndarray, factor: int) -> np.ndarray:
    """Average pooling downsampling: (N, L) -> (N, floor(L/factor))"""
    N, L = X.shape
    if L % factor != 0:
        L_new = (L // factor) * factor
        X = X[:, :L_new]
        L = L_new
    return X.reshape(N, L // factor, factor).mean(axis=2).astype(np.float32)


def per_sample_standardize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-normalize each waveform independently (mean 0, std 1)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return ((X - mu) / (sd + eps)).astype(np.float32)


def load_waveforms(csv_path: Path, expected_len: int = 25000) -> np.ndarray:
    """
    Reads CSV with no header assumption.
    Primary extraction: column R onward (Excel style) => index 17 onward.
    Fallback: last expected_len columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path, header=None)

    # Try taking from column R (index 17)
    if df.shape[1] >= 18:
        X = df.iloc[:, 17:].to_numpy()
    else:
        X = df.to_numpy()

    # If too wide, take last expected_len (safe fallback)
    if X.shape[1] > expected_len:
        X = X[:, -expected_len:]

    if X.shape[1] < expected_len:
        raise ValueError(
            f"{csv_path.name}: waveform too short ({X.shape[1]}), expected {expected_len}. "
            f"Check export/columns."
        )

    return X.astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    """Compute classification metrics at a given threshold."""
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "threshold": float(thr),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),  # [[TN, FP],[FN, TP]]
    }


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[list[dict], dict, dict]:
    """
    Evaluate multiple thresholds and pick:
    - best_f1: threshold that maximizes F1
    - best_youden: threshold that maximizes (TPR - FPR) (Youden's J)
    Returns:
      rows, best_f1_row, best_youden_row
    """
    thresholds = np.linspace(0.05, 0.95, 19)

    rows: list[dict] = []
    best_f1_row = None
    best_youden_row = None
    best_f1 = -1.0
    best_j = -1e9

    for thr in thresholds:
        m = compute_metrics(y_true, y_prob, float(thr))
        rows.append(m)

        # best F1
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_f1_row = m

        # Youden's J = TPR - FPR
        tn, fp = m["confusion_matrix"][0]
        fn, tp = m["confusion_matrix"][1]
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_youden_row = {**m, "youden_j": float(j)}

    assert best_f1_row is not None
    assert best_youden_row is not None
    return rows, best_f1_row, best_youden_row


def print_threshold_table(rows: list[dict]) -> None:
    print("\n=== THRESHOLD TUNING (EXTERNAL DATA) ===")
    print("thr | acc   | prec  | rec   | f1   | TN   FP   FN   TP")
    print("------------------------------------------------------")
    for r in rows:
        tn, fp = r["confusion_matrix"][0]
        fn, tp = r["confusion_matrix"][1]
        print(
            f"{r['threshold']:.2f} | "
            f"{r['acc']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} | {r['f1']:.4f} | "
            f"{tn:4d} {fp:4d} {fn:4d} {tp:4d}"
        )


# =========================
# MAIN
# =========================
def main():
    print("Loading model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load external data
    X_person = load_waveforms(PERSON_CSV, expected_len=EXPECTED_WAVEFORM_LEN)
    X_obj = load_waveforms(OBJECT_CSV, expected_len=EXPECTED_WAVEFORM_LEN)

    y_person = np.ones((X_person.shape[0],), dtype=np.int32)
    y_obj = np.zeros((X_obj.shape[0],), dtype=np.int32)

    X_ext = np.vstack([X_obj, X_person])
    y_ext = np.concatenate([y_obj, y_person])

    print("\nExternal loaded:")
    print("object :", X_obj.shape, "label=0")
    print("person :", X_person.shape, "label=1")
    print("TOTAL  :", X_ext.shape, y_ext.shape)
    print("Class counts:", {0: int((y_ext == 0).sum()), 1: int((y_ext == 1).sum())})

    # Preprocess exactly like training
    if USE_DOWNSAMPLE:
        X_ext = downsample_avg_pool(X_ext, DOWNSAMPLE_FACTOR)
        print("Downsampled:", X_ext.shape)

    if USE_PER_SAMPLE_ZNORM:
        X_ext = per_sample_standardize(X_ext)
        print("Applied per-sample z-norm.")

    # Add channel dimension
    X_ext = X_ext[..., None]  # (N, 2500, 1)

    # Predict probabilities
    y_prob = model.predict(X_ext, batch_size=256, verbose=0).ravel()

    # AUC is threshold-independent (good for domain shift analysis)
    auc = roc_auc_score(y_ext, y_prob) if len(np.unique(y_ext)) == 2 else float("nan")

    # Loss on external set (optional)
    ext_loss = float(model.evaluate(X_ext, y_ext, verbose=0)[0])

    # ---- Baseline threshold=0.5 (your current report) ----
    base = compute_metrics(y_ext, y_prob, THRESHOLD)

    print(f"\n=== EXTERNAL TEST METRICS (threshold={THRESHOLD:.2f}) ===")
    print(f"loss : {ext_loss:.4f}")
    print(f"acc  : {base['acc']:.4f}")
    print(f"prec : {base['prec']:.4f}")
    print(f"rec  : {base['rec']:.4f}")
    print(f"f1   : {base['f1']:.4f}")
    print(f"auc  : {auc:.4f}")
    print("\nConfusion Matrix [[TN FP],[FN TP]]:")
    print(np.array(base["confusion_matrix"]))

    # ---- Threshold tuning ----
    rows, best_f1, best_youden = tune_thresholds(y_ext, y_prob)
    print_threshold_table(rows)

    print("\nBest threshold by F1:")
    print(
        f"thr={best_f1['threshold']:.2f} | acc={best_f1['acc']:.4f} "
        f"prec={best_f1['prec']:.4f} rec={best_f1['rec']:.4f} f1={best_f1['f1']:.4f} "
        f"cm={best_f1['confusion_matrix']}"
    )

    print("\nBest threshold by Youden's J (TPR-FPR):")
    j = best_youden.get("youden_j", None)
    print(
        f"thr={best_youden['threshold']:.2f} | acc={best_youden['acc']:.4f} "
        f"prec={best_youden['prec']:.4f} rec={best_youden['rec']:.4f} f1={best_youden['f1']:.4f} "
        f"J={j:.4f} cm={best_youden['confusion_matrix']}"
    )

    # Save JSON report (includes threshold sweep)
    report = {
        "model_path": str(MODEL_PATH),
        "external_files": {"person": str(PERSON_CSV), "object": str(OBJECT_CSV)},
        "expected_waveform_len": EXPECTED_WAVEFORM_LEN,
        "use_downsample": USE_DOWNSAMPLE,
        "downsample_factor": DOWNSAMPLE_FACTOR if USE_DOWNSAMPLE else None,
        "use_per_sample_znorm": USE_PER_SAMPLE_ZNORM,
        "external_loss": ext_loss,
        "auc": float(auc),
        "baseline_threshold": float(THRESHOLD),
        "baseline_metrics": base,
        "threshold_sweep": rows,
        "best_threshold_f1": best_f1,
        "best_threshold_youden": best_youden,
        "n_samples": int(len(y_ext)),
        "class_counts": {"0": int((y_ext == 0).sum()), "1": int((y_ext == 1).sum())},
    }

    out_path = OUT_DIR / "external_test_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Saved external evaluation to:", out_path)


if __name__ == "__main__":
    main()
