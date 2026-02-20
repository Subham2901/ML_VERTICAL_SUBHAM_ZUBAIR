from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, title: str, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["object(0)", "person(1)"])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "model": name,
        "train": {
            "accuracy": float(accuracy_score(y_train, y_pred_train)),
            "precision": float(precision_score(y_train, y_pred_train, zero_division=0)),
            "recall": float(recall_score(y_train, y_pred_train, zero_division=0)),
            "f1": float(f1_score(y_train, y_pred_train, zero_division=0)),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        },
    }

    # Save confusion matrices
    save_confusion_matrix(
        y_test, y_pred_test,
        title=f"{name} - Confusion Matrix (Test)",
        out_path=RESULTS_DIR / f"cm_{name.replace(' ', '_').lower()}_test.png"
    )
    save_confusion_matrix(
        y_train, y_pred_train,
        title=f"{name} - Confusion Matrix (Train)",
        out_path=RESULTS_DIR / f"cm_{name.replace(' ', '_').lower()}_train.png"
    )

    return metrics


def main():
    # Load processed dataset
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")

    print("Loaded:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)
    print("Class counts train:", {0: int((y_train==0).sum()), 1: int((y_train==1).sum())})
    print("Class counts test :", {0: int((y_test==0).sum()), 1: int((y_test==1).sum())})

    # Pipeline 1: Logistic Regression (with PCA)
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    # Pipeline 2: Random Forest (with PCA)
    # Note: RF doesn't strictly need scaling, but we keep the pipeline consistent + PCA helps a lot.
    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])

    results = []
    results.append(evaluate_model("Logistic Regression PCA95", pipe_lr, X_train, y_train, X_test, y_test))
    results.append(evaluate_model("Random Forest PCA95", pipe_rf, X_train, y_train, X_test, y_test))

    # Save metrics
    out_json = RESULTS_DIR / "metrics_classical.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Saved:")
    print("-", out_json)
    print("-", RESULTS_DIR / "cm_logistic_regression_pca95_test.png")
    print("-", RESULTS_DIR / "cm_random_forest_pca95_test.png")

    # Print summary
    print("\n=== TEST METRICS SUMMARY ===")
    for r in results:
        t = r["test"]
        print(f"{r['model']}: "
              f"acc={t['accuracy']:.4f}, prec={t['precision']:.4f}, rec={t['recall']:.4f}, f1={t['f1']:.4f}")


if __name__ == "__main__":
    main()
