
# =========================
# COMPLETE PASTABLE CNN CODE (EDITED)
# - Fixes metrics issue (no more "compile_metrics" only)
# - Prints + saves: loss, acc, prec, rec, f1, auc
# - Still saves plots, confusion matrix, model diagram
# =========================

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# -------------------------
# CONFIG
# -------------------------
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 40

# CNN input handling
USE_DOWNSAMPLE = True
DOWNSAMPLE_FACTOR = 10      # 25000 -> 2500 (trims if not divisible)
USE_PER_SAMPLE_ZNORM = True # z-normalize each waveform independently

# Paths (run from repo root)
PROJECT_ROOT = Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

for d in [RESULTS_DIR, FIG_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -------------------------
# HELPERS
# -------------------------
def downsample_avg_pool(X: np.ndarray, factor: int) -> np.ndarray:
    """
    Average pooling downsampling.
    X: (N, L) -> (N, floor(L/factor))
    """
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


def plot_history(history: keras.callbacks.History, prefix: str = "cnn"):
    # Accuracy
    plt.figure()
    if "acc" in history.history:
        plt.plot(history.history["acc"], label="train_acc")
    if "val_acc" in history.history:
        plt.plot(history.history["val_acc"], label="val_acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_accuracy.png", dpi=150)
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{prefix}_loss.png", dpi=150)
    plt.close()


def save_confusion(y_true, y_prob, threshold: float = 0.5, prefix: str = "cnn"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["object(0)", "person(1)"])
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{prefix}_confusion_matrix.png", dpi=200)
    plt.close()


def build_cnn(input_len: int) -> keras.Model:
    inp = keras.Input(shape=(input_len, 1), name="input_layer")

    x = layers.Conv1D(32, 9, padding="same", name="conv1d")(inp)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.ReLU(name="re_lu")(x)
    x = layers.MaxPool1D(2, name="max_pooling1d")(x)

    x = layers.Conv1D(64, 7, padding="same", name="conv1d_1")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.ReLU(name="re_lu_1")(x)
    x = layers.MaxPool1D(2, name="max_pooling1d_1")(x)

    x = layers.Conv1D(128, 5, padding="same", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.ReLU(name="re_lu_2")(x)

    x = layers.GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="dense_1")(x)

    return keras.Model(inp, out, name="cnn1d")


# -------------------------
# MAIN
# -------------------------
def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load dataset
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)  # (N, 25000)
    X_test  = np.load(PROCESSED_DIR / "X_test.npy").astype(np.float32)
    y_train = np.load(PROCESSED_DIR / "y_train.npy").astype(np.float32)  # (N,)
    y_test  = np.load(PROCESSED_DIR / "y_test.npy").astype(np.float32)

    print("Loaded:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Class counts train:", {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())})
    print("Class counts test :", {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum())})

    # Downsample
    if USE_DOWNSAMPLE:
        X_train = downsample_avg_pool(X_train, DOWNSAMPLE_FACTOR)
        X_test  = downsample_avg_pool(X_test, DOWNSAMPLE_FACTOR)
        print("Downsampled:", X_train.shape, X_test.shape)

    # Per-sample z-norm
    if USE_PER_SAMPLE_ZNORM:
        X_train = per_sample_standardize(X_train)
        X_test  = per_sample_standardize(X_test)
        print("Applied per-sample z-normalization.")

    # Add channel dimension: (N, L, 1)
    X_train = X_train[..., None]
    X_test  = X_test[..., None]
    input_len = int(X_train.shape[1])

    # Train/Val split (from train)
    val_frac = 0.15
    n_val = int(len(X_train) * val_frac)
    X_val, y_val = X_train[:n_val], y_train[:n_val]
    X_tr,  y_tr  = X_train[n_val:], y_train[n_val:]

    # Class weights (mild imbalance)
    n0 = float((y_tr == 0).sum())
    n1 = float((y_tr == 1).sum())
    class_weight = {
        0: (n0 + n1) / (2.0 * n0),
        1: (n0 + n1) / (2.0 * n1),
    }
    print("Class weights:", class_weight)

    # tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .shuffle(4000, seed=RANDOM_SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Model
    model = build_cnn(input_len)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.Precision(name="prec"),
            keras.metrics.Recall(name="rec"),
        ],
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "cnn1d_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(RESULTS_DIR / "cnn1d_training_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(RESULTS_DIR / "tensorboard" / "cnn1d")),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training curves
    plot_history(history, prefix="cnn1d")

    # -------------------------
    # FIXED EVALUATION (NO "compile_metrics" ISSUE)
    # -------------------------
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Compute robust metrics
    acc = accuracy_score(y_test.astype(int), y_pred)
    prec = precision_score(y_test.astype(int), y_pred, zero_division=0)
    rec = recall_score(y_test.astype(int), y_pred, zero_division=0)
    f1 = f1_score(y_test.astype(int), y_pred, zero_division=0)
    auc = roc_auc_score(y_test.astype(int), y_prob)

    # Loss from evaluate (reliable)
    loss = float(model.evaluate(test_ds, verbose=0)[0])

    print("\n=== CNN TEST METRICS (threshold=0.5) ===")
    print(f"loss : {loss:.4f}")
    print(f"acc  : {acc:.4f}")
    print(f"prec : {prec:.4f}")
    print(f"rec  : {rec:.4f}")
    print(f"f1   : {f1:.4f}")
    print(f"auc  : {auc:.4f}")

    # Confusion matrix
    save_confusion(y_test, y_prob, threshold=0.5, prefix="cnn1d")

    # Save model
    final_model_path = MODEL_DIR / "cnn1d_final.keras"
    model.save(final_model_path)

    # Save metrics JSON properly
    payload = {
        "use_downsample": USE_DOWNSAMPLE,
        "downsample_factor": DOWNSAMPLE_FACTOR if USE_DOWNSAMPLE else None,
        "use_per_sample_znorm": USE_PER_SAMPLE_ZNORM,
        "input_len": int(input_len),
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "test_metrics": {
            "loss": float(loss),
            "acc": float(acc),
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "auc": float(auc),
        },
        "history_keys": list(history.history.keys()),
    }
    with open(RESULTS_DIR / "metrics_cnn.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Plot model architecture image (requires graphviz + pydot)
    # Mac install:
    #   brew install graphviz
    #   pip install pydot
    try:
        plot_model = tf.keras.utils.plot_model
        plot_model(
            model,
            to_file=str(FIG_DIR / "cnn1d_model.png"),
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=150,
        )
        print("\n✅ Saved model diagram to:", FIG_DIR / "cnn1d_model.png")
    except Exception as e:
        print("\n⚠️ Could not generate model diagram (install graphviz + pydot). Error:")
        print(e)

    print("\n✅ Saved outputs:")
    print("-", RESULTS_DIR / "cnn1d_training_log.csv")
    print("-", RESULTS_DIR / "metrics_cnn.json")
    print("-", FIG_DIR / "cnn1d_accuracy.png")
    print("-", FIG_DIR / "cnn1d_loss.png")
    print("-", RESULTS_DIR / "cnn1d_confusion_matrix.png")
    print("-", final_model_path)


if __name__ == "__main__":
    main()