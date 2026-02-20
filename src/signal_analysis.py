from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def load_processed():
    X_train = np.load(PROCESSED_DIR / "X_train.npy")  # float32
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")  # int64
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def plot_random_waveforms(X, y, n_each=5, prefix="train"):
    rng = np.random.default_rng(RANDOM_SEED)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    pick0 = rng.choice(idx0, size=min(n_each, len(idx0)), replace=False)
    pick1 = rng.choice(idx1, size=min(n_each, len(idx1)), replace=False)

    # Class 0
    plt.figure()
    for i in pick0:
        plt.plot(X[i])
    plt.title(f"Random waveforms (class 0 / object) - {prefix}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    out = FIG_DIR / f"{prefix}_random_waveforms_class0.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    # Class 1
    plt.figure()
    for i in pick1:
        plt.plot(X[i])
    plt.title(f"Random waveforms (class 1 / person) - {prefix}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    out = FIG_DIR / f"{prefix}_random_waveforms_class1.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"✅ Saved random waveform plots to {FIG_DIR}")


def plot_mean_std_envelope(X, y, prefix="train"):
    X0 = X[y == 0]
    X1 = X[y == 1]

    m0, s0 = X0.mean(axis=0), X0.std(axis=0)
    m1, s1 = X1.mean(axis=0), X1.std(axis=0)

    # Class 0 envelope
    plt.figure()
    plt.plot(m0)
    plt.fill_between(np.arange(len(m0)), m0 - s0, m0 + s0, alpha=0.25)
    plt.title(f"Mean ± 1 std (class 0 / object) - {prefix}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    out = FIG_DIR / f"{prefix}_mean_std_class0.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    # Class 1 envelope
    plt.figure()
    plt.plot(m1)
    plt.fill_between(np.arange(len(m1)), m1 - s1, m1 + s1, alpha=0.25)
    plt.title(f"Mean ± 1 std (class 1 / person) - {prefix}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    out = FIG_DIR / f"{prefix}_mean_std_class1.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    # Difference of means (very informative)
    plt.figure()
    plt.plot(m1 - m0)
    plt.title(f"Mean difference: person - object - {prefix}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude difference")
    out = FIG_DIR / f"{prefix}_mean_difference_person_minus_object.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved mean/std envelope plots to {FIG_DIR}")


def compute_simple_features(X):
    """
    Quick global features to see if classes separate at all.
    Not for final model (yet), just EDA.
    """
    # RMS energy per waveform
    rms = np.sqrt(np.mean(X.astype(np.float32) ** 2, axis=1))

    # Peak-to-peak amplitude
    ptp = (X.max(axis=1) - X.min(axis=1)).astype(np.float32)

    # Absolute mean
    abs_mean = np.mean(np.abs(X), axis=1).astype(np.float32)

    return rms, ptp, abs_mean


def plot_feature_histogram(feature, y, name, prefix="train"):
    f0 = feature[y == 0]
    f1 = feature[y == 1]

    plt.figure()
    plt.hist(f0, bins=60, alpha=0.6, label="class 0 (object)")
    plt.hist(f1, bins=60, alpha=0.6, label="class 1 (person)")
    plt.title(f"Feature distribution: {name} - {prefix}")
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.legend()
    out = FIG_DIR / f"{prefix}_feature_hist_{name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_processed()

    print("=== Signal Analysis ===")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Train class counts:", {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())})
    print("Test class counts :", {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum())})

    # 1) Random waveforms
    plot_random_waveforms(X_train, y_train, n_each=6, prefix="train")

    # 2) Mean ± std envelopes + mean difference
    plot_mean_std_envelope(X_train, y_train, prefix="train")

    # 3) Simple feature distributions
    rms, ptp, abs_mean = compute_simple_features(X_train)
    plot_feature_histogram(rms, y_train, "RMS", prefix="train")
    plot_feature_histogram(ptp, y_train, "Peak to Peak", prefix="train")
    plot_feature_histogram(abs_mean, y_train, "Abs Mean", prefix="train")

    print(f" Saved feature distribution plots to {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
