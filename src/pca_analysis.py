from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load only training data
    X_train = np.load(PROCESSED_DIR / "X_train.npy")

    print("Loaded X_train:", X_train.shape)

    # Standardize (VERY IMPORTANT)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("Data standardized.")

    # PCA without limiting components first
    pca = PCA()
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Plot explained variance
    plt.figure()
    plt.plot(explained)
    plt.title("Explained Variance Ratio per Component")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Variance Ratio")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_explained_variance.png", dpi=200)
    plt.close()

    # Plot cumulative variance
    plt.figure()
    plt.plot(cumulative)
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.axhline(0.90)
    plt.axhline(0.95)
    plt.axhline(0.99)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_cumulative_variance.png", dpi=200)
    plt.close()

    # Print component counts
    comp_90 = np.argmax(cumulative >= 0.90) + 1
    comp_95 = np.argmax(cumulative >= 0.95) + 1
    comp_99 = np.argmax(cumulative >= 0.99) + 1

    print("\n=== PCA SUMMARY ===")
    print(f"Components for 90% variance: {comp_90}")
    print(f"Components for 95% variance: {comp_95}")
    print(f"Components for 99% variance: {comp_99}")
    print("Total original features:", X_train.shape[1])


if __name__ == "__main__":
    main()
