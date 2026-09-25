"""
Exploratory Data Analysis for the CWRU feature dataset.
Generates figures for the technical report.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "features.csv")
FIG_DIR = os.path.join(BASE_DIR, "report", "figures")
STATS_FILE = os.path.join(BASE_DIR, "report", "class_statistics.csv")

FEATURES = ["rms", "crest_factor", "kurtosis", "skewness"]
CLASS_ORDER = ["Normal", "InnerRace", "OuterRace", "Ball"]


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_class_distribution(df):
    plt.figure(figsize=(7, 5))
    counts = df["label"].value_counts().reindex(CLASS_ORDER)
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.title("Class distribution")
    plt.ylabel("Number of windows")
    plt.xlabel("Class")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 200, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "class_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def compute_class_statistics(df):
    stats = df.groupby("label")[FEATURES].agg(["mean", "std", "min", "max"])
    stats = stats.round(4)
    stats.to_csv(STATS_FILE)
    print(f"Saved: {STATS_FILE}")
    return stats


def plot_feature_boxplots(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feat in enumerate(FEATURES):
        sns.boxplot(
            data=df,
            x="label",
            y=feat,
            order=CLASS_ORDER,
            ax=axes[i],
            hue="label",
            palette="viridis",
            legend=False,
        )
        axes[i].set_title(feat)
        axes[i].set_xlabel("")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "feature_boxplots.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_matrix(df):
    plt.figure(figsize=(6, 5))
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "feature_correlation.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_pairplot(df):
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    g = sns.pairplot(
        sample,
        vars=FEATURES,
        hue="label",
        hue_order=CLASS_ORDER,
        palette="viridis",
        plot_kws={"s": 8, "alpha": 0.5},
    )
    path = os.path.join(FIG_DIR, "feature_pairplot.png")
    g.savefig(path, dpi=100)
    plt.close()
    print(f"Saved: {path}")


def main():
    ensure_dirs()
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Rows: {len(df)}")
    print(f"Classes: {df['label'].unique().tolist()}")

    print("\nGenerating figures...")
    plot_class_distribution(df)
    stats = compute_class_statistics(df)
    plot_feature_boxplots(df)
    plot_correlation_matrix(df)
    plot_pairplot(df)

    print("\nClass statistics:")
    print(stats)
    print("\nEDA done.")


if __name__ == "__main__":
    main()
