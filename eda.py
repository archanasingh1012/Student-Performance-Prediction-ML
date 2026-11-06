import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, FIGURES_DIR
from data_preprocessing import load_dataset


def run_eda():
    df = load_dataset()

    print("\nDATASET SHAPE:", df.shape)
    print("\nFIRST FIVE ROWS:\n", df.head())
    print("\nDATA TYPES:\n", df.dtypes)
    print("\nMISSING VALUES:\n", df.isnull().sum())
    print("\nDUPLICATE ROWS:", df.duplicated().sum())
    print("\nTARGET DISTRIBUTION:\n", df["risk_level"].value_counts(dropna=False))

    df.describe(include="all").transpose().to_csv(RESULTS_DIR / "descriptive_statistics.csv")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="risk_level")
    plt.title("Distribution of Student Risk Levels")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "risk_level_distribution.png", dpi=300)
    plt.close()

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        plt.figure(figsize=(11, 8))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap of Numeric Features")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
        plt.close()

    print(f"\nEDA outputs saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    run_eda()
