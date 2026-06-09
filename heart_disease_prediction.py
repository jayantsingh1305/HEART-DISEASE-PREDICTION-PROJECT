from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


OUTPUT_DIR = Path(".")


def load_data(file_path: str = "heart.csv") -> pd.DataFrame:
    return pd.read_csv(file_path)


def print_basic_summary(df: pd.DataFrame) -> None:
    print(df.head())
    print("------")
    print(df.columns)
    print("------")
    print(df.shape)
    print("------")
    df.info()
    print("------")
    print(df.describe())
    print("------")
    print(df.duplicated().sum())
    print("------")
    print(df.isnull().sum())
    print("------")


def save_numeric_histograms(df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    numeric_columns = ["Age", "RestingBP", "Cholesterol", "MaxHR"]

    for axis, column in zip(axes.flat, numeric_columns):
        sns.histplot(df[column], kde=True, ax=axis)
        axis.set_title(column)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


def save_class_balance_plot(df: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="HeartDisease", ax=axis)
    axis.set_title("HeartDisease Distribution")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


def save_categorical_plots(df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_specs = [
        ("Sex", "HeartDisease", "Sex vs HeartDisease"),
        ("ChestPainType", "HeartDisease", "ChestPainType vs HeartDisease"),
        ("FastingBS", "HeartDisease", "FastingBS vs HeartDisease"),
        ("ST_Slope", "HeartDisease", "ST_Slope vs HeartDisease"),
    ]

    for axis, (x_column, hue_column, title) in zip(axes.flat, plot_specs):
        sns.countplot(data=df, x=x_column, hue=hue_column, ax=axis)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


def save_relationship_plots(df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df, x="HeartDisease", y="Cholesterol", ax=axes[0])
    sns.violinplot(data=df, x="HeartDisease", y="Age", ax=axes[1])
    sns.heatmap(df.corr(numeric_only=True), annot=True, ax=axes[2])
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(df, drop_first=True).astype(int)
    numerical_columns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    scaler = StandardScaler()
    encoded[numerical_columns] = scaler.fit_transform(encoded[numerical_columns])
    return encoded


def main() -> None:
    df = load_data()

    print_basic_summary(df)
    save_numeric_histograms(df, OUTPUT_DIR / "eda_histograms.png")
    save_class_balance_plot(df, OUTPUT_DIR / "heart_disease_balance.png")
    save_categorical_plots(df, OUTPUT_DIR / "categorical_plots.png")
    save_relationship_plots(df, OUTPUT_DIR / "relationship_plots.png")

    print("------")
    print(df["Cholesterol"].value_counts())
    print("------")

    cholesterol_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
    print(cholesterol_mean)
    print("------")

    df_encoded = preprocess_data(df)
    print(df_encoded.head())
    print("Script executed successfully.")


if __name__ == "__main__":
    main()