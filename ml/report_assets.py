from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from train import build_xy, categorize_time, encode_features, load_df


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
REPORT_DIR = ARTIFACTS / "report_assets"
TABLES_DIR = REPORT_DIR / "tables"
VISUALS_DIR = REPORT_DIR / "visuals"


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)


def save_bar_plot(series: pd.Series, title: str, ylabel: str, out_path: Path, color: str = "#15803d") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    series.plot(kind="bar", ax=ax, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_grouped_risk_plot(df: pd.DataFrame, feature: str, out_path: Path, normalize: bool = False) -> None:
    plot_df = (
        df.groupby([feature, "High_Risk_Label"]).size().unstack(fill_value=0)
    )
    if normalize:
        plot_df = plot_df.div(plot_df.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", ax=ax, color=["#22c55e", "#ef4444"])
    ax.set_title(f"{feature} vs Risk Level")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_xlabel("")
    ax.legend(title="Risk")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_speed_histogram(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    low = df[df["High_Risk_Label"] == "Low Risk"]["Speed Limit (km/h)"]
    high = df[df["High_Risk_Label"] == "High Risk"]["Speed Limit (km/h)"]
    ax.hist(low, bins=20, alpha=0.7, label="Low Risk", color="#22c55e")
    ax.hist(high, bins=20, alpha=0.7, label="High Risk", color="#ef4444")
    ax.set_title("Speed Limit Distribution by Risk Level")
    ax.set_xlabel("Speed Limit (km/h)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_feature_importance(model_report: dict, out_path: Path) -> None:
    importance = pd.Series(model_report["feature_importance_xgb"]).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="#15803d")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def model_table(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / pos) if pos else 1.0

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=1),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        ),
    }

    rows: list[dict] = []
    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "Model": name,
                "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "F1 Score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "ROC-AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def generate() -> None:
    ensure_dirs()

    raw_df = load_df()
    working_df = raw_df.copy()
    working_df["Time_Category"] = working_df["Time of Day"].apply(categorize_time)
    working_df["High_Risk"] = working_df["Accident Severity"].apply(lambda x: 1 if x in ["Serious", "Fatal"] else 0)
    working_df["High_Risk_Label"] = working_df["High_Risk"].map({0: "Low Risk", 1: "High Risk"})

    X_raw, y = build_xy(raw_df)
    X_encoded, label_encoders = encode_features(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    report = json.loads((ARTIFACTS / "model_report.json").read_text(encoding="utf-8"))
    performance_df = model_table(X_train, X_test, y_train, y_test)
    performance_df.to_csv(TABLES_DIR / "model_performance_table.csv", index=False)

    comparison_df = performance_df.copy()
    best_accuracy = float(comparison_df["Accuracy"].max())
    comparison_df["Accuracy Gap From Best"] = (best_accuracy - comparison_df["Accuracy"]).round(4)
    comparison_df.to_csv(TABLES_DIR / "comparative_table.csv", index=False)

    dataset_description = pd.DataFrame(
        [
            {"Metric": "Dataset Source", "Value": "pathsense/dataset/pedestrian_accidents.csv"},
            {"Metric": "Rows", "Value": len(raw_df)},
            {"Metric": "Columns", "Value": raw_df.shape[1]},
            {"Metric": "Categorical Features", "Value": len(label_encoders)},
            {"Metric": "Numeric Features", "Value": len(X_raw.columns) - len(label_encoders)},
            {"Metric": "Target Variable", "Value": "High_Risk (Serious/Fatal = 1, Minor = 0)"},
        ]
    )
    dataset_description.to_csv(TABLES_DIR / "dataset_description.csv", index=False)

    initial_results = pd.DataFrame(
        [
            {"Metric": "Training Rows", "Value": len(X_train)},
            {"Metric": "Testing Rows", "Value": len(X_test)},
            {"Metric": "Low Risk Samples", "Value": int((working_df["High_Risk"] == 0).sum())},
            {"Metric": "High Risk Samples", "Value": int((working_df["High_Risk"] == 1).sum())},
            {"Metric": "Missing Values Removed", "Value": int(raw_df.isna().sum().sum())},
            {"Metric": "Engineered Feature", "Value": "Time_Category derived from Time of Day"},
        ]
    )
    initial_results.to_csv(TABLES_DIR / "initial_result_table.csv", index=False)

    write_markdown(
        TABLES_DIR / "preprocessing_steps.md",
        """
        # Data Preprocessing Steps

        1. Loaded the pedestrian accident dataset from the project dataset folder.
        2. Removed rows with missing values using `dropna()`.
        3. Trimmed column names to prevent whitespace mismatches.
        4. Converted `Time of Day` into a categorical feature called `Time_Category`.
        5. Built the binary target `High_Risk`, where Serious and Fatal cases are mapped to 1.
        6. Dropped non-model columns: `Accident Severity`, `High_Risk`, `Pedestrian_Involved`, and the raw `Time of Day`.
        7. Label-encoded all categorical predictors to make them model-ready.
        8. Split the encoded data into stratified training and testing sets with an 80/20 ratio.
        9. Trained Logistic Regression, Decision Tree, Random Forest, and XGBoost on the same split for comparison.
        """,
    )

    risk_counts = working_df["High_Risk_Label"].value_counts().reindex(["Low Risk", "High Risk"])
    save_bar_plot(risk_counts, "Risk Class Distribution", "Count", VISUALS_DIR / "eda_01_risk_class_distribution.png")
    save_grouped_risk_plot(working_df, "Weather Conditions", VISUALS_DIR / "eda_02_weather_vs_risk.png")
    save_grouped_risk_plot(working_df, "Road Type", VISUALS_DIR / "eda_03_road_type_vs_risk.png")
    save_speed_histogram(working_df, VISUALS_DIR / "eda_04_speed_limit_distribution.png")
    save_grouped_risk_plot(working_df, "Time_Category", VISUALS_DIR / "eda_05_time_category_vs_risk.png")
    save_grouped_risk_plot(working_df, "Day of Week", VISUALS_DIR / "eda_06_day_of_week_vs_risk.png", normalize=True)
    save_grouped_risk_plot(working_df, "Traffic Control Presence", VISUALS_DIR / "eda_07_traffic_control_vs_risk.png")
    save_feature_importance(report, VISUALS_DIR / "eda_08_feature_importance_xgb.png")

    top_weather = (
        working_df.groupby("Weather Conditions")["High_Risk"].mean().sort_values(ascending=False).head(3)
    )
    top_road = (
        working_df.groupby("Road Type")["High_Risk"].mean().sort_values(ascending=False).head(2)
    )
    avg_speed = working_df.groupby("High_Risk_Label")["Speed Limit (km/h)"].mean().round(2)
    best_model = performance_df.iloc[0]

    observations = f"""
    # Key Observations From Visuals

    1. The dataset contains {int((working_df['High_Risk'] == 1).sum())} high-risk rows and {int((working_df['High_Risk'] == 0).sum())} low-risk rows, showing that the target is moderately imbalanced toward high-risk outcomes.
    2. The weather categories with the highest high-risk share are: {", ".join(f"{idx} ({val:.2%})" for idx, val in top_weather.items())}.
    3. The road types with the highest high-risk share are: {", ".join(f"{idx} ({val:.2%})" for idx, val in top_road.items())}.
    4. High-risk scenarios have a higher average speed limit ({avg_speed['High Risk']} km/h) than low-risk scenarios ({avg_speed['Low Risk']} km/h).
    5. Lighting conditions, road condition, and number of vehicles appear as the strongest XGBoost features, indicating that environment and traffic density matter more than road type alone.
    6. The best holdout-accuracy model in the current comparison is {best_model['Model']} with accuracy {best_model['Accuracy']:.4f} and ROC-AUC {best_model['ROC-AUC']:.4f}.
    """
    write_markdown(TABLES_DIR / "key_observations.md", observations)

    summary = f"""
    # Report Asset Summary

    Generated outputs:

    - `tables/dataset_description.csv`
    - `tables/initial_result_table.csv`
    - `tables/model_performance_table.csv`
    - `tables/comparative_table.csv`
    - `tables/preprocessing_steps.md`
    - `tables/key_observations.md`
    - `visuals/eda_01_risk_class_distribution.png`
    - `visuals/eda_02_weather_vs_risk.png`
    - `visuals/eda_03_road_type_vs_risk.png`
    - `visuals/eda_04_speed_limit_distribution.png`
    - `visuals/eda_05_time_category_vs_risk.png`
    - `visuals/eda_06_day_of_week_vs_risk.png`
    - `visuals/eda_07_traffic_control_vs_risk.png`
    - `visuals/eda_08_feature_importance_xgb.png`
    """
    write_markdown(REPORT_DIR / "README.md", summary)


if __name__ == "__main__":
    generate()
