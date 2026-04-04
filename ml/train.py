# -*- coding: utf-8 -*-
"""
Training pipeline derived from Colab `pds.py`: binary High_Risk (Serious/Fatal vs Minor).
Saves XGBoost model, label encoders, and model_report.json under ml/artifacts/.
Also fits LogisticRegression, DecisionTree, RandomForest on the same encoded features for comparison.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_CANDIDATES = [
    ROOT.parent / "dataset" / "pedestrian_accidents.csv",
    ROOT.parent.parent / "dataset" / "pedestrian_accidents.csv",
]


def load_df(path: Path | None = None) -> pd.DataFrame:
    p = path
    if p is None:
        for c in DATA_CANDIDATES:
            if c.is_file():
                p = c
                break
    if p is None or not p.is_file():
        raise FileNotFoundError(
            "Place pedestrian_accidents.csv under pathsense/dataset/. "
            f"Tried: {[str(c) for c in DATA_CANDIDATES]}"
        )
    df = pd.read_csv(p)
    df = df.dropna()
    df.columns = df.columns.str.strip()
    return df


def categorize_time(time_str: str) -> str:
    parts = str(time_str).split(":")
    hour = int(parts[0])
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def build_xy(df: pd.DataFrame):
    df = df.copy()
    df["Time_Category"] = df["Time of Day"].apply(categorize_time)
    df = df.drop(columns=["Time of Day"])
    df["High_Risk"] = df["Accident Severity"].apply(
        lambda x: 1 if x in ["Serious", "Fatal"] else 0
    )
    y = df["High_Risk"]
    X = df.drop(columns=["Accident Severity", "High_Risk", "Pedestrian_Involved"])
    return X, y


def encode_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    X = X.copy()
    label_encoders: dict[str, LabelEncoder] = {}
    for column in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
    return X, label_encoders


def train(
    data_path: Path | None = None,
    save_plots: bool = False,
) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = load_df(data_path)
    X_raw, y = build_xy(df)
    X_encoded, label_encoders = encode_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = float(neg / pos) if pos else 1.0

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    }
    
    xgb = XGBClassifier(**xgb_params, early_stopping_rounds=20)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb = xgb.predict(X_test)
    roc = float(roc_auc_score(y_test, y_prob_xgb))
    acc_xgb = float(accuracy_score(y_test, y_pred_xgb))

    xgb_cv = XGBClassifier(**xgb_params)
    cv_scores = cross_val_score(xgb_cv, X_encoded, y, cv=5, scoring="accuracy")
    cv_mean = float(cv_scores.mean())

    # Comparison models on same encoded split
    lr = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    acc_lr = float(accuracy_score(y_test, lr.predict(X_test)))

    dt = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    acc_dt = float(accuracy_score(y_test, dt.predict(X_test)))

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc_rf = float(accuracy_score(y_test, rf.predict(X_test)))

    feature_names = list(X_encoded.columns)
    importance = {
        k: float(v)
        for k, v in sorted(
            zip(feature_names, xgb.feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
    }

    report = {
        "target": "High_Risk (Serious or Fatal = 1)",
        "n_samples": int(len(df)),
        "features": feature_names,
        "primary_model": "XGBClassifier",
        "xgboost": {
            "roc_auc": roc,
            "accuracy_holdout": acc_xgb,
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_folds": [float(x) for x in cv_scores],
            "confusion_matrix": confusion_matrix(y_test, y_pred_xgb).tolist(),
            "classification_report": classification_report(
                y_test, y_pred_xgb, output_dict=True
            ),
        },
        "comparison_holdout_accuracy": {
            "logistic_regression": acc_lr,
            "decision_tree": acc_dt,
            "random_forest": acc_rf,
        },
        "feature_importance_xgb": importance,
    }

    (ARTIFACTS / "model_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    joblib.dump(xgb, ARTIFACTS / "xgboost_risk.pkl")
    joblib.dump(rf, ARTIFACTS / "random_forest.pkl")
    joblib.dump(label_encoders, ARTIFACTS / "label_encoders.pkl")
    meta = {
        "feature_columns": feature_names,
        "categorical_columns": list(label_encoders.keys()),
        "primary_model_file": "xgboost_risk.pkl",
    }
    (ARTIFACTS / "model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if save_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"XGB (AUC={roc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — High_Risk")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "roc_curve.png", dpi=120)
        plt.close()

    # Console summary (matches pds.py style)
    print("ROC-AUC:", roc)
    print("Accuracy (XGB holdout):", acc_xgb)
    print("\nClassification report (XGB):\n")
    print(classification_report(y_test, y_pred_xgb))
    print("\nConfusion matrix (XGB):\n")
    print(confusion_matrix(y_test, y_pred_xgb))
    print("\nCross-validation scores:", cv_scores)
    print("Average CV Accuracy:", cv_mean)
    print("\nFeature importance (XGB):")
    for k, v in list(importance.items())[:12]:
        print(f"  {k}: {v:.4f}")
    print("\nArtifacts written to", ARTIFACTS)

    return report


def risk_bucket(prob: float) -> str:
    if prob > 0.7:
        return "VERY_HIGH"
    if prob > 0.4:
        return "MODERATE"
    return "LOW"


def main():
    train(save_plots=True)


if __name__ == "__main__":
    main()
