# -*- coding: utf-8 -*-
"""
Load saved XGBoost + label encoders from `train.py` and score route/context rows.
Includes optional short audio alert generation (WAV) for high-risk probability.
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def _risk_bucket(prob: float) -> str:
    if prob > 0.7:
        return "VERY_HIGH"
    if prob > 0.4:
        return "MODERATE"
    return "LOW"


class RiskPredictor:
    def __init__(self, artifacts_dir: Path | None = None):
        d = artifacts_dir or ARTIFACTS
        self.model = joblib.load(d / "xgboost_risk.pkl")
        self.label_encoders: dict = joblib.load(d / "label_encoders.pkl")
        meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
        self.feature_columns: list[str] = meta["feature_columns"]

    def predict_proba_high_risk(self, input_data: dict) -> float:
        """Return P(High_Risk=1) for one row, same schema as `pds.py` `predict_risk`."""
        sample_df = pd.DataFrame([input_data])
        for column in sample_df.columns:
            if column in self.label_encoders:
                le = self.label_encoders[column]
                val = str(sample_df[column].iloc[0])
                if val not in le.classes_:
                    # Unseen category: map to first class to avoid crash (degraded)
                    val = le.classes_[0]
                sample_df[column] = le.transform([val])
        sample_df = sample_df[self.feature_columns]
        return float(self.model.predict_proba(sample_df)[0, 1])

    def alert_message(self, prob: float) -> str:
        b = _risk_bucket(prob)
        if b == "VERY_HIGH":
            return (
                "Very high risk of serious or fatal conditions on this segment. "
                "Stop and replan if possible."
            )
        if b == "MODERATE":
            return "Moderate risk. Proceed with extra caution and stay on safest path."
        return "Low estimated risk for serious or fatal accident under these conditions."


def predict_risk(input_data: dict, artifacts_dir: Path | None = None) -> float:
    """Drop-in replacement for Colab `pds.py` `predict_risk`."""
    return RiskPredictor(artifacts_dir).predict_proba_high_risk(input_data)


def write_alert_wav(
    out_path: Path,
    prob: float,
    sample_rate: int = 22050,
    duration_s: float = 0.35,
) -> None:
    """Higher risk -> higher pitch beep (simple accessibility cue)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    freq = 440 + 660 * min(1.0, max(0.0, prob))
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    wave_data = (0.2 * np.sin(2 * math.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave_data.tobytes())


if __name__ == "__main__":
    # Same scenarios as pds.py (fixed: no undefined risk_prob)
    low = {
        "Weather Conditions": "Clear",
        "Lighting Conditions": "Daylight",
        "Road Type": "Urban Road",
        "Road Condition": "Dry",
        "Speed Limit (km/h)": 40,
        "Time_Category": "Morning",
        "Day of Week": "Tuesday",
        "Number of Vehicles Involved": 1,
        "Traffic Control Presence": "Signals",
    }
    moderate = {
        "Weather Conditions": "Rainy",
        "Lighting Conditions": "Dusk",
        "Road Type": "Urban Road",
        "Road Condition": "Wet",
        "Speed Limit (km/h)": 60,
        "Time_Category": "Evening",
        "Day of Week": "Friday",
        "Number of Vehicles Involved": 2,
        "Traffic Control Presence": "Signs",
    }
    high = {
        "Weather Conditions": "Stormy",
        "Lighting Conditions": "Dark",
        "Road Type": "National Highway",
        "Road Condition": "Wet",
        "Speed Limit (km/h)": 100,
        "Time_Category": "Night",
        "Day of Week": "Saturday",
        "Number of Vehicles Involved": 4,
        "Traffic Control Presence": "Unknown",
    }
    rp = RiskPredictor()
    for name, row in [("Low", low), ("Moderate", moderate), ("High", high)]:
        p = rp.predict_proba_high_risk(row)
        print(f"{name} scenario P(high_risk)={p:.4f} — {rp.alert_message(p)}")
