# -*- coding: utf-8 -*-
"""
PathSense FastAPI Backend
Serves risk prediction and model stats for the frontend dashboard.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ml"
ARTIFACTS = ML_DIR / "artifacts"
env_path = ROOT / ".env"
env_example_path = ROOT / ".env.example"

load_dotenv(env_path)
if not env_path.exists() and env_example_path.exists():
    # Allow local development to work when credentials were added only to the template file.
    load_dotenv(env_example_path, override=False)

import sys
sys.path.insert(0, str(ML_DIR))

from predictor import RiskPredictor, _risk_bucket  # noqa: E402

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PathSense API",
    description="Pedestrian accident risk prediction for safer urban navigation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = RiskPredictor(ARTIFACTS)

# ── schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    weather: str = Field(..., example="Clear")
    lighting: str = Field(..., example="Daylight")
    road_type: str = Field(..., example="Urban Road")
    road_condition: str = Field(..., example="Dry")
    speed_limit: int = Field(..., ge=0, le=200, example=40)
    time_category: str = Field(..., example="Morning")
    day_of_week: str = Field(..., example="Tuesday")
    num_vehicles: int = Field(..., ge=1, le=20, example=1)
    traffic_control: str = Field(..., example="Signals")


class PredictResponse(BaseModel):
    probability: float
    risk_level: Literal["LOW", "MODERATE", "VERY_HIGH"]
    message: str
    color: str


# ── helpers ───────────────────────────────────────────────────────────────────
RISK_COLORS = {
    "LOW": "#22c55e",
    "MODERATE": "#f59e0b",
    "VERY_HIGH": "#ef4444",
}


def _mask_phone(number: str) -> str:
    if len(number) <= 6:
        return number
    return f"{number[:4]}{'X' * max(len(number) - 8, 0)}{number[-4:]}"


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _clean_error_text(message: str) -> str:
    return " ".join(ANSI_ESCAPE_RE.sub("", message).split())


def _friendly_twilio_error(exc: Exception, from_number: str, to_number: str) -> str:
    message = _clean_error_text(str(exc))
    lowered = message.lower()

    if "21659" in message or "is not a twilio phone number" in lowered:
        return (
            "Your Twilio sender number is invalid for SMS. Update "
            f"TWILIO_FROM_NUMBER from {_mask_phone(from_number)} to a real Twilio SMS-capable number."
        )
    if "unverified" in lowered and "trial" in lowered:
        return (
            f"The destination number {_mask_phone(to_number)} is not verified in your Twilio trial account."
        )
    if "'to' and 'from' number cannot be the same" in lowered:
        return "The recipient number cannot be the same as your Twilio sender number."
    return message


def _map_request(req: PredictRequest) -> dict:
    return {
        "Weather Conditions": req.weather,
        "Lighting Conditions": req.lighting,
        "Road Type": req.road_type,
        "Road Condition": req.road_condition,
        "Speed Limit (km/h)": req.speed_limit,
        "Time_Category": req.time_category,
        "Day of Week": req.day_of_week,
        "Number of Vehicles Involved": req.num_vehicles,
        "Traffic Control Presence": req.traffic_control,
    }


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "PathSense API v1.0"}


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    try:
        prob = predictor.predict_proba_high_risk(_map_request(req))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    level = _risk_bucket(prob)
    return PredictResponse(
        probability=round(prob, 4),
        risk_level=level,
        message=predictor.alert_message(prob),
        color=RISK_COLORS[level],
    )


class SOSRequest(BaseModel):
    phone_numbers: list[str] = Field(
        ..., example=["+1234567890", "+1987654321"]
    )
    message: str = Field(
        ..., example="SOS! I need help immediately."
    )


@app.post("/send-sos", tags=["notification"])
def send_sos(req: SOSRequest):
    try:
        from twilio.rest import Client
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Twilio package not installed. "
                "Install it with 'pip install twilio'."
            ),
        ) from exc

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if not account_sid or not auth_token or not from_number:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing Twilio configuration. Please set "
                "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER."
            ),
        )

    client = Client(account_sid, auth_token)
    results = []

    for to_number in req.phone_numbers:
        print(
            "[PathSense SOS] Attempting send",
            {
                "from": _mask_phone(from_number),
                "to": _mask_phone(to_number),
            },
        )
        if not to_number.startswith("+"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Phone numbers must be in international format starting with '+'. "
                    f"Invalid value: {to_number}"
                ),
            )
        try:
            message = client.messages.create(
                body=req.message,
                from_=from_number,
                to=to_number,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Failed to send SMS to {to_number}: "
                    f"{_friendly_twilio_error(exc, from_number, to_number)}"
                ),
            ) from exc
        results.append({
            "to": to_number,
            "sid": message.sid,
            "status": message.status,
        })

    return {"status": "sent", "sent": len(results), "results": results}


@app.get("/model/stats", tags=["model"])
def model_stats():
    """Return model performance report for the dashboard."""
    path = ARTIFACTS / "model_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_report.json not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/model/meta", tags=["model"])
def model_meta():
    path = ARTIFACTS / "model_meta.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_meta.json not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/options", tags=["prediction"])
def get_options():
    """Return valid categorical values per feature (derived from label encoders)."""
    import joblib
    le_dict = joblib.load(ARTIFACTS / "label_encoders.pkl")
    return {col: list(le.classes_) for col, le in le_dict.items()}


@app.post("/predict/audio", tags=["prediction"])
def predict_audio(req: PredictRequest):
    """
    Returns a WAV file — spoken risk summary via pyttsx3 (offline TTS),
    or a pitched beep tone as fallback.
    """
    import tempfile
    from fastapi.responses import FileResponse

    prob  = predictor.predict_proba_high_risk(_map_request(req))
    level = _risk_bucket(prob)
    pct   = round(prob * 100)

    spoken = (
        f"Risk assessment result. "
        f"Conditions: {req.weather} weather, {req.lighting} lighting, "
        f"{req.road_type}, road is {req.road_condition}, "
        f"speed limit {req.speed_limit} kilometres per hour, "
        f"{req.day_of_week} {req.time_category}. "
        f"Result: {level.replace('VERY_HIGH','very high').replace('_',' ')} risk. "
        f"Probability of serious or fatal accident: {pct} percent. "
        f"{predictor.alert_message(prob)}"
    )

    tmp_dir  = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "alert.wav"

    # Try pyttsx3 spoken TTS first
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.save_to_file(spoken, str(wav_path))
        engine.runAndWait()
    except Exception:
        # Fallback: pitched beep whose frequency encodes risk
        from predictor import write_alert_wav
        write_alert_wav(wav_path, prob)

    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename="pathsense_alert.wav",
        headers={
            "X-Risk-Level": level,
            "X-Risk-Probability": str(round(prob, 4)),
            "X-Spoken-Text": spoken[:200],           # for debugging
        },
    )
