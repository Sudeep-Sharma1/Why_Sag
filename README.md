# PathSense - Pedestrian Risk Intelligence

PathSense is an end-to-end data science and web application project for predicting pedestrian accident risk using environmental, road, traffic, and time-based features.

The project includes:
- a trained machine learning pipeline
- a FastAPI backend
- a browser-based frontend dashboard
- voice support and accessibility controls
- live location and map support
- report-ready EDA assets and notebook outputs

## Project Structure

```text
pathsense/
|-- api/
|   `-- main.py
|-- backend/
|-- dataset/
|   `-- pedestrian_accidents.csv
|-- frontend/
|   |-- index.html
|   |-- index.css
|   |-- index.js
|   `-- vendor/
|       `-- leaflet/
|-- ml/
|   |-- train.py
|   |-- predictor.py
|   |-- report_assets.py
|   |-- report_review.ipynb
|   `-- artifacts/
|       |-- xgboost_risk.pkl
|       |-- random_forest.pkl
|       |-- label_encoders.pkl
|       |-- model_meta.json
|       |-- model_report.json
|       |-- roc_curve.png
|       `-- report_assets/
|-- .env
|-- .env.example
`-- requirements.txt
```

## Main Problem Statement

Predict whether a pedestrian accident scenario is high risk using:
- weather conditions
- lighting conditions
- road type
- road condition
- traffic control
- speed limit
- number of vehicles involved
- time category
- day of week

## Models Used

Primary prediction model used in the application:
- XGBoost

Comparison models used for evaluation:
- Logistic Regression
- Decision Tree
- Random Forest

## Requirements

- Python 3.11+ recommended
- pip
- a modern browser such as Chrome or Edge

Install dependencies:

```powershell
cd c:\PathSense_Complete\PathSense_Complete\PDS\pathsense
pip install -r requirements.txt
```

## Environment Setup

Create or update the `.env` file in the project root.

Example:

```env
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1234567890
```

Notes:
- Twilio is only needed for the SOS feature
- the backend loads `.env` automatically
- weather is currently fetched from the frontend OpenWeather API logic in `frontend/index.html`

## How To Run The Project

### 1. Start the backend

```powershell
cd c:\PathSense_Complete\PathSense_Complete\PDS\pathsense\api
uvicorn main:app --reload --port 8000
```

Backend URLs:
- API root: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### 2. Start the frontend

Use a local server instead of opening the HTML file directly.

```powershell
cd c:\PathSense_Complete\PathSense_Complete\PDS\pathsense\frontend
python -m http.server 5500
```

Then open:

```text
http://localhost:5500/index.html
```

Why this is recommended:
- geolocation works better on `localhost`
- local assets such as Leaflet load correctly
- browser permissions are more reliable

## How To Use The Application

### Prediction flow

1. Open the dashboard
2. Allow location access
3. Wait for auto-detected values to load
4. Fill the road details
5. Optionally use voice input for vehicle and traffic control details
6. Click `Analyse Risk`
7. Review the result, gauge, and spoken summary

### Keyboard shortcuts

- `R` runs prediction
- `Space` reads the latest result
- `S` stops speech

### Accessibility features

- high contrast toggle
- auto voice toggle
- screen reader live regions
- guided voice input
- spoken risk result

### Live location and map

- the project uses local Leaflet assets from `frontend/vendor/leaflet`
- if map tiles fail, the UI falls back gracefully instead of crashing
- the live location section uses browser geolocation

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Run a risk prediction |
| `POST` | `/send-sos` | Send SOS through Twilio |
| `GET` | `/model/stats` | Full model report |
| `GET` | `/model/meta` | Model metadata |
| `GET` | `/options` | Valid categorical options |
| `POST` | `/predict/audio` | Generate audio alert |

### Example prediction request

```json
{
  "weather": "Stormy",
  "lighting": "Dark",
  "road_type": "National Highway",
  "road_condition": "Wet",
  "speed_limit": 100,
  "time_category": "Night",
  "day_of_week": "Saturday",
  "num_vehicles": 4,
  "traffic_control": "Unknown"
}
```

### Example response

```json
{
  "probability": 0.7821,
  "risk_level": "VERY_HIGH",
  "message": "Very high risk of serious or fatal conditions on this segment. Stop and replan if possible.",
  "color": "#ef4444"
}
```

## How To Retrain Models

If you want to retrain the machine learning models:

```powershell
cd c:\PathSense_Complete\PathSense_Complete\PDS\pathsense\ml
python train.py
```

This updates:
- `xgboost_risk.pkl`
- `random_forest.pkl`
- `label_encoders.pkl`
- `model_meta.json`
- `model_report.json`
- `roc_curve.png`

## How To Generate Report Assets

To generate the tables and visuals used for reviews and viva/demo:

```powershell
cd c:\PathSense_Complete\PathSense_Complete\PDS\pathsense
python ml\report_assets.py
```

Generated outputs go to:
- `ml/artifacts/report_assets/tables/`
- `ml/artifacts/report_assets/visuals/`

Generated tables:
- `dataset_description.csv`
- `initial_result_table.csv`
- `model_performance_table.csv`
- `comparative_table.csv`
- `preprocessing_steps.md`
- `key_observations.md`

Generated visuals:
- `eda_01_risk_class_distribution.png`
- `eda_02_weather_vs_risk.png`
- `eda_03_road_type_vs_risk.png`
- `eda_04_speed_limit_distribution.png`
- `eda_05_time_category_vs_risk.png`
- `eda_06_day_of_week_vs_risk.png`
- `eda_07_traffic_control_vs_risk.png`
- `eda_08_feature_importance_xgb.png`

## How To Open The Review Notebook

The class/demo notebook is:
- `ml/report_review.ipynb`

Open it in VS Code or Jupyter-style notebook view and run the cells from top to bottom.

What it shows:
- dataset preview
- dataset info
- null value check
- preprocessing logic
- live EDA plotting code
- initial result table
- model performance table
- comparative table
- key observations

## UI/UX Changes Added

The frontend was improved with:
- stronger hero section and visual hierarchy
- cleaner card layout and reusable CSS classes
- polished forms and result sections
- local Leaflet integration instead of CDN dependency
- map fallback handling
- live location improvements
- notebook/report support for project review presentation

Files mainly updated:
- `frontend/index.html`
- `frontend/index.css`
- `frontend/vendor/leaflet/*`

## Current Model Performance

From the generated comparison table:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.6550 |
| Random Forest | 0.6433 |
| Decision Tree | 0.5833 |
| XGBoost | 0.5183 |

Project note:
- XGBoost is still the primary deployed model in the app
- Logistic Regression currently gives the best holdout accuracy in the generated comparison table

## Git Push Steps

```powershell
cd c:\PathSense_Complete\PathSense_Complete\PDS\pathsense
git status
git add .
git commit -m "Update UI, local map assets, and report review outputs"
git push
```

If pushing a branch for the first time:

```powershell
git push -u origin <branch-name>
```

## Troubleshooting

### Map not loading
- make sure you are using `http://localhost:5500/index.html`
- allow location access
- local Leaflet files should already be used from `frontend/vendor/leaflet`

### Live location not showing
- check browser location permission
- check Windows location settings
- use Chrome or Edge
- run the frontend on `localhost`, not `file://`

### Weather fetch failing
- check OpenWeather API key validity
- weather is currently fetched in the frontend
- if OpenWeather returns `401`, the key or account access is the likely issue

### Voice input shows `no-speech`
- speak clearly after the prompt
- check microphone permission
- use Chrome or Edge
- verify the correct microphone is selected

## Dataset Used

Dataset file:
- `dataset/pedestrian_accidents.csv`

This dataset is used for:
- preprocessing
- model training
- EDA
- report review notebook

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: FastAPI, Uvicorn
- ML: XGBoost, scikit-learn
- Data handling: pandas, NumPy
- Visualisation: matplotlib
- Serialization: joblib, JSON
