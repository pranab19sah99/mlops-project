# app/main.py
import os
import json
import logging
import sqlite3
import subprocess
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from src.utils import load_model, validate_input

# --------------------
# Logging
# --------------------
LOG_FILE = os.getenv("LOG_FILE", "logs/predictions.log")
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --------------------
# SQLite setup
# --------------------
DB_PATH = os.getenv("DB_PATH", "logs/predictions.db")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_json TEXT,
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --------------------
# Prometheus metrics
# --------------------
REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
ERROR_COUNT = Counter("prediction_errors_total", "Total number of prediction errors")

# --------------------
# Model load
# --------------------
MODEL = None
try:
    MODEL = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="Iris MLOps API + MLflow",
              description="Unified API for Iris predictions and MLflow tracking",
              version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# MLflow server thread
# --------------------
MLFLOW_HOST = "0.0.0.0"
MLFLOW_PORT = int(os.getenv("MLFLOW_PORT", 5000))
BACKEND_STORE_URI = os.getenv("BACKEND_STORE_URI", "sqlite:///mlflow.db")
ARTIFACT_ROOT = os.getenv("ARTIFACT_ROOT", "./mlruns")

def start_mlflow():
    subprocess.run([
        "mlflow", "server",
        "--host", MLFLOW_HOST,
        "--port", str(MLFLOW_PORT),
        "--backend-store-uri", BACKEND_STORE_URI,
        "--default-artifact-root", ARTIFACT_ROOT
    ])

@app.on_event("startup")
def launch_mlflow():
    threading.Thread(target=start_mlflow, daemon=True).start()

# --------------------
# API Models
# --------------------
class FeaturesRequest(BaseModel):
    features: conlist(conlist(float, min_items=4, max_items=4), min_items=1) | conlist(float, min_items=4, max_items=4)

# --------------------
# Routes
# --------------------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/mlflow")
def mlflow_ui():
    return RedirectResponse(url=f"http://localhost:{MLFLOW_PORT}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: FeaturesRequest):
    REQUEST_COUNT.inc()
    try:
        features = req.features
        X = validate_input(features)
        preds = MODEL.predict(X).tolist()

        log_entry = {"input": features, "prediction": preds}
        logger.info(json.dumps(log_entry))

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO requests (input_json, prediction) VALUES (?, ?)", (json.dumps(features), json.dumps(preds)))
        conn.commit()
        conn.close()

        return {"prediction": preds}
    except Exception as e:
        ERROR_COUNT.inc()
        logger.exception("Prediction error")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
