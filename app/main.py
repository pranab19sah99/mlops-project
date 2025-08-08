from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import subprocess
import threading
import os

app = FastAPI(
    title="MLflow API",
    description="FastAPI wrapper for MLflow UI and Tracking API",
    version="1.0.0"
)

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

@app.get("/")
def root():
    """Redirect root to FastAPI docs."""
    return RedirectResponse(url="/docs")

@app.get("/mlflow")
def mlflow_ui():
    """Redirect to MLflow UI."""
    return RedirectResponse(url=f"http://localhost:{MLFLOW_PORT}")
