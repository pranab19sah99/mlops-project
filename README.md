# MLOps Project - CI/CD with FastAPI, MLflow, and GitHub Actions

This project demonstrates an MLOps pipeline using FastAPI, MLflow, Docker, and GitHub Actions for CI/CD, model retraining, and logging/monitoring.

---

## Features

- **FastAPI** application that wraps MLflow Tracking Server and UI
- **MLflow** for experiment tracking and model registry
- **GitHub Actions** for scheduled retraining
- **Docker Compose** for local development and deployment
- **Logging & Monitoring** built-in
- **Automatic documentation** at `/docs` (FastAPI Swagger UI)

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   └── main.py           # FastAPI wrapper for MLflow
├── src/
│   ├── __init__.py
│   └── retrain.py        # Model retraining script
├── scripts/              # (Optional) helper scripts
├── requirements.txt
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── retrain.yml   # GitHub Actions workflow
└── README.md
```

---

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- GitHub account (for CI/CD)
- MLflow installed (`pip install mlflow`)

---

## Setup & Run Locally

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run with Docker Compose
```bash
docker-compose up --build
```
This starts:
- **FastAPI** at `http://localhost:8000`
- **MLflow UI** at `http://localhost:5000`

### 5. Run without Docker
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
MLflow will be served as a subprocess.

---

## Example FastAPI `main.py`
```python
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
    return RedirectResponse(url="/docs")

@app.get("/mlflow")
def mlflow_ui():
    return RedirectResponse(url=f"http://localhost:{MLFLOW_PORT}")
```

---

## Example `src/retrain.py`
```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def retrain_model():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("iris_classification")

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Retrained model with accuracy: {acc:.4f}")

if __name__ == "__main__":
    retrain_model()
```

---

## Docker Compose (`docker-compose.yml`)
```yaml
version: "3.9"

services:
  fastapi:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_PORT=5000
      - BACKEND_STORE_URI=sqlite:///mlflow.db
      - ARTIFACT_ROOT=/mlruns
    volumes:
      - ./mlruns:/mlruns
    depends_on:
      - mlflow

  mlflow:
    image: python:3.10
    working_dir: /app
    command: >
      bash -c "pip install mlflow && 
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri sqlite:///mlflow.db
               --default-artifact-root /mlruns"
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
```

---

## GitHub Actions Workflow (`.github/workflows/retrain.yml`)
```yaml
name: Scheduled Retraining

on:
  schedule:
    - cron: "0 0 * * 0" # Every Sunday at midnight UTC
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run retraining
        run: |
          python src/retrain.py
```

---

## Access

- **FastAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow UI:** [http://localhost:5000](http://localhost:5000)

---

## License
MIT License
