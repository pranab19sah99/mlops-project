# Iris ML API with Monitoring

## Project Overview

This project serves a machine learning Iris classification API, tracked and monitored using Prometheus and Grafana. It includes:

- A FastAPI-based ML API serving predictions using a trained Logistic Regression model.
- MLflow for experiment tracking.
- Prometheus for metrics scraping.
- Grafana for visualizing metrics and building dashboards.
- Automated CI/CD pipeline with GitHub Actions to build and publish Docker images.

---

## Tech Stack

- Python (FastAPI, scikit-learn)
- MLflow (Model tracking)
- Prometheus (Monitoring & metrics scraping)
- Grafana (Visualization & dashboards)
- Docker & Docker Compose (Containerization and orchestration)
- GitHub Actions (CI/CD pipeline)
- Docker Hub (Image registry)

---

## Getting Started

### 1. Clone the repo

```bash
git clone <repo_url>
cd <repo_folder>
````

### 2. Run with Docker Compose (using local build)

```bash
docker-compose up -d
```

---

### Alternative: Pull pre-built images from Docker Hub

The CI/CD pipeline builds and pushes Docker images automatically on every push to the main branch.

To run the project using the published images:

```bash
docker-compose pull
docker-compose up -d
```

This fetches the latest images from Docker Hub and runs them locally.

---

## Usage

### Access Prometheus UI

Open [http://localhost:9090](http://localhost:9090) to explore Prometheus metrics.

### Access Grafana UI

Open [http://localhost:3000](http://localhost:3000) and login with:

* Username: `admin`
* Password: `admin`

---

### Add Prometheus Datasource in Grafana

* Navigate to **Configuration → Data Sources**
* Add new Prometheus datasource
* Set URL to `http://prometheus:9090`
* Save and test connection

---

### Create and Save a Dashboard

* Click the **+** icon → **Dashboard**
* Add new panel → select Prometheus datasource
* Example query: `up{job="iris_api"}`
* Customize and save the dashboard with a name

---

## CI/CD Pipeline

* GitHub Actions workflow automatically:

  * Builds Docker images for the API and other services on push
  * Runs tests if any
  * Pushes images to Docker Hub (`pranabdock/mlops-iris:latest`)
* This ensures the latest working version is always published and ready to pull

---

## Notes

* Docker volumes persist MLflow runs and Grafana data.
* Update `prometheus.yml` to add new scrape targets as needed.

---

Thanks for checking out the project! Feel free to reach out with questions.