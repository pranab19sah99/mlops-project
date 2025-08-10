# Iris ML API with Monitoring

## Project Overview

This project serves a machine learning Iris classification API with monitoring and visualization using Prometheus and Grafana. It features:

- FastAPI-based ML API serving Iris predictions.
- MLflow for experiment tracking.
- Prometheus for metrics collection.
- Grafana for dashboards and visualization.
- GitHub Actions CI/CD pipeline for automated builds and Docker image publishing.

---

## Tech Stack

- Python (FastAPI, scikit-learn)
- MLflow
- Prometheus
- Grafana
- Docker & Docker Compose
- GitHub Actions
- Docker Hub

---

## Quickstart: Pull and Run Pre-built Images

### 1. Pull the latest Docker images

From the project root directory (where `docker-compose.yml` is), run:

```bash
docker-compose pull
````

This downloads the latest published images for all services.

---

### 2. Start the containers

Run all services in detached mode:

```bash
docker-compose up -d
```

---

### 3. Verify running containers

Check that containers are up with:

```bash
docker ps
```

You should see containers for:

* `iris_api` (ML API)
* `mlflow_ui` (MLflow tracking)
* `prometheus` (metrics scraping)
* `grafana` (dashboard UI)

---

### 4. Access the services in your browser

| Service       | URL                                            | Description            |
| ------------- | ---------------------------------------------- | ---------------------- |
| Iris API      | [http://localhost:5000](http://localhost:5000) | ML prediction API      |
| MLflow UI     | [http://localhost:5001](http://localhost:5001) | Experiment tracking UI |
| Prometheus UI | [http://localhost:9090](http://localhost:9090) | Metrics exploration UI |
| Grafana UI    | [http://localhost:3000](http://localhost:3000) | Dashboards and visuals |

Login to Grafana with default credentials:

* Username: `admin`
* Password: `admin`

---

## Adding Prometheus as a Data Source in Grafana

1. After logging into Grafana, go to the **Configuration (gear icon) → Data Sources**.
2. Click **Add data source**.
3. Select **Prometheus**.
4. Set the URL to:

   ```
   http://prometheus:9090
   ```

   (This uses the Docker service name, allowing Grafana to connect inside the Docker network.)
5. Click **Save & Test** to verify the connection.

---

## Creating a Dashboard in Grafana

1. Click the **+** icon in the left sidebar and choose **Dashboard**.

2. Click **Add new panel**.

3. Select **Prometheus** as the data source.

4. Enter a Prometheus query, for example:

   ```
   up{job="iris_api"}
   ```

   This query checks if the Iris API is up (returns 1 if up, 0 if down).

5. Customize the visualization type (Graph, Stat, Gauge, etc.) as you prefer.

6. Click **Apply**.

7. Save the dashboard with a meaningful name by clicking the **Save dashboard** icon.

---

## GitHub Actions CI/CD Pipeline

* **Workflow triggers:**
  The GitHub Actions pipeline runs automatically on every push to the **main** branch.

* **What it does:**

  * Builds Docker images for the API and related services.
  * Runs tests (if any).
  * Pushes the built images to Docker Hub under the repository `pranabdock/mlops-iris:latest`.

* **Benefits:**
  This automated pipeline ensures your Docker images on Docker Hub are always up-to-date with the latest code changes, so you can easily pull and run the latest version locally or in other environments.

---

## Notes

* The `docker-compose.yml` mounts local directories as volumes to persist logs, MLflow runs, and Grafana data.

* Modify `prometheus/prometheus.yml` to add or change scrape targets.

* If you want to build and run the images locally instead of pulling, you can run:

  ```bash
  docker-compose build
  docker-compose up -d
  ```

* To stop and remove all running containers:

  ```bash
  docker-compose down
  ```

---

Thanks for exploring this project! Feel free to ask if you want help setting up Grafana dashboards or GitHub Actions workflows.
