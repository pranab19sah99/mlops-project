import os
import pickle
import logging
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("models", exist_ok=True)

def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

def make_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

def train_and_log():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg": LogisticRegression(max_iter=200, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    mlflow.set_experiment("iris-classification")

    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = make_pipeline(model)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("model_name", name)
            if name == "rf":
                mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", float(acc))
            mlflow.sklearn.log_model(pipeline, artifact_path=f"{name}_model")

            local_path = f"models/{name}.pkl"
            with open(local_path, "wb") as f:
                pickle.dump(pipeline, f)

            logger.info(f"{name}: accuracy={acc:.4f}, saved to {local_path}")
            results[name] = {"pipeline": pipeline, "accuracy": acc, "path": local_path}

    # Select best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model_info = results[best_name]
    best_local_path = f"models/{best_name}.pkl"
    if best_local_path != best_model_info["path"]:
        with open(best_local_path, "wb") as f:
            pickle.dump(best_model_info["pipeline"], f)
    logger.info(f"Best model: {best_name} with accuracy {results[best_name]['accuracy']:.4f}")
    logger.info(f"Saved best model to {best_local_path}")

if __name__ == "__main__":
    train_and_log()
