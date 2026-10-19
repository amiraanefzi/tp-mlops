import os
from pathlib import Path

# Évite l'avertissement si Git n'est pas installé (doit précéder l'import de mlflow)
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import mlflow.sklearn
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y


def validate_dataset(X, y):
    if X.empty:
        raise ValueError("Le jeu de données ne contient aucune observation.")

    if len(X) != len(y):
        raise ValueError("Le nombre d'observations et de cibles diffère.")

    if X.isnull().any().any():
        raise ValueError("Des valeurs manquantes ont été détectées.")

    if y.nunique() < 2:
        raise ValueError("La cible doit contenir au moins deux classes.")

    if not all(dtype.kind in "if" for dtype in X.dtypes):
        raise TypeError("Toutes les variables doivent être numériques.")


def load_config(path="config/config.yaml"):
    with Path(path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def build_pipeline(config):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config["model"]["C"],
                    max_iter=config["model"]["max_iter"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


def split_data(X, y, config):
    return train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["seed"],
        stratify=y,
    )


def train_and_evaluate(X_train, X_test, y_train, y_test, config):
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return pipeline, metrics


def run_training(config_path="config/config.yaml"):
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X, y = load_dataset()
    validate_dataset(X, y)

    print(f"Dimensions des variables : {X.shape}")
    print(f"Classes présentes : {[int(c) for c in sorted(y.unique())]}")

    X_train, X_test, y_train, y_test = split_data(X, y, config)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "seed": config["seed"],
                "test_size": config["data"]["test_size"],
                "C": config["model"]["C"],
                "max_iter": config["model"]["max_iter"],
                "dataset": "Breast Cancer Wisconsin (scikit-learn)",
            }
        )

        pipeline, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test, config
        )

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

        print("Métriques :", metrics)


if __name__ == "__main__":
    run_training()