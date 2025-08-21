# iris_ml_dag.py
from __future__ import annotations
import logging
import pendulum
from airflow.decorators import dag, task


def _lazy_imports():
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd
    return {
        "load_iris": load_iris,
        "LogisticRegression": LogisticRegression,
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
        "pd": pd,
    }


@dag(
    dag_id="iris_ml_pipeline",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,     # запускать вручную
    catchup=False,
    tags=["ml", "example"],
)
def D_iris_ml_pipeline():

    @task
    def load_data() -> dict:
        mods = _lazy_imports()
        iris = mods["load_iris"]()
        X, y = iris.data, iris.target
        logging.info("Loaded Iris dataset: X=%s, y=%s", X.shape, y.shape)
        return {"X": X.tolist(), "y": y.tolist()}

    @task
    def train_and_evaluate(data: dict) -> dict:
        mods = _lazy_imports()
        pd = mods["pd"]

        X = pd.DataFrame(data["X"])
        y = pd.Series(data["y"])

        X_train, X_test, y_train, y_test = mods["train_test_split"](
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = mods["LogisticRegression"](max_iter=200)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = mods["accuracy_score"](y_test, preds)

        logging.info("Test Accuracy: %.3f", acc)
        return {"accuracy": acc}

    data = load_data()
    _ = train_and_evaluate(data)


D_iris_ml_pipeline()
