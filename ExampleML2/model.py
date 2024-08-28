import json
from typing import Any, Dict, Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, **rfc_params: Dict[str, Any]
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier model and return it.

    :param X_train: Training set features.
    :param y_train: Training set target.
    :param rfc_params: Random Forest Classifier parameters.
    :return: A trained Random Forest Classifier model.
    """
    model = RandomForestClassifier(**rfc_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    float_precision: int = 4,
) -> Dict[str, Union[float, int]]:
    """
    Evaluate a model on a test set and return metrics.

    :param model: A trained model.
    :param X_test: Test set features.
    :param y_test: Test set target.
    :param float_precision: Precision for float values.
    :return: A dictionary with metrics.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return json.loads(
        json.dumps(metrics),
        parse_float=lambda x: round(float(x), float_precision)
    )
