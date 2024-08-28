import json
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


def save_confusion_matrix(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """
    Save a confusion matrix plot.

    :param model: A trained model.
    :param X_test: Test set features.
    :param y_test: Test set target.
    """

    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                              cmap=plt.cm.Blues)
    plt.savefig("ExampleML2/images/confusion_matrix.png")


def save_metrics(metrics: Dict[str, Union[float, int]]):
    """
    Save metrics to a json file.

    :param metrics: A dictionary with metrics.
    """
    with open("ExampleML2/metrics/metrics.json", "w") as fp:
        json.dump(metrics, fp)


def save_predictions(y_test: Union[pd.Series, np.array], y_pred: np.array):
    """Store predictions data for confusion matrix plot."""
    cdf = pd.DataFrame(
        np.column_stack([y_test, y_pred]),
        columns=["true_label", "predicted_label"]
    ).astype(int)
    cdf.to_csv("ExampleML2/predictions/predictions.csv", index=None)


def save_roc_curve(y_test: Union[pd.Series, np.array], y_pred_proba: np.array):
    """
    Save roc curve data.

    :param y_test: Test set target.
    :param y_pred_proba: Predicted probabilities.
    """
    # Calcualte ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    # Store roc curve data
    cdf = pd.DataFrame(np.column_stack([fpr, tpr]),
                       columns=["fpr", "tpr"]).astype(float)
    cdf.to_csv("ExampleML2/metrics/roc_curve.csv", index=None)
