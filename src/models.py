"""
Model training and evaluation utilities for the Bus Service Reliability pipeline.

Mirrors the model definitions in 04_model_training.ipynb and the metric computation
in 05_evaluation.ipynb, refactored into reusable functions.
"""

import time
import builtins
from pyspark.sql import DataFrame
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)


def get_evaluators():
    """Return the standard set of evaluators used across all models."""
    return {
        "accuracy": MulticlassClassificationEvaluator(
            labelCol="target", predictionCol="prediction", metricName="accuracy"
        ),
        "precision": MulticlassClassificationEvaluator(
            labelCol="target", predictionCol="prediction", metricName="weightedPrecision"
        ),
        "recall": MulticlassClassificationEvaluator(
            labelCol="target", predictionCol="prediction", metricName="weightedRecall"
        ),
        "f1": MulticlassClassificationEvaluator(
            labelCol="target", predictionCol="prediction", metricName="f1"
        ),
        "roc_auc": BinaryClassificationEvaluator(
            labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        ),
    }


def build_models():
    """Return the 3 classifiers compared in this project, with fixed configs."""
    return {
        "Logistic Regression": LogisticRegression(
            featuresCol="features", labelCol="target", maxIter=10
        ),
        "Random Forest": RandomForestClassifier(
            featuresCol="features", labelCol="target", numTrees=20, seed=42, maxBins=128
        ),
        "Decision Tree": DecisionTreeClassifier(
            featuresCol="features", labelCol="target", maxBins=128
        ),
    }


def evaluate_model(predictions: DataFrame, model_name: str, evaluators: dict) -> dict:
    """Compute all 5 classification metrics for one model's predictions."""
    return {
        "Model": model_name,
        "Accuracy": evaluators["accuracy"].evaluate(predictions),
        "Precision": evaluators["precision"].evaluate(predictions),
        "Recall": evaluators["recall"].evaluate(predictions),
        "F1-score": evaluators["f1"].evaluate(predictions),
        "ROC-AUC": evaluators["roc_auc"].evaluate(predictions),
    }


def time_and_score(estimator, model_name: str, train_data: DataFrame, test_data: DataFrame, f1_evaluator) -> dict:
    """
    Train a model, time the fit, and compute Model Efficiency (F1-score per
    second of training) as required by the assignment brief.
    """
    start = time.time()
    fitted_model = estimator.fit(train_data)
    duration = time.time() - start

    preds = fitted_model.transform(test_data)
    f1 = f1_evaluator.evaluate(preds)
    efficiency = f1 / duration

    return {
        "Model": model_name,
        "Training Time (s)": builtins.round(duration, 2),
        "F1-score": builtins.round(f1, 4),
        "Model Efficiency (F1/sec)": builtins.round(efficiency, 4),
    }