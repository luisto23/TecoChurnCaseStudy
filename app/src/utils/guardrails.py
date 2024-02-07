import functools
import os

import pandas as pd

from src.utils.store import CaseStudyStore

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

class ModelTrainWarning(Exception):
    pass

def validate_stage_outputs(assert_func):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            assert_func()
            return result

        return _wrapper

    return _decorator

def _validate_prediction_results():
    try:
        df = CaseStudyStore().get_predictions("results.csv")
        if set(df.columns) != {"cutomerID", "churn"}:
            raise ModelTrainWarning(
                "predict_model.py must create output csv with schema: `cutomerID,churn`"
            )
    except FileNotFoundError as exc:
        raise ModelTrainWarning(
            "predict_model.py must create output csv of predictions in evaluation/results.csv"
        ) from exc


def _validate_evaluation_metrics():
    try:
        metrics = CaseStudyStore().get_metrics("metrics.json")
        if not (isinstance(metrics, dict) and len(metrics) > 0):
            raise ModelTrainWarning(
                "train_model.py must create output dictionary with at least 1 evaluation metric"
            )
    except FileNotFoundError as exc:
        raise ModelTrainWarning(
            "train_model.py must create output json of evaluation metrics in evaluation/metrics.json"
        ) from exc

validate_prediction_results = validate_stage_outputs(_validate_prediction_results)
validate_evaluation_metrics = validate_stage_outputs(_validate_evaluation_metrics)