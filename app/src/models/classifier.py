from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class SklearnClassifier():
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        self.clf_acc = accuracy_score(df_test[self.target], self.predict(df_test))
        metric = {}
        metric['accuracy'] = self.clf_acc
        return metric

    def predict(self, df: pd.DataFrame):
        return self.clf.predict(df[self.features])