from __future__ import annotations
from typing import Union
from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from src import utils
from src.training.utils import CVResults


log = utils.get_logger(__name__)


class TrainedModel:
    @classmethod
    def load(cls, path: str) -> TrainedModel:
        model = joblib.load(path)
        return cls(model)

    def __init__(self, model: RandomizedSearchCV) -> None:
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.model.predict_proba(X)
        return probas[:, 1]

    def save(self, prefix):
        joblib.dump(self.model, f"{prefix}model.pkl")
        joblib.dump(self.model.best_params_, f"{prefix}best_params.pkl")


class Selector:
    def __init__(
        self,
        tuner: RandomizedSearchCV,
    ) -> None:
        self.tuner = tuner

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        path_prefix: Union[str, Path] = "",
    ) -> TrainedModel:
        self.tuner.fit(X, y)

        self.save_results(
            cv_path=f"{path_prefix}cv.pkl",
            results_path=f"{path_prefix}cv_results.csv",
        )

        model = TrainedModel(self.tuner)
        model.save(path_prefix)

        return model

    def save_results(
        self,
        cv_path: Union[str, Path],
        results_path: Union[str, Path],
    ) -> None:
        cv_results = self.tuner.cv_results_
        cv_results = CVResults(cv_results, scoring="auroc")
        cv_results.save(str(cv_path))
        cv_summary = cv_results.get_cv_summary()
        cv_summary.round(6).to_csv(results_path)
