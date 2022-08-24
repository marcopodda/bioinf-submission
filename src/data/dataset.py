from __future__ import annotations

from abc import ABC, abstractproperty
from typing import List

import joblib
import numpy as np
import pandas as pd

from src import settings
from src.data import FEATURES


def load_dataset(data_dir, proteome_path, species):
    if species != "Benchmark":
        data_dir = data_dir / "evaluation"
        pos = joblib.load(data_dir / "positive.pkl")
        neg = joblib.load(data_dir / "negative.pkl")

        train_pos = pos[~pos.Description.str.contains(species)]
        train_neg = neg.iloc[train_pos.index]
        train_df = pd.concat([train_pos, train_neg], axis=0, ignore_index=True)

        test_pos = pos[pos.Description.str.contains(species)]
        test_neg = neg.iloc[test_pos.index]
        test_df = pd.concat([test_pos, test_neg], axis=0, ignore_index=True)
        assert set(test_df.Seq).intersection(train_df.Seq) == set()

        predict_df = joblib.load(proteome_path)
        predict_df.loc[predict_df.Seq.isin(pos.Seq), "Antigen"] = True
        predict_df = predict_df[predict_df.Loc.isin(settings.OUTER)].reset_index(drop=True)

        assert set(predict_df.Seq).intersection(train_df.Seq) == set()
        return train_df, test_df, predict_df

    if "features" in data_dir.as_posix():
        print("Benchmark for features in not implemented")
        exit(0)

    data_dir = data_dir / "benchmark"
    pos = joblib.load(data_dir / "positive.pkl")
    neg = joblib.load(data_dir / "negative.pkl")

    train_df = pd.concat([pos, neg], axis=0, ignore_index=True)
    predict_df = joblib.load(proteome_path)
    train_df = train_df[~train_df.Seq.isin(predict_df.Seq)].reset_index(drop=True)

    assert set(predict_df.Seq).intersection(train_df.Seq) == set()
    return train_df, predict_df, predict_df


class Dataset(ABC):
    id_column: str = "ID"
    target_column: str = "Antigen"
    meta_columns: List[str] = ["ID", "Species", "Antigen", "Loc", "Description", "Seq"]

    def __init__(self, dim: int, proteome: str, species: str) -> None:
        self.dim = dim
        self.species = species

        self.train_df, self.test_df, self.predict_df = load_dataset(
            self.data_path, proteome, species
        )

    def _get_target(self, df: pd.DataFrame) -> np.ndarray:
        target = df[self.target_column].fillna(False)
        return target.values.astype(np.int64)  # type: ignore

    def _get_data(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.columns].values.astype(np.float32)  # type: ignore

    def _get_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        metadata = df[self.meta_columns]
        return metadata  # type: ignore

    def _get_ids(self, df: pd.DataFrame) -> List[str]:
        ids = df[self.id_column]
        return ids.tolist()

    def shuffle(self):
        self.train_df = self.train_df.sample(frac=1.0).reset_index(drop=True)
        return self

    @property
    def X_train(self) -> np.ndarray:
        return self._get_data(self.train_df)  # type: ignore

    @property
    def y_train(self) -> np.ndarray:
        return self._get_target(self.train_df)  # type: ignore

    @property
    def X_test(self) -> np.ndarray:
        test_df = self.test_df.reset_index(drop=True)
        return self._get_data(test_df)  # type: ignore

    @property
    def X_predict(self) -> np.ndarray:
        predict_df = self.predict_df.reset_index(drop=True)
        return self._get_data(predict_df)  # type: ignore

    @property
    def y_test(self) -> np.ndarray:
        test_df = self.test_df.reset_index(drop=True)
        return self._get_target(test_df)  # type: ignore

    @property
    def id_train(self) -> List[str]:
        return self._get_ids(self.train_df)  # type: ignore

    @property
    def id_test(self) -> List[str]:
        return self._get_ids(self.test_df)  # type: ignore

    @property
    def metadata_train(self):
        return self._get_metadata(self.train_df)  # type: ignore

    @property
    def metadata_test(self):
        test_df = self.test_df.reset_index(drop=True)
        return self._get_metadata(test_df)  # type: ignore

    @property
    def metadata_predict(self):
        predict_df = self.predict_df.reset_index(drop=True)
        return self._get_metadata(predict_df)  # type: ignore

    @property
    def data_path(self):
        return settings.DATA_DIR / self.data_type  # type: ignore

    @abstractproperty
    def data_type(self):
        """Return data type."""

    @abstractproperty
    def columns(self):
        """Return columns."""


class EmbeddingDataset(Dataset):
    @property
    def columns(self):
        return [f"embs{i}" for i in range(1, self.dim + 1)]

    @property
    def data_type(self):
        return "pses"


class FeaturesDataset(Dataset):
    @property
    def columns(self):
        return FEATURES[:]

    @property
    def data_type(self):
        return "features"