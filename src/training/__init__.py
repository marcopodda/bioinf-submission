import numpy as np

from pathlib import Path
from typing import Union

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src import settings
from src.data.dataset import Dataset
from src.training.selector import Selector


def split_data(dataset):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return list(skf.split(dataset.X_train, y=dataset.y_train))
    # train_df = dataset.train_df
    # gram = dataset.test_df.Gram.iloc[0]

    # same_gram = train_df[train_df.Gram==gram]
    # diff_gram_idx = train_df[train_df.Gram!=gram].index.values

    # same_gram_x, same_gram_y = same_gram.index.values, same_gram.Antigen.values

    # skf = StratifiedKFold(n_splits=5, shuffle=True)
    # gram_split = list(skf.split(same_gram_x, y=same_gram_y))

    # splits = []
    # for gram_train, gram_val in gram_split:
    #     train_idx, val_idx = same_gram_x[gram_train], same_gram_x[gram_val]
    #     splits.append((np.hstack([train_idx, diff_gram_idx]), val_idx))

    # return splits


def train_and_test(
    dataset: Dataset,
    tuner: RandomizedSearchCV,
    path_prefix: Union[Path, str] = "",
):
    model = Selector(tuner).train(
        X=dataset.X_train,
        y=dataset.y_train,
        path_prefix=path_prefix,
    )

    metadata_test = dataset.metadata_test
    metadata_test.insert(2, "prob", model.predict(dataset.X_test))

    metadata_predict = dataset.metadata_predict
    metadata_predict.insert(2, "prob", model.predict(dataset.X_predict))

    return metadata_test, metadata_predict
