from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from scipy import stats


class LayerGenerator:
    def __init__(self, *args) -> None:
        # assert len(args) > 1, print("At least two dimensions!")

        self.num_layers = len(args) - 1 # type:ignore
        self.gens = []

        for i in range(1, self.num_layers + 1):
            gen = stats.randint(high=args[i - 1], low=args[i])  # type:ignore
            self.gens.append(gen)

        self.prob_thres = [0.0] + [0.5] * (self.num_layers - 1)
        self.prob = stats.uniform(loc=0, scale=1)  # type:ignore

    def rvs(self, random_state: Optional[int] = None) -> Tuple:
        hidden_layer_sizes = []

        for prob, gen in zip(self.prob_thres, self.gens):
            add_value = self.prob.rvs(random_state=random_state)
            if add_value > prob:
                hidden_layer_sizes.append(gen.rvs(random_state=random_state))

        return tuple(hidden_layer_sizes)


class CVResults:
    def __init__(
        self,
        cv_results: Dict[str, Any],
        scoring: str = "score",
    ) -> None:
        self.scoring = scoring
        self.params = cv_results.pop("params")
        self.mean_fit_time = cv_results.pop("mean_fit_time", None)
        self.std_fit_time = cv_results.pop("std_fit_time", None)
        self.mean_score_time = cv_results.pop("mean_score_time", None)
        self.std_score_time = cv_results.pop("std_score_time", None)
        self.keys = [k for k in cv_results if not k.startswith("param")]

        for key in self.keys:
            setattr(self, key, cv_results[key])

        self.best_index = np.argmin(getattr(self, f"rank_test_{scoring}"))
        self.metrics = ["_".join(k.split("_")[2:]) for k in self.keys]
        self.num_splits = self.infer_num_splits()

    def infer_num_splits(self) -> int:
        keys = [k.split("_")[0] for k in self.keys if k.startswith("split")]
        splits = [int(k.split("split")[1]) for k in keys]
        return max(splits) + 1

    def get_metric(
        self,
        metric: str,
        partition: str = "test",
    ) -> Any:
        return getattr(self, f"mean_{partition}_{metric}")

    def get_best_metric(
        self,
        metric: str,
        partition: str = "test",
    ) -> float:
        return self.get_metric(metric, partition=partition)[self.best_index]

    def get_metric_by_split(
        self,
        metric: str,
        split_index: int,
        partition: str = "test",
    ) -> Any:
        return getattr(self, f"split{split_index}_{partition}_{metric}")

    def get_metrics(
        self,
        partition: str = "test",
    ) -> pd.DataFrame:
        metrics_dict = {}
        for metric in self.metrics:
            mean = getattr(self, f"mean_{partition}_{metric}")[self.best_index]
            metrics_dict[f"{metric}_mean"] = mean
            std = getattr(self, f"std_{partition}_{metric}")[self.best_index]
            metrics_dict[f"{metric}_std"] = std

        metrics_dict = dict(sorted(metrics_dict.items()))
        return pd.DataFrame(metrics_dict, index=[0])

    def get_cv_summary(
        self,
        partition: str = "test",
    ) -> pd.DataFrame:
        rows = []
        for split_index in range(self.num_splits):
            split_dict = {}
            for metric in self.metrics:
                split_metric = self.get_metric_by_split(
                    metric,
                    split_index,
                    partition=partition,
                )
                split_dict[metric] = split_metric[self.best_index]
            rows.append(split_dict)
        return pd.DataFrame(rows)

    def get_params(self) -> List[Dict]:
        return self.params

    def get_best_params(self) -> Dict:
        return self.params[self.best_index]

    def save(self, path: str):
        joblib.dump(self, path)
