import numpy as np
import pandas as pd

from src import settings
from src import utils
from src.evaluation import metrics


NAMES = {
    "pses": "PSE-based",
    "features": "Feature-based",
}


class Evaluator:
    def __init__(self, species):
        self.species = species
        self.metrics = metrics.PROB_METRICS

    def _score_metrics(self, method):
        base_dir = settings.EXP_DIR / self.species / method
        method_name = method.capitalize()

        all_scores = []
        for path in sorted(base_dir.glob(f"*_test.csv")):
            seed = int(path.stem.split("_")[0])
            utils.seed_everything(seed)

            df = pd.read_csv(path)
            y_true = df.Antigen.values.astype(int)
            y_pred = df.prob.values

            scores = {
                "Species":settings.PROT2SPECIES[self.species],
                "Model": NAMES[method],
                "Trial": seed
            }
            for metric, metric_fn in metrics.CLF_METRICS.items():
                scores[metric.name] = metric_fn(y_true, y_pred)
            all_scores.append(scores)

        return pd.DataFrame(all_scores, columns=["Species", "Model", "Trial", "AUROC", "AUPR", "F1", "MCC", "PREC", "REC"])

    def score_metrics(self):
        emb_auroc = self._score_metrics("pses")
        feat_auroc = self._score_metrics("features")
        return pd.concat([emb_auroc, feat_auroc], axis=0, ignore_index=True)

    def _score_nadr(self, method, baseline=False):
        base_dir = settings.EXP_DIR / self.species / method
        sort = not baseline

        all_scores = []
        for path in sorted(base_dir.glob(f"*_ranking.csv")):
            seed = int(path.stem.split("_")[0])
            utils.seed_everything(seed)

            df = pd.read_csv(path)
            df.loc[df.Antigen.isna(), "Antigen"] = False

            y_true = df.Antigen.values.astype(int)
            y_pred = df.prob.values

            scores = {
                "Species":settings.PROT2SPECIES[self.species],
                "Method": NAMES[method] if not baseline else "Standard RV",
                "Trial": seed,
                "nADR": metrics.nadr(y_true, y_pred, sort=sort)
            }
            all_scores.append(scores)

        return pd.DataFrame(all_scores, columns=["Species", "Method", "Trial", "nADR"])

    def score_nadr(self):
        emb_scores = self._score_nadr("pses")
        base_scores = self._score_nadr("pses", baseline=True)
        scores = [emb_scores, base_scores]
        return pd.concat(scores, axis=0, ignore_index=True)

    def _score_counts(self, method, baseline=False):
        base_dir = settings.EXP_DIR / self.species / method

        all_counts = []
        for path in base_dir.glob(f"*ranking.csv"):
            seed = int(path.stem.split("_")[0])
            utils.seed_everything(seed)

            df = pd.read_csv(path)

            if not baseline:
                df = df.sort_values("prob", ascending=False).reset_index(drop=True)
            else:
                perm = np.random.permutation(df.shape[0])
                df = df.iloc[perm].reset_index(drop=True)

            cumsum = np.cumsum(df.Antigen==True)
            for i, cs in enumerate(cumsum, 1):
                all_counts.append({
                    "Species": settings.PROT2SPECIES[self.species],
                    "Method": NAMES[method] if not baseline else "Standard RV",
                    "Trial": seed,
                    "Step": i,
                    "Antigens Discovered": cs,
                    "N. of in-vivo tests": i,
                })

        return pd.DataFrame(all_counts, columns=["Species", "Method", "Trial", "Step", "Antigens Discovered", "N. of in-vivo tests"])

    def score_counts(self):
        emb_counts = self._score_counts("pses")
        base_counts = self._score_counts("pses", baseline=True)
        counts = [emb_counts, base_counts]
        return pd.concat(counts, axis=0, ignore_index=True)

    def _score_cv(self, method):
        base_dir = settings.EXP_DIR / self.species / method

        all_cvs = []
        for path in base_dir.glob(f"*cv_results.csv"):
            seed = int(path.stem.split("_")[0])
            utils.seed_everything(seed)

            df = pd.read_csv(path)
            all_cvs.append({
                "Species": settings.PROT2SPECIES[self.species],
                "Model": NAMES[method],
                "Trial": seed,
                "AUROC": df["auroc"].mean()
            })

        return pd.DataFrame(all_cvs, columns=["Species", "Model", "Trial", "AUROC"])

    def score_cv(self):
        emb_cv = self._score_cv("pses")
        feat_cv = self._score_cv("features")
        return pd.concat([emb_cv, feat_cv], axis=0, ignore_index=True)

    def _score_no_experiments(self, method, baseline=False, how_many=None):
        base_dir = settings.EXP_DIR / self.species / method

        all_no_exps = []
        for path in base_dir.glob(f"*ranking.csv"):
            seed = int(path.stem.split("_")[0])
            utils.seed_everything(seed)

            df = pd.read_csv(path)
            if baseline is True:
                df = df.sample(frac=1.0).reset_index(drop=True)
            antigens = df[df.Antigen==True]

            if how_many is not None:
                antigens = antigens[:how_many]


            no_exps = {
                "Species":settings.PROT2SPECIES[self.species],
                "Method": NAMES[method] if not baseline else "Standard RV",
                "Trial": seed,
                "N. Experiments": antigens.index[-1] if not antigens.empty else how_many
            }

            all_no_exps.append(no_exps)

        return pd.DataFrame(all_no_exps, columns=["Species", "Method", "Trial", "N. Experiments"])

    def score_no_experiments(self, how_many=None):
        emb_no_exps = self._score_no_experiments("pses", how_many=how_many)
        base_no_exps = self._score_no_experiments("pses", baseline=True, how_many=how_many)
        no_exps = [emb_no_exps, base_no_exps]
        return pd.concat(no_exps, axis=0, ignore_index=True)

