from enum import Enum
from functools import partial

import numpy as np

from scipy import stats
from scipy.integrate import trapz

from sklearn import metrics

def auroc(y_true, y_prob):
    return metrics.roc_auc_score(y_true, y_prob, average="weighted")


def mcc(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return metrics.matthews_corrcoef(y_true, y_pred)


def aupr(y_true, y_prob):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    return metrics.auc(recall, precision)


def f1(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return metrics.f1_score(y_true, y_pred, average="weighted")


def prec(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0) # type: ignore


def rec(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0) # type: ignore


def nadr(y_true: np.ndarray, y_score: np.ndarray, k: int = None, sort: bool = True): # type: ignore
    k = k or y_true.shape[0]

    if sort:
        order = np.argsort(y_score)[::-1]
    else:
        order = np.random.permutation(y_true.shape[0])
    y_true = y_true[order][:k]

    # ideal vector has all 1s in first n positions
    ideal = np.zeros_like(y_true)
    ideal[:min(k, y_true.sum())] = 1

    area = trapz(np.cumsum(y_true), dx=1)
    total_area = trapz(np.cumsum(ideal), dx=1)

    if total_area == 0.0:
        return 0.0

    return area / total_area


def pearson(y_true, y_score):
    """Not used, just for compatibility"""
    return 0.0


def jaccard(y_true, y_score):
    """Not used, just for compatibility"""
    return 0.0


def naudc(y_true, y_score):
    """Not used, just for compatibility"""
    return 0.0


class Metric(Enum):
    nadr = "nadr"
    AUROC = "auroc"
    AUPR = "aupr"
    F1 = "f1"
    MCC = "mcc"
    PREC = "prec"
    REC = "rec"



PROB_METRICS = {
    Metric.nadr: nadr,
}


CLF_METRICS = {
    Metric.AUROC: auroc,
    Metric.AUPR: aupr,
    Metric.F1: f1,
    Metric.MCC: mcc,
    Metric.PREC: prec,
    Metric.REC: rec
}


def get_scorers():
    return {
        "auroc": metrics.make_scorer(auroc, needs_proba=True),
    }