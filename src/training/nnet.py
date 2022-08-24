from typing import Tuple

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from skorch import NeuralNetBinaryClassifier
from skorch.helper import SliceDict

import torch
from torch import nn
from torch.nn import functional as F


class NetModule(nn.Module):
    def __init__(
        self,
        dim_input: int = 1280,
        dim_hidden: int = 2560,
        dropout: float = 0.5,
    ):
        super(NetModule, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_input),
            nn.ReLU(),
        )

        self.output = nn.Linear(dim_hidden, 1)

    def forward(self, X, **kwargs):
        hidden_noise = self.encoder(X + torch.randn_like(X) * 0.1)
        hidden = self.encoder(F.dropout(X, 0.5, self.training))
        rec = self.decoder(hidden_noise)
        logits = self.output(hidden)
        return logits.view(-1), rec


class NeuralNetClassifier(NeuralNetBinaryClassifier):
    def fit(self, X, y, **fit_params):
        self.set_params(module__dim_input=X.shape[1])
        super().fit(X, y.astype(np.float32))

    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits, rec = y_pred
        return 0.5 * F.binary_cross_entropy_with_logits(logits, y_true) + 0.5 * F.mse_loss(rec, X) # type: ignore
