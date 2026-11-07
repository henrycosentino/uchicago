import numpy as np
from typing import Tuple, List
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, h: int = 1000, sigma: float = 1.0, lam: float = 1e-3, seed: int = 42):
        assert sigma > 0

        self.h = h
        self.sigma = sigma
        self.lam = lam
        self.rng = np.random.default_rng(seed = seed)

        self.W = None
        self.b = None
        self.coef_ = None

    def _draw_features(self, p: int) -> None:
        self.W = self.rng.normal(loc = 0, scale = 1 / self.sigma, size = (p, self.h))
        self.b = self.rng.uniform(low = 0, high = 2 * np.pi, size = self.h)


    def _features(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape

        u = X @ self.W + self.b

        Z = np.empty(shape = (n, self.h * 2))
        Z[:, 0::2] = np.sin(u)
        Z[:, 1::2] = np.cos(u)

        return Z


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._draw_features(X.shape[1])

        Z = self._features(X)

        n, p = Z.shape

        A = Z.T @ Z + np.identity(p) * self.lam
        b = Z.T @ y

        coef = np.linalg.solve(A, b)
        self.coef_ = coef

        return None


    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._features(X)

        return Z @ self.coef_


def rks_sweep(params: List[Tuple], X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int = 42) -> Tuple:
    best_model   = None
    best_params  = None
    best_val_mse = np.inf
    loss = {
        'tr'  : [],
        'val' : [],
    }
    for h, s, l in tqdm(params, total=len(params), desc="RKS Param Sweep"):
        model = RandomKitchenSinks(
            h     = h,
            sigma = s,
            lam   = l,
            seed  = seed
        )
        model.fit(X_tr, y_tr)
        tr_pred  = model.predict(X_tr)
        val_pred = model.predict(X_val)
        tr_mse   = np.mean((y_tr - tr_pred) ** 2)
        val_mse  = np.mean((y_val - val_pred) ** 2)

        loss['tr'].append(tr_mse)
        loss['val'].append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model   = model
            best_params  = (h, s, l)  

    return best_model, best_params, best_val_mse, loss


class MLP(nn.Module):
    def __init__(
            self, 
            d_i: int, 
            d_h: int, 
            d_o: int = 1, 
            bias: bool = True,
            batch_size: int = 128,
            epochs: int = 100,
            lr: float = 1e-3,
            device: str = 'cpu',
            seed: int = 42
        ):
        super().__init__()

        torch.manual_seed(seed)

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.net = nn.Sequential(
            nn.Linear(d_i, d_h, bias=bias, device=device),
            nn.ReLU(),
            nn.Linear(d_h, d_o, bias=bias, device=device),
        )

        self.config = {
            'in_dim'        : d_i, 
            'hidden_dim'    : d_h, 
            'out_dim'       : d_o, 
            'bias'          : bias,
            'batch_size'    : batch_size, 
            'epochs'        : epochs, 
            'learning_rate' : lr, 
            'seed'          : seed,
        }


    def mse_loss(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((p - y.view_as(p)) ** 2)


    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:

        dataset = TensorDataset(X, y)
        loader  = DataLoader(dataset, self.batch_size, shuffle=True)

        optimizer = torch.optim.SGD(params = self.parameters(), lr = self.lr, maximize = False)

        self.train()

        for _ in range(self.epochs):

            for X_b, y_b in loader:
                optimizer.zero_grad()
                out = self.forward(X_b)
                loss = self.mse_loss(out, y_b)
                loss.backward()
                optimizer.step()


    def forward(self, X: torch.Tensor):
        return self.net(X)


    def predict(self, X: torch.Tensor):
        self.eval()
        with torch.no_grad():
            return self.forward(X)


def mlp_sweep(params: List[Tuple], X_tr: torch.Tensor, y_tr: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, seed: int = 42) -> Tuple:
    n, p = X_tr.shape

    best_model   = None
    best_params  = None
    best_val_mse = np.inf

    loss = {
        'tr'  : [],
        'val' : [],
    }
    for d_h, lr, bs, bias, epochs in tqdm(params, total=len(params), desc="MLP Param Sweep"):
        model = MLP(
            d_i        = p,
            d_h        = d_h,
            d_o        = 1,
            bias       = bias,
            batch_size = bs,
            epochs     = epochs,
            lr         = lr,
            device     = 'cpu',
            seed       = seed,
        )
        model.fit(X_tr, y_tr)
        tr_pred  = model.predict(X_tr)
        val_pred = model.predict(X_val)
        tr_mse  = torch.mean((tr_pred - y_tr.view_as(tr_pred)) ** 2).item()
        val_mse = torch.mean((val_pred - y_val.view_as(val_pred)) ** 2).item()

        loss['tr'].append(tr_mse)
        loss['val'].append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model   = model
            best_params  = (d_h, lr, bs, bias, epochs)

    return best_model, best_params, best_val_mse, loss
        
