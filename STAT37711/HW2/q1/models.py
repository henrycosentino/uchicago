import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# --- MLP ----
class MLP(nn.Module):
    def __init__(
            self, 
            d_in: int, 
            d_hidden: int, 
            d_out: int = 2, 
            activation_type: None | str = 'relu', 
            bias: bool = True, 
            batch_size: int = 128, 
            epochs: int = 100, 
            learning_rate: float = 1e-3,
            device: str = 'cpu',
            kwargs: dict | None = None,
            seed: int = 0
        ):
        super().__init__()

        non_linear_funcs = {
            'relu' : nn.ReLU,
            'gelu' : nn.GELU,
            'silu' : nn.SiLU,
            'tanh' : nn.Tanh,
        }

        assert all([d_in > 0, d_hidden > 0, d_out == 2])
        assert activation_type in non_linear_funcs.keys() or activation_type is None

        torch.manual_seed(seed)
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.network = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias = bias, device = self.device),
            non_linear_funcs[activation_type](**(kwargs or {})) if activation_type is not None else nn.Identity(),
            nn.Linear(d_hidden, d_out, bias = bias, device = self.device)
        )

    def loss_func(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f_mu = output[:, 0:1]
        log_sigma = torch.clamp(output[:, 1:2], -7.0, 5.0)

        nll = log_sigma + (y - f_mu) ** 2 / (2 * torch.exp(2 * log_sigma))
        return nll.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        dataset = TensorDataset(X, y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, maximize=False)

        self.network.train()

        for epoch in range(0, self.epochs):

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = self.loss_func(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)

    def predict(self, X: np.ndarray):
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy()