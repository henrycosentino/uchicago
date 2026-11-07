import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# --- MLP w/o Embedding Layer ----
class MLP(nn.Module):
    def __init__(
            self, 
            d_in: int, 
            d_hidden: int, 
            d_out: int = 1, 
            bias: bool = True, 
            batch_size: int = 128, 
            epochs: int = 100, 
            learning_rate: float = 1e-3,
            device: str = 'cpu',
            seed: int = 0
        ):
        super().__init__()

        assert all([d_in > 0, d_hidden > 0, d_out == 1])

        torch.manual_seed(seed)
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.network = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias = bias, device = self.device),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out, bias = bias, device = self.device)
        )

    def mse_loss(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        v = p - y
        return (v ** 2).mean()

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, patience: int | None = None):
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        early_stopping = (X_val is not None and y_val is not None and patience is not None)

        if early_stopping:
            X_val = torch.from_numpy(X_val).float().to(self.device)
            y_val = torch.from_numpy(y_val).float().view(-1, 1).to(self.device)

            best_loss = np.inf
            counter = 0
            best_model_state = None

        dataset = TensorDataset(X, y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, maximize=False)

        self.network.train()

        for epoch in range(0, self.epochs):

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = self.mse_loss(output, y_batch)
                loss.backward()
                optimizer.step()

            # Early stopping
            if early_stopping:
                self.network.eval()
                with torch.no_grad():
                    val_out = self.forward(X_val)
                    val_loss = self.mse_loss(val_out, y_val).item()

                self.network.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        if early_stopping:
            self.network.load_state_dict(best_model_state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)

    def predict(self, X: np.ndarray):
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy()


# --- MLP w/ Embedding Layer ----
class EmbeddingMLP(nn.Module):
    def __init__(
            self, 
            d_in: int, 
            d_hidden: int,
            n_cat: list[int],
            d_embed: int,
            d_out: int = 1, 
            bias: bool = True, 
            batch_size: int = 128, 
            epochs: int = 100, 
            learning_rate: float = 1e-3,
            device: str = 'cpu',
            seed: int = 0
        ):
        super().__init__()

        assert all([d_in > 0, d_embed > 0, len(n_cat) == d_in, d_hidden > 0, d_out == 1])

        torch.manual_seed(seed)
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.embedding = nn.ModuleList([
            nn.Embedding(num_embeddings = n, embedding_dim = d_embed, padding_idx = 0, device = self.device)
            for n in n_cat
        ])

        self.network = nn.Sequential(
            nn.Linear(d_in * d_embed, d_hidden, bias = bias, device = self.device),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out, bias = bias, device = self.device)
        )

    def mse_loss(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        v = p - y
        return (v ** 2).mean()

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, patience: int | None = None):
        X = torch.from_numpy(X).long().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        early_stopping = (X_val is not None and y_val is not None and patience is not None)

        if early_stopping:
            X_val = torch.from_numpy(X_val).long().to(self.device)
            y_val = torch.from_numpy(y_val).float().view(-1, 1).to(self.device)

            best_loss = np.inf
            counter = 0
            best_model_state = None

        dataset = TensorDataset(X, y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, maximize=False)

        self.train()

        for epoch in range(0, self.epochs):

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = self.mse_loss(output, y_batch)
                loss.backward()
                optimizer.step()

            # Early stopping
            if early_stopping:
                self.eval()
                with torch.no_grad():
                    val_out = self.forward(X_val)
                    val_loss = self.mse_loss(val_out, y_val).item()

                self.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        if early_stopping:
            self.load_state_dict(best_model_state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat([e(X[:, i]) for i, e in enumerate(self.embedding)], dim = 1)
        output = self.network(embedded)
        return output

    def predict(self, X: np.ndarray):
        self.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).long().to(self.device)
            return self.forward(X).detach().cpu().numpy()


# --- MLP w/ Embedding Layer & Target Encoding ----
class HybridMLP(nn.Module):
    def __init__(
            self, 
            d_hidden: int,
            n_cat: list[int],
            d_te: int,
            d_embed: int,
            d_out: int = 1, 
            bias: bool = True, 
            batch_size: int = 128, 
            epochs: int = 100, 
            learning_rate: float = 1e-3,
            device: str = 'cpu',
            seed: int = 0
        ):
        super().__init__()

        assert all([len(n_cat) > 0, d_embed > 0, d_te >= 0, d_hidden > 0, d_out == 1])

        torch.manual_seed(seed)
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.embedding = nn.ModuleList([
            nn.Embedding(num_embeddings = n, embedding_dim = d_embed, padding_idx = 0) for n in n_cat
        ])

        d_combined = len(n_cat) * d_embed + d_te
        self.network = nn.Sequential(
            nn.Linear(d_combined, d_hidden, bias=bias),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out, bias=bias),
        )
        self.to(self.device)


    def _tensors(self, X_cat, X_te, y=None):
        out = [
            torch.from_numpy(X_cat).long().to(self.device),
            torch.from_numpy(X_te).float().to(self.device)
        ]
        if y is not None:
            out.append(torch.from_numpy(y).float().view(-1,1).to(self.device))
        return out


    def mse_loss(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        v = p - y
        return (v ** 2).mean()

    def fit(
            self, 
            X_cat: np.ndarray, 
            X_te: np.ndarray,
            y: np.ndarray, 
            X_cat_val: np.ndarray | None = None, 
            X_te_val: np.ndarray | None = None, 
            y_val: np.ndarray | None = None, 
            patience: int | None = None
            ):
        Xc, Xt, y = self._tensors(X_cat, X_te, y)

        early_stopping = all(v is not None for v in (X_cat_val, X_te_val, y_val, patience))
        
        if early_stopping:
            Xc_v, Xt_v, y_v = self._tensors(X_cat_val, X_te_val, y_val)
            best_loss = np.inf
            counter = 0
            best_model_state = None

        dataset = TensorDataset(Xc, Xt, y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, maximize=False)

        self.train()

        for epoch in range(0, self.epochs):

            for xc_b, xt_b, y_b in loader:
                optimizer.zero_grad()
                output = self.forward(xc_b, xt_b)
                loss = self.mse_loss(output, y_b)
                loss.backward()
                optimizer.step()

            # Early stopping
            if early_stopping:
                self.eval()
                with torch.no_grad():
                    val_loss = self.mse_loss(self.forward(Xc_v, Xt_v), y_v).item()

                self.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        if early_stopping and best_model_state is not None:
            self.load_state_dict(best_model_state)

    def forward(self, X_cat: torch.Tensor, X_te: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([e(X_cat[:, i]) for i, e in enumerate(self.embedding)], dim = 1)
        out = torch.cat([emb, X_te], dim=1)
        return self.network(out)

    def predict(self, X_cat, X_te):
        self.eval()
        with torch.no_grad():
            X_cat_t, X_te_t = self._tensors(X_cat, X_te)
            return self.forward(X_cat_t, X_te_t).cpu().numpy()