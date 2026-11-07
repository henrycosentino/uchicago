import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


def plot_surface_2d(
    model_fn: Callable[[np.ndarray], np.ndarray] | nn.Module,
    xlim: Tuple[float, float], 
    ylim: Tuple[float, float],
    title: str = "Fitted surface", 
    filename: str = "surface.png", 
    grid: int = 200
):
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X1, X2 = np.meshgrid(xs, ys)
    Xgrid = np.stack([X1.ravel(), X2.ravel()], axis=1)
    if isinstance(model_fn, nn.Module):
        with torch.no_grad():
            X_grid = torch.from_numpy(Xgrid).float().to(next(model_fn.parameters()).device)
            Z = model_fn(X_grid).cpu().numpy().reshape(grid, grid)
    else:
        Z = model_fn(Xgrid).reshape(grid, grid)

    plt.figure()
    plt.contourf(X1, X2, Z, levels=30)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(label="pred")
    plt.tight_layout()
    plt.show()