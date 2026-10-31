import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import MLP
from itertools import product
from tqdm import tqdm
import pathlib


SEED: int = 42


# --- Load Data ------------------------------------------------------------------------------------------------------------------
taxi_trips = pd.read_parquet('../data/taxi_trips.parquet')
y_col  = 'trip_duration'
x_cols = [x for x in taxi_trips.columns if x != y_col]
X = taxi_trips[x_cols].values
y = taxi_trips[y_col].values
X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size = 0.30, random_state = SEED)
X_val, X_te, y_val, y_te   = train_test_split(X_temp, y_temp, test_size = 2/3, random_state = SEED)

scaler = StandardScaler()
y_tr  = scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
y_val = scaler.transform(y_val.reshape(-1, 1)).ravel()
y_te  = scaler.transform(y_te.reshape(-1, 1)).ravel()


# --- Parameter Search for MLP ---------------------------------------------------------------------------------------------------
params = {
    'd_hidden'        : [32, 256, 1024],
    'activation_type' : ['relu', 'tanh', 'silu'],
    'learning_rate'   : [1e-2, 1e-3, 1e-4]
}

params_product = list(product(*params.values()))

results_path = pathlib.Path('saved_models/results.csv')
results = pd.read_csv(results_path).values.tolist() if results_path.exists() else []
for d_h, act, lr in tqdm(params_product, desc="Parameter search: ", total=len(params_product)):
    path = pathlib.Path(f'saved_models/mlp_{d_h}_{act}_{lr}.pt')
    if path.exists(): 
        continue

    mlp = MLP(
    d_in = X_tr.shape[1],
    d_hidden = d_h, 
    d_out = 2,
    activation_type = act,
    epochs = 100,
    learning_rate = lr,
    batch_size = 256,
    device = 'cpu',
    seed = SEED,
    ) 

    mlp.fit(X_tr, y_tr)
    output = torch.from_numpy(mlp.predict(X_val)).float()
    y_v = torch.from_numpy(y_val).float().view(-1, 1)
    loss = mlp.loss_func(output, y_v).item()

    results.append((d_h, act, lr, loss))
    torch.save(mlp.state_dict(), path)
    pd.DataFrame(results, columns = list(params.keys()) + ['loss']).to_csv(results_path, index = False)


# --- Train Baseline MLP ----------------------------------------------------------------------------------------------------------
mlp_baseline = MLP(
    d_in = X_tr.shape[1],
    d_hidden = 128,
    activation_type = None,
    bias = True,
    batch_size = 256,
    epochs = 100,
    learning_rate = 1e-3,
    device = 'cpu',
    seed = SEED
)

print("Fitting Baseline MLP")
mlp_baseline.fit(X_tr, y_tr)
torch.save(mlp_baseline.state_dict(), f'saved_models/mlp_baseline.pt')