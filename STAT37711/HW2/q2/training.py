from models import *
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import pathlib


SEED: int = 42


# --- Part a, b -------------------------------------------------------------------------------------
retail_df = pd.read_csv('../data/retail.csv')
X = retail_df[['StockCode', 'CustomerID']]
y = retail_df['Quantity'].values

log_y = np.log(y)   # Log transform y to account for heavy right tails

X_tr, X_temp, y_tr, y_temp = train_test_split(X, log_y, test_size = 0.3, random_state = SEED)
X_val, X_te, y_val, y_te   = train_test_split(X_temp, y_temp, test_size = 2/3, random_state = SEED)

encoder = TargetEncoder(cols = ['StockCode', 'CustomerID'])

X_tr_encoded = encoder.fit_transform(X_tr, y_tr)
X_val_encoded = encoder.transform(X_val, y_val)
X_te_encoded = encoder.transform(X_te, y_te)

X_tr_encoded = X_tr_encoded.to_numpy()
X_val_encoded = X_val_encoded.to_numpy()
X_te_encoded = X_te_encoded.to_numpy()

path = pathlib.Path("saved_models/mlp_b.pt")
if not path.exists():
    mlp = MLP(
        d_in = X_tr_encoded.shape[1],
        d_hidden = 128,
        d_out = 1,
        bias = True,
        batch_size = 128,
        epochs = 100,
        learning_rate = 1e-3,
        device = 'cpu',
        seed = 42  
    )
    print('Training Common MLP...')
    mlp.fit(X_tr_encoded, y_tr, X_val_encoded, y_val, patience = 5)
    torch.save(mlp.state_dict(), "saved_models/mlp_b.pt")


# --- Part c ------------------------------------------------------------------------------------------
ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
X_tr_idx = (ordinal_encoder.fit_transform(X_tr) + 1).astype(int)
X_val_idx = (ordinal_encoder.transform(X_val) + 1).astype(int)
X_te_idx = (ordinal_encoder.transform(X_te) + 1).astype(int)

n_cat = [len(c) + 1 for c in ordinal_encoder.categories_]

path = pathlib.Path("saved_models/mlp_c.pt")
if not path.exists():
    emb_mlp = EmbeddingMLP(
        d_in = X_tr_idx.shape[1],
        d_hidden = 128,
        n_cat = n_cat,
        d_embed = 16, 
        d_out = 1,
        bias = True,
        batch_size = 128,
        epochs = 100,
        learning_rate = 1e-3,
        device = 'cpu',
        seed = 42  
    )
    print('Training Embedded cMLP...')
    emb_mlp.fit(X_tr_idx, y_tr, X_val_idx, y_val, patience = 5)
    torch.save(emb_mlp.state_dict(), "saved_models/mlp_c.pt")


# --- Part d ------------------------------------------------------------------------------------------
ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
X_tr_idx = (ordinal_encoder.fit_transform(X_tr) + 1).astype(int)
X_val_idx = (ordinal_encoder.transform(X_val) + 1).astype(int)
X_te_idx = (ordinal_encoder.transform(X_te) + 1).astype(int)

n_cat = [len(c) + 1 for c in ordinal_encoder.categories_]

path = pathlib.Path("saved_models/mlp_c.pt")
if not path.exists():
    emb_mlp = EmbeddingMLP(
        d_in = X_tr_idx.shape[1],
        d_hidden = 128,
        n_cat = n_cat,
        d_embed = 16, 
        d_out = 1,
        bias = True,
        batch_size = 128,
        epochs = 100,
        learning_rate = 1e-3,
        device = 'cpu',
        seed = 42  
    )
    print('Training Embedded cMLP...')
    emb_mlp.fit(X_tr_idx, y_tr, X_val_idx, y_val, patience = 5)
    torch.save(emb_mlp.state_dict(), "saved_models/mlp_c.pt")