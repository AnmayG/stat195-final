#!/usr/bin/env python3
"""
Stacked LSTM → SVR with **Out‑of‑Fold (OOF) stacking**
=====================================================
This version matches the reference pipeline: every base LSTM produces
5‑fold out‑of‑fold predictions across the *training* window; the SVR
meta‑learner is trained on those OOF rows, then each LSTM is re‑trained
on the full training span and its predictions on the chronologically
held‑out *test* span become the meta‑features for inference.

Edit the constants in **Section 0** to change data paths, split dates,
or hyper‑parameters.

Dependencies: pandas, numpy, torch≥2.2, scikit‑learn, matplotlib.
"""
from __future__ import annotations

# ------------------------------------------------------------------
# 0. CONSTANTS
# ------------------------------------------------------------------
from pathlib import Path
from datetime import datetime
CSV_PATH     = Path("wti_daily.csv")
PRICE_COLUMN = "Price"
LOOKBACK     = 30           # LSTM window
FOLDS        = 5            # K for OOF stacking
TEST_SPLIT   = 0.15         # last 15 % of samples become the test span
SEED         = 42

# Base‑LSTM hyper‑param grid
GRID = [
    {"hidden": 32,  "layers": 1, "drop": 0.1, "lr": 1e-3},
    {"hidden": 64,  "layers": 1, "drop": 0.2, "lr": 5e-4},
    {"hidden": 128, "layers": 1, "drop": 0.3, "lr": 1e-4},
]

# SVR meta‑learner hyper‑parameters
SVR_C      = 1000.0
SVR_GAMMA  = "auto"

# ------------------------------------------------------------------
# 1. IMPORTS & REPRODUCIBILITY
# ------------------------------------------------------------------
import random, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 2. DATA PREP
# ------------------------------------------------------------------

def load_scaled(csv: Path, col: str):
    df = pd.read_csv(csv, parse_dates=["Date"]).sort_values("Date")
    prices = df[col].values.reshape(-1,1)
    mm = MinMaxScaler(); scaled = mm.fit_transform(prices).astype(np.float32)
    return df, scaled, mm

def make_sequences(arr: np.ndarray, window: int):
    X,y=[],[]
    for t in range(window, len(arr)):
        X.append(arr[t-window:t]); y.append(arr[t])
    return np.array(X), np.array(y)

class SeqDS(Dataset):
    def __init__(self,X,y): self.X=torch.from_numpy(X); self.y=torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

# ------------------------------------------------------------------
# 3. BASE LSTM
# ------------------------------------------------------------------
class PriceLSTM(nn.Module):
    def __init__(self, hidden:int, layers:int, drop:float):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, dropout=drop, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self,x): out,_=self.lstm(x); return self.fc(out[:,-1,:])

# ------------------------------------------------------------------
# 4. TRAIN / PREDICT HELPERS
# ------------------------------------------------------------------

def train_lstm(net, X, y, epochs=200, bs=32, lr=1e-3, patience=15):
    net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    best, wait, best_state = np.inf, 0, None
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), bs):
            b = idx[i:i+bs]
            opt.zero_grad(); loss = loss_fn(net(X_t[b]), y_t[b]); loss.backward(); opt.step()
        # simple patience on full‑batch loss
        with torch.no_grad():
            net.eval(); cur = loss_fn(net(X_t), y_t).item()
        if cur < best - 1e-6: best, wait, best_state = cur, 0, {k:v.clone() for k,v in net.state_dict().items()}
        else: wait += 1
        if wait >= patience: break
    net.load_state_dict(best_state); net.cpu(); torch.cuda.empty_cache(); return net


def predict_lstm(net, X, bs=128):
    net.eval(); out=[]; X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), bs): out.append(net(X_t[i:i+bs]).numpy())
    return np.vstack(out).flatten()

# ------------------------------------------------------------------
# 5. MAIN PIPELINE
# ------------------------------------------------------------------

def main():
    # 1. Data
    df, series, mm = load_scaled(CSV_PATH, PRICE_COLUMN)
    X_all, y_all = make_sequences(series, LOOKBACK)

    # Chronological split into train vs. test
    test_size = int(len(X_all) * TEST_SPLIT)
    train_len = len(X_all) - test_size
    X_train, y_train = X_all[:train_len], y_all[:train_len]
    X_test,  y_test  = X_all[train_len:], y_all[train_len:]

    # 2. Out‑of‑fold predictions for each base learner
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(X_train), len(GRID)))
    base_retrain_models = []  # keep models for later test prediction

    for m, cfg in enumerate(GRID):
        print(f"\nBase learner {m+1}/{len(GRID)} — hidden={cfg['hidden']}")
        oof_preds = np.zeros(len(X_train))
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
            print(f"  Fold {fold}/{FOLDS}")
            net = PriceLSTM(cfg['hidden'], cfg['layers'], cfg['drop'])
            net = train_lstm(net, X_train[tr_idx], y_train[tr_idx], lr=cfg['lr'])
            preds = predict_lstm(net, X_train[val_idx])
            oof_preds[val_idx] = preds
        oof_meta[:, m] = oof_preds
        # retrain on FULL training span for later inference
        full_net = PriceLSTM(cfg['hidden'], cfg['layers'], cfg['drop'])
        full_net = train_lstm(full_net, X_train, y_train, lr=cfg['lr'])
        base_retrain_models.append(full_net)

    # 3. Fit SVR meta‑learner on OOF predictions
    ss = StandardScaler(); X_meta_std = ss.fit_transform(oof_meta)
    svr = SVR(kernel='rbf', C=SVR_C, gamma=SVR_GAMMA).fit(X_meta_std, y_train.flatten())

    # 4. Build test meta‑features
    test_meta = np.column_stack([predict_lstm(m, X_test) for m in base_retrain_models])
    test_meta_std = ss.transform(test_meta)
    meta_pred_scaled = svr.predict(test_meta_std).reshape(-1,1)

    # 5. Inverse‑scale & evaluation
    y_true = mm.inverse_transform(y_test)
    base_preds_inv = mm.inverse_transform(test_meta)
    meta_pred = mm.inverse_transform(meta_pred_scaled)

    print("\n===== Test MAPE =====")
    for k in range(len(GRID)):
        print(f"LSTM{k+1}: {mean_absolute_percentage_error(y_true, base_preds_inv[:,k])*100:6.2f}%")
    print(f"Stacked : {mean_absolute_percentage_error(y_true, meta_pred)*100:6.2f}%")

    # 6. Plot
    dates_test = df["Date"].values[-len(y_true):]
    plt.figure(figsize=(10,5))
    plt.plot(dates_test, y_true, label="True")
    for k in range(base_preds_inv.shape[1]):
        plt.plot(dates_test, base_preds_inv[:,k], linestyle="--", label=f"LSTM{k+1}")
    plt.plot(dates_test, meta_pred, label="Stacked", linewidth=2)
    plt.xlabel("Date"); plt.ylabel("WTI Price"); plt.title("OOF‑stacked LSTM+SVR test fit"); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

