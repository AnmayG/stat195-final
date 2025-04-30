from __future__ import annotations
from pathlib import Path

CSV_PATH      = Path("wti_daily.csv")  # path to your daily WTI price file
PRICE_COLUMN  = "Price"                # column in CSV that holds the price
TEST_SIZE     = 0.15                   # fraction for final test split
VAL_SIZE      = 0.15                   # fraction of training for validation
LOOKBACK      = 30                     # window length fed to each LSTM
RNG_SEED      = 42                     # reproducibility seed

# ---------------------------------------------------------------------
# 1. Imports & environment setup
# ---------------------------------------------------------------------
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------
# 2. Data loading utilities
# ---------------------------------------------------------------------

def load_and_scale(csv_path: Path, target_col: str):
    df = (
        pd.read_csv(csv_path, parse_dates=["Date"], infer_datetime_format=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    prices = df[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices).astype(np.float32)
    return scaled, scaler


def make_sequences(arr: np.ndarray, lookback: int):
    X, y = [], []
    for t in range(lookback, len(arr)):
        X.append(arr[t - lookback : t])
        y.append(arr[t])
    return np.array(X), np.array(y)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx):  # type: ignore[override]
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------
# 3. Model definition
# ---------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last timestep
        return self.fc(out)


# ---------------------------------------------------------------------
# 4. Training helpers
# ---------------------------------------------------------------------

def train_lstm(model, train_loader, val_loader, lr=1e-3, epochs=200, patience=15):
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, wait, best_state = float("inf"), 0, None
    for _ in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(dev), yb.to(dev)
            optim.zero_grad(); loss = criterion(model(Xb), yb); loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([criterion(model(Xv.to(dev)), yv.to(dev)).item() for Xv, yv in val_loader])
        if val_loss < best_loss - 1e-6:
            best_loss, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)  # type: ignore[arg-type]
    return model


def predict_lstm(model, loader):
    model.eval(); outs = []
    with torch.no_grad():
        for Xb, _ in loader:
            outs.append(model(Xb.to(dev)).cpu().numpy())
    return np.vstack(outs).flatten()


# ---------------------------------------------------------------------
# 5. Stacking pipeline
# ---------------------------------------------------------------------

def train_base_learners(X_tr, y_tr, X_val, y_val, param_grid):
    models = []
    for p in param_grid:
        print(f"Training base learner: {p}")
        net = LSTMForecaster(p["hidden"], p["dropout"]).to(dev)
        tr_loader  = DataLoader(SequenceDataset(X_tr, y_tr),  batch_size=32, shuffle=True)
        val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=32, shuffle=False)
        train_lstm(net, tr_loader, val_loader, lr=p["lr"])
        models.append(net)
    return models


def make_meta_features(models, X):
    dummy_y = np.zeros((len(X), 1), dtype=np.float32)
    loader = DataLoader(SequenceDataset(X, dummy_y), batch_size=64, shuffle=False)
    return np.column_stack([predict_lstm(m, loader) for m in models])


def evaluate_and_plot(y_true, base_preds, meta_pred):
    print("\n========== Test MAPE ==========")
    for k in range(base_preds.shape[1]):
        print(f"Base LSTM {k+1}: {mean_absolute_percentage_error(y_true, base_preds[:, k])*100:6.2f}%")
    print(f"Stacked SVR : {mean_absolute_percentage_error(y_true, meta_pred)*100:6.2f}%")
    print("================================")

    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="True")
    for k in range(base_preds.shape[1]):
        plt.plot(base_preds[:, k], label=f"LSTM{k+1}")
    plt.plot(meta_pred, label="Stacked")
    plt.title("Test‑set forecasts vs. true prices")
    plt.xlabel("Time steps"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 6. Main execution
# ---------------------------------------------------------------------

def main():
    # 1 Load data
    series, scaler = load_and_scale(CSV_PATH, PRICE_COLUMN)
    X, y = make_sequences(series, LOOKBACK)

    # 2 Train/val/test split (no shuffling to preserve chronology)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_fraction, shuffle=False)

    # 3 Train base learners
    PARAM_GRID = [
        {"hidden": 32,  "dropout": 0.1, "lr": 1e-3},
        {"hidden": 64,  "dropout": 0.2, "lr": 5e-4},
        {"hidden": 128, "dropout": 0.3, "lr": 1e-4},
    ]
    base_models = train_base_learners(X_train, y_train, X_val, y_val, PARAM_GRID)

    # 4 Meta‑learner
    X_meta_train = make_meta_features(base_models, X_val)
    X_meta_test  = make_meta_features(base_models, X_test)
    svr = SVR(kernel="rbf", C=100.0, gamma="scale").fit(X_meta_train, y_val.flatten())

    # 5 Predictions
    base_preds_test = X_meta_test
    meta_pred_test  = svr.predict(X_meta_test)

    # 6 Inverse scale & evaluation
    y_test_inv      = scaler.inverse_transform(y_test)
    base_preds_inv  = scaler.inverse_transform(base_preds_test)
    meta_pred_inv   = scaler.inverse_transform(meta_pred_test.reshape(-1,1))

    evaluate_and_plot(y_test_inv, base_preds_inv, meta_pred_inv)


if __name__ == "__main__":
    main()
