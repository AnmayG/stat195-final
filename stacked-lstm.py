import pandas as pd
import numpy as np
import torch, torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from datetime import datetime
import joblib, random
import matplotlib.pyplot as plt
import os
FIG_DIR = "run_figs"
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

df = pd.read_csv("wti_daily.csv", parse_dates=["Date"]).sort_values("Date")
prices = df["Price"].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

SEQ_LEN = 60  # look-back window

def make_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(scaled, SEQ_LEN)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- DEFINE THE BASE LSTM MODEL ----------
class PriceLSTM(nn.Module):
    def __init__(self, hidden, layers, drop):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            dropout=drop, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # use last time step output

def train_one(net, Xtr, ytr, epochs=60, lr=1e-3, bs=128):
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr)
    loss_fn = nn.MSELoss()
    Xtr_tensor = torch.tensor(Xtr, dtype=torch.float32).to(device)
    ytr_tensor = torch.tensor(ytr, dtype=torch.float32).to(device)
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        idx = torch.randperm(len(Xtr_tensor))
        for i in range(0, len(Xtr_tensor), bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_tensor[b]), ytr_tensor[b])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        if epoch % 10 == 0:
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch:3d}/{epochs} - Loss: {avg_loss:.6f}")
    net.cpu()
    torch.cuda.empty_cache()
    return net

# ---------- CREATE K‑FOLD OUT‑OF‑FOLD PREDICTIONS for the base learners ----------
FOLDS = 5
HIDDEN_SET = [32, 64]  # two LSTM variants
# This matrix will hold each learner's out‑of‑fold (OOF) predictions.
oof_meta = np.zeros((len(X_all), len(HIDDEN_SET)))
# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
kf = KFold(n_splits=FOLDS)

# For plotting the individual learner's OOF predictions on the entire dataset.
predictions_per_learner = {}

for m, hidden in enumerate(HIDDEN_SET):
    print(f"\nTraining LSTM base learner #{m+1} (hidden units={hidden})")
    preds_learner = np.zeros(len(X_all))  # store OOF predictions for learner m
    for tr_idx, val_idx in kf.split(X_all):
        print(f"Fold {len(preds_learner[tr_idx])} -> {len(preds_learner[val_idx])}")
        net = PriceLSTM(hidden, layers=2, drop=0)
        net = train_one(net, X_all[tr_idx], y_all[tr_idx])
        with torch.no_grad():
            preds = net(torch.tensor(X_all[val_idx], dtype=torch.float32)).numpy().squeeze()
        preds_learner[val_idx] = preds
        oof_meta[val_idx, m] = preds
    predictions_per_learner[m] = preds_learner

# ---------- TRAIN THE SVR META LEARNER ----------
# SVR hyperparameters (RBF kernel) as per the paper's description :contentReference[oaicite:0]{index=0}.
meta = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.001)
meta.fit(oof_meta, y_all.ravel())  # note: y_all is still scaled

# ---------- SPLIT THE DATA INTO TRAIN/TEST (Chronological) ----------
# Training: from beginning until 4 Aug 2017; Test: remaining data.
split_date = datetime(2017, 8, 4)
# Find the split index adjusting for the sequence length.
split_idx = df.index[df["Date"] == split_date][0] - SEQ_LEN
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

# Refit base learners on the full training set and predict on the test set.
test_meta = np.zeros((len(X_test), len(HIDDEN_SET)))
pred_test_per_learner = {}  # for plotting test predictions of each base learner
print("\n=== Base-Learner Test Scores ===")
for m, hidden in enumerate(HIDDEN_SET):
    print(f"\nRefitting LSTM base learner #{m+1} (hidden units={hidden}) on full training set")
    net = PriceLSTM(hidden, layers=2, drop=0)
    net = train_one(net, X_train, y_train)
    with torch.no_grad():
        preds_test_scaled = net(torch.tensor(X_test, dtype=torch.float32)).numpy().squeeze()
    test_meta[:, m] = preds_test_scaled
    pred_test_per_learner[m] = preds_test_scaled
    
    # inverse-scale to the price domain
    preds_test = scaler.inverse_transform(preds_test_scaled.reshape(-1, 1)).squeeze()
    truth      = scaler.inverse_transform(y_test).squeeze()
    
    # compute metrics
    mse  = mean_squared_error(truth, preds_test)
    mape = mean_absolute_percentage_error(truth, preds_test) * 100
    
    print(f"LSTM (hidden={hidden:>3})  —  Test MSE: {mse:10.4f}   MAPE: {mape:6.2f}%")

final_pred_scaled = meta.predict(test_meta).reshape(-1, 1)
final_pred = scaler.inverse_transform(final_pred_scaled)
truth = scaler.inverse_transform(y_test)

mse = mean_squared_error(truth, final_pred)
mape = mean_absolute_percentage_error(truth, final_pred) * 100
print(f"\nStacked Model Test MSE: {mse:.4f}   MAPE: {mape:.2f}%")

# ---------- PLOTTING ----------

# (a) Plot the final stacked learner's fit on the test set.
test_dates = df["Date"].values[-len(truth):]  # assumes test set corresponds to the last dates
plt.figure(figsize=(12, 6))
plt.plot(test_dates, truth, label="Actual", color="blue")
plt.plot(test_dates, final_pred, label="Stacked Prediction (SVR meta)", color="red", linestyle="--")
plt.title("Final Stacked Learner (LSTM-SVR) Fit on Test Data")
plt.xlabel("Date")
plt.ylabel("WTI Price")
plt.legend()
plt.tight_layout()
fname = os.path.join(FIG_DIR, "stacked_vs_actual_test.png")
plt.savefig(fname, dpi=300)
plt.close()
print(f"Saved {fname}")

# (b) Plot the individual learner's OOF predictions (on the entire dataset).
for m, hidden in enumerate(HIDDEN_SET):
    # The OOF predictions are aligned with X_all starting at index SEQ_LEN.
    pred_learner = scaler.inverse_transform(predictions_per_learner[m].reshape(-1, 1)).squeeze()
    y_all_inv = scaler.inverse_transform(y_all).squeeze()
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"].values[SEQ_LEN:], y_all_inv, label="Actual", color="blue")
    plt.plot(df["Date"].values[SEQ_LEN:], pred_learner, label=f"LSTM (hidden={hidden}) OOF Prediction", 
             color="green", linestyle="--")
    plt.title(f"LSTM Base Learner (hidden={hidden}) OOF Fit")
    plt.xlabel("Date")
    plt.ylabel("WTI Price")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"oof_hidden{hidden}.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved {fname}")

# (c) Plot the individual learner's predictions on the test set.
for m, hidden in enumerate(HIDDEN_SET):
    pred_test = scaler.inverse_transform(pred_test_per_learner[m].reshape(-1, 1)).squeeze()
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, scaler.inverse_transform(y_test).squeeze(), label="Actual", color="blue")
    plt.plot(test_dates, pred_test, label=f"LSTM (hidden={hidden}) Test Prediction", 
             color="purple", linestyle="--")
    plt.title(f"LSTM Base Learner (hidden={hidden}) Test Fit")
    plt.xlabel("Date")
    plt.ylabel("WTI Price")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"test_hidden{hidden}.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved {fname}")

