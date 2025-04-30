# pytorch_lstm_wti.py
import pandas as pd
import numpy as np
import torch, torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
import matplotlib.pyplot as plt

# ---------- 1. LOAD & PRE‑PROCESS ----------
# csv must contain a Date column and a Close (WTI price) column
df = pd.read_csv("wti_daily.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)
prices = df["Price"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

SEQ_LEN = 60                       # look‑back window (~3 months of trading days)

def make_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, SEQ_LEN)

# ---------- 2. TRAIN / TEST SPLIT ----------
split_date = datetime(2017, 8, 4)          # 4 Aug 2017
split_idx  = df.index[df["Date"] == split_date][0] - SEQ_LEN
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# convert to tensors
device = "cuda" if torch.cuda.is_available() else "cpu"
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
y_test  = torch.tensor(y_test,  dtype=torch.float32).to(device)

# ---------- 3. MODEL ----------
class PriceLSTM(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden,
                            num_layers=layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last time step
        return self.fc(out)

model = PriceLSTM().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- 4. TRAIN ----------
EPOCHS = 100
BATCH  = 128

def batches(X, y, bs):
    for i in range(0, len(X), bs):
        yield X[i:i+bs], y[i:i+bs]

for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in batches(X_train, y_train, BATCH):
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  train‑loss {loss.item():.6f}")

# ---------- 5. EVALUATE ----------
model.eval()
with torch.no_grad():
    pred = model(X_test).cpu().numpy()
    truth = y_test.cpu().numpy()

pred_inv   = scaler.inverse_transform(pred)
truth_inv  = scaler.inverse_transform(truth)

mse  = mean_squared_error(truth_inv, pred_inv)
mape = mean_absolute_percentage_error(truth_inv, pred_inv) * 100
print(f"Test MSE  : {mse:.4f}")
print(f"Test MAPE : {mape:.2f}%")

# ---------- 6. PLOT ----------
plt.figure(figsize=(10,4))
plt.plot(df["Date"][-len(truth_inv):], truth_inv, label="Actual")
plt.plot(df["Date"][-len(pred_inv):], pred_inv, label="Predicted")
plt.title("One‑day‑ahead WTI price forecast")
plt.legend(); plt.tight_layout(); plt.show()

