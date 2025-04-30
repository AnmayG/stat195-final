import os, time, random, math
import numpy  as np
import pandas as pd
import torch, torch.nn as nn
from datetime        import datetime
from sklearn.preprocessing        import MinMaxScaler
from sklearn.svm                  import SVR
from sklearn.metrics              import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

FIG_DIR = "run_figs"
os.makedirs(FIG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT]  Using device: {device}")

# Load raw price series, cut TRAIN / TEST, scale TRAIN-only
CSV_FILE   = "wti_daily.csv"           # must contain Date, Price columns
SEQ_LEN    = 60                        # look-back window (days)
SPLIT_DATE = datetime(2017, 8, 4)      # identical to AoOR study

df = pd.read_csv(CSV_FILE, parse_dates=["Date"]).sort_values("Date")
prices_raw = df["Price"].values.reshape(-1, 1)        # USD / bbl

# index of split point (first test row)
split_idx  = df.index[df["Date"] == SPLIT_DATE][0]
train_raw  = prices_raw[:split_idx]                   # 2 Jan 2008 … 4 Aug 2017
test_raw   = prices_raw[split_idx:]                   # 7 Aug 2017 …

# fit scaler **ONLY on training data**
scaler = MinMaxScaler().fit(train_raw)
train_scaled = scaler.transform(train_raw)
test_scaled  = scaler.transform(test_raw)

print(f"[DATA]  Train samples (days): {len(train_raw):,}")
print(f"[DATA]  Test  samples (days): {len(test_raw):,}")

# Utilities
def make_sequences(arr: np.ndarray, seq_len: int):
    """Return X, y arrays shaped (N, seq_len, 1) and (N, 1)."""
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len])
    return np.array(X), np.array(y)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom[denom == 0] = 1e-8
    return 100 * np.mean(np.abs(y_pred - y_true) / denom)

# Build TRAIN / TEST sequences (no leakage)
X_train, y_train = make_sequences(train_scaled, SEQ_LEN)

# for the test window we need the final context from TRAIN
scaled_for_test  = np.concatenate([train_scaled[-SEQ_LEN:], test_scaled])
X_test,  y_test  = make_sequences(scaled_for_test, SEQ_LEN)

print(f"[SEQ ]  Train sequences : {len(X_train):,}")
print(f"[SEQ ]  Test  sequences : {len(X_test):,}")

# Define the base LSTM model and training routine
class PriceLSTM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=2, dropout=0, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_one(net, X, y, epochs=60, lr=1e-3, bs=128, tag=""):
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        idx = torch.randperm(len(X_t))
        ep_loss = 0.0
        for i in range(0, len(X_t), bs):
            b = idx[i:i+bs]
            optim.zero_grad()
            loss = lossf(net(X_t[b]), y_t[b])
            loss.backward()
            optim.step()
            ep_loss += loss.item() * len(b)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[{tag}]  Epoch {ep:3d}/{epochs}  "
                  f"loss {ep_loss/len(X_t):.6f}")
    dur = time.time() - t0
    print(f"[{tag}]  Finished in {dur:.1f}s\n")
    return net

# Fit base learners once (no CV) + collect meta-features
HIDDEN_SET   = [32, 64, 128, 256, 512, 1024]
train_meta   = np.zeros((len(X_train), len(HIDDEN_SET)))
test_meta    = np.zeros((len(X_test),  len(HIDDEN_SET)))
base_preds   = []                       # scaled test preds for evaluation

for k, h in enumerate(HIDDEN_SET, 1):
    print(f"\n[FIT]  LSTM{h} — training on {len(X_train):,} seq")
    net = PriceLSTM(h)
    net = train_one(net, X_train, y_train, tag=f"LSTM{h}")

    with torch.no_grad():
        X_tr_gpu = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_te_gpu = torch.tensor(X_test,  dtype=torch.float32, device=device)
        train_meta[:, k-1] = net(X_tr_gpu).cpu().numpy().squeeze()
        test_meta[:,  k-1] = net(X_te_gpu).cpu().numpy().squeeze()
        base_preds.append(test_meta[:, k-1].copy())

# SVR stack (leak-free: trained on TRAIN meta)
svr = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.001)
svr.fit(train_meta, y_train.ravel())
stack_pred_scaled = svr.predict(test_meta)
print("[STACK]  SVR meta-learner trained.")

# Multi-horizon evaluation  (1-, 5-, 22-day ahead)
horizons = [1, 5, 22]                      # days ahead
truth_usd = scaler.inverse_transform(y_test).squeeze()

def eval_one(name, pred_scaled):
    pred_usd = scaler.inverse_transform(pred_scaled.reshape(-1,1)).squeeze()
    for h in horizons:
        if h >= len(pred_usd): continue
        y_h = truth_usd[h:]
        f_h = pred_usd[:-h]
        mae  = mean_absolute_error(y_h, f_h)
        mse  = mean_squared_error(y_h, f_h)
        s    = smape(y_h, f_h)
        print(f"{name:<8}  h={h:2d}  MAE {mae:8.2f}  MSE {mse:10.2f}  SMAPE {s:6.2f}%")

print("\n=== Multi-horizon metrics (USD) ===")
for name, pred in zip([f"LSTM{h}" for h in HIDDEN_SET], base_preds):
    eval_one(name, pred)
if stack_pred_scaled is not None:
    eval_one("STACK", stack_pred_scaled)

# Plot: stacked vs. actual (next-day horizon, if stack is used)
plot_pred_scaled = (stack_pred_scaled if stack_pred_scaled is not None
                    else base_preds[-1])
plot_pred = scaler.inverse_transform(plot_pred_scaled.reshape(-1,1)).squeeze()

test_dates = df["Date"].iloc[split_idx: ].values
plt.figure(figsize=(12,6))
plt.plot(test_dates, truth_usd, label="Actual", color="blue")
plt.plot(test_dates, plot_pred, label="Prediction", color="red", linestyle="--")
plt.title("Oil-price forecast - paper protocol")
plt.xlabel("Date"); plt.ylabel("WTI price (USD)"); plt.legend(); plt.tight_layout()

fname = os.path.join(FIG_DIR, "pred_vs_actual.png")
plt.savefig(fname, dpi=300); plt.close()
print(f"[PLOT]  Saved {fname}")

for h_units, pred_scaled in zip(HIDDEN_SET, base_preds):
    pred_usd = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).squeeze()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, truth_usd, label="Actual", color="blue")
    plt.plot(test_dates, pred_usd,  label=f"LSTM{h_units} Prediction",
             color="purple", linestyle="--")
    plt.title(f"LSTM (hidden={h_units}) – Test Window")
    plt.xlabel("Date"); plt.ylabel("WTI price (USD)"); plt.legend(); plt.tight_layout()

    fname = os.path.join(FIG_DIR, f"test_LSTM{h_units}.png")
    plt.savefig(fname, dpi=300); plt.close()
    print(f"[PLOT]  Saved {fname}")
    
pred_usd = scaler.inverse_transform(stack_pred_scaled.reshape(-1,1)).squeeze()

plt.figure(figsize=(12, 6))
plt.plot(test_dates, truth_usd, label="Actual", color="blue")
plt.plot(test_dates, pred_usd,  label="SVR Stack", color="red", linestyle="--")
plt.title("SVR Stack – Test Window"); plt.xlabel("Date"); plt.ylabel("WTI price (USD)")
plt.legend(); plt.tight_layout()

fname = os.path.join(FIG_DIR, "test_STACK.png")
plt.savefig(fname, dpi=300); plt.close()
print(f"[PLOT]  Saved {fname}") 