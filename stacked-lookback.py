# Imports, reproducibility, folders 
import os, time, random
import numpy  as np
import pandas as pd
import torch, torch.nn as nn
from datetime               import datetime
from sklearn.preprocessing  import MinMaxScaler
from sklearn.svm            import SVR
from sklearn.metrics        import mean_squared_error, mean_absolute_error
import matplotlib.pyplot    as plt

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

FIG_DIR = "lookback_figs"; os.makedirs(FIG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT]  device={device}")

# Parameters you may want to tweak 
SEQ_SET    = [14, 30, 60, 120, 180]               # heterogeneous windows
HIDDEN_SET = [512, 512, 512, 512, 512]              # one per window (same length!)
EPOCHS     = 60                               # per LSTM
SPLIT_DATE = datetime(2017, 8, 4)             # AoOR protocol
CSV_FILE   = "wti_daily.csv"

assert len(SEQ_SET) == len(HIDDEN_SET), "SEQ_SET and HIDDEN_SET must match!"

# Load & split raw prices 
df   = pd.read_csv(CSV_FILE, parse_dates=["Date"]).sort_values("Date")
price_raw = df["Price"].values.reshape(-1, 1)

split_idx = df.index[df["Date"] == SPLIT_DATE][0]        # first test row
train_raw = price_raw[:split_idx]
test_raw  = price_raw[split_idx:]

scaler = MinMaxScaler().fit(train_raw)                   # TRAIN-only fit
train_scaled = scaler.transform(train_raw)
test_scaled  = scaler.transform(test_raw)

print(f"[DATA] train days={len(train_raw):,}  test days={len(test_raw):,}")

# Helper functions 
def make_seq(arr, L):
    X, y = [], []
    for i in range(len(arr) - L):
        X.append(arr[i:i+L])
        y.append(arr[i+L])
    return np.array(X), np.array(y)

def smape(a, f):
    denom = (np.abs(a) + np.abs(f)) / 2
    denom[denom == 0] = 1e-8
    return 100*np.mean(np.abs(f-a) / denom)

# Prepare common target arrays (align to longest window) 
max_L      = max(SEQ_SET)

# build once for *true* targets
X_train_max, y_train_max = make_seq(train_scaled, max_L)
scaled_for_test = np.concatenate([train_scaled[-max_L:], test_scaled])
X_test_max,  y_test_max  = make_seq(scaled_for_test, max_L)

# These y_*_max are the reference targets every learner must match
len_tr = len(y_train_max);  len_te = len(y_test_max)
print(f"[ALIGN] common train seq={len_tr:,}  test seq={len_te:,}")

# LSTM definition & training util 
class PriceLSTM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=2, dropout=0, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1, :])

def fit_lstm(model, X, y, tag="", epochs=EPOCHS):
    model.to(device)
    opt  = torch.optim.Adam(model.parameters(), 1e-3)
    loss = nn.MSELoss()
    X_t  = torch.tensor(X, dtype=torch.float32, device=device)
    y_t  = torch.tensor(y, dtype=torch.float32, device=device)

    for ep in range(1, epochs+1):
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 128):
            b = idx[i:i+128]
            opt.zero_grad(); l = loss(model(X_t[b]), y_t[b]); l.backward(); opt.step()
        if ep==1 or ep%5==0 or ep==epochs:
            print(f"[{tag}] ep {ep:3d}/{epochs} loss {l.item():.6f}")
    return model

# Train heterogeneous base learners & align predictions 
n_learn = len(SEQ_SET)
train_meta = np.zeros((len_tr, n_learn))
test_meta  = np.zeros((len_te, n_learn))
base_preds = []                                    # scaled test predictions

for j, (L, H) in enumerate(zip(SEQ_SET, HIDDEN_SET), 1):
    print(f"\n[FIT] LSTM{H}  (look-back {L}d)")
    X_tr, y_tr = make_seq(train_scaled, L)
    X_te_ctx   = np.concatenate([train_scaled[-L:], test_scaled])
    X_te, _    = make_seq(X_te_ctx, L)

    # offset so that predictions line up with the y_*_max reference
    off = max_L - L

    net = fit_lstm(PriceLSTM(H), X_tr, y_tr, tag=f"L{L}h{H}")
    with torch.no_grad():
        train_meta[:, j-1] = net(torch.tensor(X_tr, dtype=torch.float32,
                                              device=device)).cpu().numpy().squeeze()[off:]
        test_meta[:,  j-1] = net(torch.tensor(X_te, dtype=torch.float32,
                                              device=device)).cpu().numpy().squeeze()
        base_preds.append(test_meta[:, j-1].copy())

# SVR stack trained on aligned TRAIN meta 
svr = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.001)
svr.fit(train_meta, y_train_max.ravel())
stack_pred_scaled = svr.predict(test_meta)
print("[STACK] SVR meta-learner trained.")

# Multi-horizon metrics (1/5/22-day) 
truth_usd = scaler.inverse_transform(y_test_max).squeeze()
horizons = [1,5,22]

def evaluate(name, pred_scaled):
    p_usd = scaler.inverse_transform(pred_scaled.reshape(-1,1)).squeeze()
    for h in horizons:
        y_h, f_h = truth_usd[h:], p_usd[:-h]
        print(f"{name:<8} h={h:2d} "
              f"MAE {mean_absolute_error(y_h,f_h):8.2f}  "
              f"MSE {mean_squared_error(y_h,f_h):10.2f}  "
              f"SMAPE {smape(y_h,f_h):6.2f}%")

def evaluate_by_horizon():
    print("\n=== Multi-horizon metrics (USD) ===")
    models = [f"LSTM{h}L{SEQ_SET[i]}" for i, h in enumerate(HIDDEN_SET)] + ["STACK"]
    predictions = base_preds + [stack_pred_scaled]
    
    for h in horizons:
        print(f"\n--- Horizon {h} days ---")
        print("Model      & h &  MAE    &   MSE     &  SMAPE \\")
        for name, pred_scaled in zip(models, predictions):
            p_usd = scaler.inverse_transform(pred_scaled.reshape(-1,1)).squeeze()
            y_h, f_h = truth_usd[h:], p_usd[:-h]
            print(f"{name:<10} & h={h:2d} &"
                  f"{mean_absolute_error(y_h,f_h):8.2f} &"
                  f"{mean_squared_error(y_h,f_h):10.2f} &"
                  f"{smape(y_h,f_h):6.2f}\% \\")

# Replace original evaluation code with:
evaluate_by_horizon()


# Plots  – one per model + stack
test_dates = df["Date"].iloc[split_idx: ].values   # aligns to y_test_max

def save_plot(name, pred_scaled, clr):
    pred_usd = scaler.inverse_transform(pred_scaled.reshape(-1,1)).squeeze()
    plt.figure(figsize=(12,6))
    plt.plot(test_dates, truth_usd, label="Actual", color="blue")
    plt.plot(test_dates, pred_usd,  label=name, color=clr, linestyle="--")
    plt.title(f"{name} – Test Window"); plt.xlabel("Date"); plt.ylabel("WTI (USD)")
    plt.legend(); plt.tight_layout()
    fn = os.path.join(FIG_DIR, f"test_{name.replace(' ','_')}.png")
    plt.savefig(fn, dpi=300); plt.close(); print(f"[PLOT] saved {fn}")

for (L,H), pr in zip(zip(SEQ_SET,HIDDEN_SET), base_preds):
    save_plot(f"LSTM{H}_L{L}", pr, "purple")
save_plot("SVR_Stack", stack_pred_scaled, "red")