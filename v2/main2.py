#!/usr/bin/env python3
"""
Ensemble Methods for Forecasting Oil Prices in Times of Crisis (PyTorch)
------------------------------------------------------------------------
Fix: **Non‑constant stacked output**
----------------------------------
The SVR meta‑learner was occasionally collapsing to a single constant
because the scale of its input features (base learners’ predictions) was
very small and unstandardised. We now:

1. **Standardise** base‑prediction features with `StandardScaler` before
   fitting/predicting with the SVR.
2. Boost SVR capacity (`C=1 000`, `gamma="auto"`).

Everything else—paths, splits, plotting—remains unchanged. Run the script
and you should see a properly varying “Stacked” curve.
"""
from __future__ import annotations

# ------------------------------------------------------------------
# 0. Constants
# ------------------------------------------------------------------
from pathlib import Path
CSV_PATH     = Path("wti_daily.csv")
PRICE_COLUMN = "Price"
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
LOOKBACK     = 30
RNG_SEED     = 42

# ------------------------------------------------------------------
# 1. Imports & seeds
# ------------------------------------------------------------------
import random, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 2. Data utilities
# ------------------------------------------------------------------

def load_and_scale(csv: Path, col: str):
    df = pd.read_csv(csv, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    prices = df[col].values.reshape(-1,1)
    mm = MinMaxScaler(); scaled = mm.fit_transform(prices).astype(np.float32)
    return scaled, mm

def make_sequences(arr: np.ndarray, lookback: int):
    X,y=[],[]
    for t in range(lookback,len(arr)):
        X.append(arr[t-lookback:t]); y.append(arr[t])
    return np.array(X), np.array(y)

class SeqDS(Dataset):
    def __init__(self,X,y): self.X=torch.from_numpy(X); self.y=torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

# ------------------------------------------------------------------
# 3. LSTM model
# ------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self,hid:int,do:float):
        super().__init__(); self.lstm=nn.LSTM(1,hid,batch_first=True); self.do=nn.Dropout(do); self.fc=nn.Linear(hid,1)
    def forward(self,x): out,_=self.lstm(x); out=self.do(out[:,-1,:]); return self.fc(out)

# ------------------------------------------------------------------
# 4. Training helpers
# ------------------------------------------------------------------

def train(net, tr_loader, val_loader, lr=1e-3, epochs=200, patience=15):
    crit=nn.MSELoss(); opt=torch.optim.Adam(net.parameters(),lr=lr)
    best=np.inf; wait=0; best_state=None
    for _ in range(epochs):
        net.train()
        for xb,yb in tr_loader:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); loss=crit(net(xb), yb); loss.backward(); opt.step()
        with torch.no_grad():
            net.eval(); v=np.mean([crit(net(xv.to(dev)), yv.to(dev)).item() for xv,yv in val_loader])
        if v < best-1e-6: best,wait=v,0; best_state={k:v.clone() for k,v in net.state_dict().items()}
        else: wait+=1;  
        if wait>=patience: break
    net.load_state_dict(best_state); return net

def predict(net, loader):
    net.eval(); out=[]
    with torch.no_grad():
        for xb,_ in loader: out.append(net(xb.to(dev)).cpu().numpy())
    return np.vstack(out).flatten()

# ------------------------------------------------------------------
# 5. Ensemble pipeline
# ------------------------------------------------------------------

def train_base(X_tr,y_tr,X_val,y_val,grid):
    ms=[]
    for p in grid:
        print("Training",p)
        m=LSTM(p['hidden'],p['drop']); m.to(dev)
        trL=DataLoader(SeqDS(X_tr,y_tr),32,shuffle=True)
        vlL=DataLoader(SeqDS(X_val,y_val),32,shuffle=False)
        ms.append(train(m,trL,vlL,lr=p['lr']))
    return ms

def meta_features(models,X):
    loader=DataLoader(SeqDS(X,np.zeros((len(X),1),dtype=np.float32)),64,shuffle=False)
    return np.column_stack([predict(m,loader) for m in models])

# ------------------------------------------------------------------
# 6. Evaluate + plot
# ------------------------------------------------------------------

def eval_plot(y_true, base, meta):
    print("\n==== Test MAPE ====")
    for k in range(base.shape[1]):
        print(f"LSTM{k+1}: {mean_absolute_percentage_error(y_true,base[:,k])*100:6.2f}%")
    print(f"Stacked : {mean_absolute_percentage_error(y_true,meta)*100:6.2f}%")
    plt.figure(figsize=(10,5)); plt.plot(y_true,label='True')
    for k in range(base.shape[1]): plt.plot(base[:,k],label=f'LSTM{k+1}')
    plt.plot(meta,label='Stacked'); plt.tight_layout(); plt.legend(); plt.show()

# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------

def main():
    series,mm=load_and_scale(CSV_PATH,PRICE_COLUMN)
    X,y=make_sequences(series,LOOKBACK)
    X_tmp,X_test,y_tmp,y_test=train_test_split(X,y,test_size=TEST_SIZE,shuffle=False)
    X_tr,X_val,y_tr,y_val=train_test_split(X_tmp,y_tmp,test_size=VAL_SIZE/(1-TEST_SIZE),shuffle=False)

    GRID=[{'hidden':32,'drop':0.1,'lr':1e-3},{'hidden':64,'drop':0.2,'lr':5e-4},{'hidden':128,'drop':0.3,'lr':1e-4}]
    base=train_base(X_tr,y_tr,X_val,y_val,GRID)

    Xmeta_tr=meta_features(base,X_val)
    Xmeta_te=meta_features(base,X_test)

    # --- NEW: standardise meta features ---
    ss=StandardScaler(); Xm_tr_s=ss.fit_transform(Xmeta_tr); Xm_te_s=ss.transform(Xmeta_te)
    svr=SVR(kernel='rbf',C=1000.0,gamma='auto').fit(Xm_tr_s,y_val.flatten())
    meta_pred=svr.predict(Xm_te_s)

    base_inv=mm.inverse_transform(Xmeta_te)
    meta_inv=mm.inverse_transform(meta_pred.reshape(-1,1))
    y_true_inv=mm.inverse_transform(y_test)

    eval_plot(y_true_inv,base_inv,meta_inv)

if __name__=='__main__':
    main()

