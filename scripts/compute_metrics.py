#compute_metrics.py

import json
import argparse
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MambaTS

FEATURE_COLS = ["log_range", "log_ret_close", "log_ret_volume"]
TARGET_COL = "target_log_range_next"


# ----------------------------
# Dataset (same logic as training)
# ----------------------------
class MultiStockWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: List[str],
        target_col: str,
        ticker_to_id: Dict[str, int],
    ):
        self.seq_len = int(seq_len)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.X_list, self.y_list, self.tid_list, self.index = [], [], [], []
        for ticker, g in df.groupby("ticker", sort=False):
            if ticker not in ticker_to_id:
                continue
            tid = ticker_to_id[ticker]

            X = g[feature_cols].to_numpy(np.float32)
            y = g[target_col].to_numpy(np.float32)

            n = len(X) - self.seq_len + 1
            if n <= 0:
                continue

            sid = len(self.X_list)
            self.X_list.append(X)
            self.y_list.append(y)
            self.tid_list.append(tid)
            for i in range(n):
                self.index.append((sid, i))

        if not self.index:
            raise ValueError("No valid windows. Check seq_len or input data.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        sid, i = self.index[k]
        X = self.X_list[sid]
        y = self.y_list[sid]
        tid = self.tid_list[sid]

        xw = X[i : i + self.seq_len]          # (T,3)
        yw = y[i : i + self.seq_len][:, None] # (T,1)
        return torch.from_numpy(xw), torch.from_numpy(yw), torch.tensor(tid, dtype=torch.long)


# ----------------------------
# Split / Target
# ----------------------------
def add_next_logrange_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df[TARGET_COL] = df.groupby("ticker", sort=False)["log_range"].shift(-1)
    return df


def split_by_date_3way(df: pd.DataFrame, train=0.7, val=0.15, test=0.15):
    dates = np.sort(df["date"].unique())
    n = len(dates)
    i1 = int(train * n)
    i2 = int((train + val) * n)
    i1 = min(max(1, i1), n - 2)
    i2 = min(max(i1 + 1, i2), n - 1)
    cut1 = dates[i1]
    cut2 = dates[i2]
    train_df = df[df["date"] < cut1].copy()
    val_df = df[(df["date"] >= cut1) & (df["date"] < cut2)].copy()
    test_df = df[df["date"] >= cut2].copy()
    return train_df, val_df, test_df


# ----------------------------
# Metric accumulation (streaming)
# ----------------------------
@dataclass
class Accum:
    n_sym: int
    sse: np.ndarray
    sae: np.ndarray
    cnt: np.ndarray
    sse_all: float = 0.0
    sae_all: float = 0.0
    n_all: int = 0

    @staticmethod
    def create(n_sym: int) -> "Accum":
        return Accum(
            n_sym=n_sym,
            sse=np.zeros(n_sym, dtype=np.float64),
            sae=np.zeros(n_sym, dtype=np.float64),
            cnt=np.zeros(n_sym, dtype=np.int64),
        )

    def add(self, y_true: np.ndarray, y_pred: np.ndarray, tid: np.ndarray):
        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)
        tid = tid.astype(np.int64)

        ok = np.isfinite(y_true) & np.isfinite(y_pred) & (tid >= 0) & (tid < self.n_sym)
        if ok.sum() == 0:
            return

        yt = y_true[ok]
        yp = y_pred[ok]
        ti = tid[ok]
        err = yp - yt
        se = err * err
        ae = np.abs(err)

        self.sse_all += float(se.sum())
        self.sae_all += float(ae.sum())
        self.n_all += int(se.size)

        np.add.at(self.sse, ti, se)
        np.add.at(self.sae, ti, ae)
        np.add.at(self.cnt, ti, 1)

    def finalize(self) -> Dict[str, float]:
        if self.n_all == 0:
            return dict(mse_micro=np.nan, mse_macro=np.nan, mae_micro=np.nan, mae_macro=np.nan, n_points=0)

        mse_micro = self.sse_all / max(1, self.n_all)
        mae_micro = self.sae_all / max(1, self.n_all)

        m = self.cnt > 0
        mse_macro = float(np.mean(self.sse[m] / self.cnt[m])) if m.any() else float("nan")
        mae_macro = float(np.mean(self.sae[m] / self.cnt[m])) if m.any() else float("nan")

        return dict(
            mse_micro=float(mse_micro),
            mse_macro=float(mse_macro),
            mae_micro=float(mae_micro),
            mae_macro=float(mae_macro),
            n_points=int(self.n_all),
        )


# ----------------------------
# Masks
# ----------------------------
def mask_positions(seq_len: int, loss_mode: str, burn_in: int) -> np.ndarray:
    T = int(seq_len)
    if loss_mode == "full":
        return np.arange(T, dtype=np.int64)
    if loss_mode == "last":
        return np.array([T - 1], dtype=np.int64)
    if loss_mode == "burnin":
        b = int(burn_in)
        if not (0 <= b < T):
            raise ValueError(f"burn_in must satisfy 0<=burn_in<seq_len. Got burn_in={b}, T={T}")
        return np.arange(b, T, dtype=np.int64)
    raise ValueError("loss_mode must be one of: full, last, burnin")


# ----------------------------
# Baseline helpers
# ----------------------------
def rolling_mean_inclusive_torch(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    x: (B,T)
    returns m: (B,T) where m[:,t] = mean of x[:,max(0,t-w+1):t+1]
    """
    B, T = x.shape
    w = int(window)
    if w <= 1:
        return x

    ps = torch.cumsum(x, dim=1)
    m = torch.empty_like(x)

    t_small = min(w - 1, T - 1)
    if t_small >= 0:
        denom = torch.arange(1, t_small + 2, device=x.device, dtype=x.dtype)
        m[:, : t_small + 1] = ps[:, : t_small + 1] / denom[None, :]

    if w <= T:
        num = ps[:, w - 1 :] - torch.cat(
            [torch.zeros((B, 1), device=x.device, dtype=x.dtype), ps[:, : T - w]],
            dim=1,
        )
        m[:, w - 1 :] = num / float(w)

    return m


def ewma_inclusive_torch(x: torch.Tensor, lam: float) -> torch.Tensor:
    """
    EWMA over time within each window:
      ewma[:,0] = x[:,0]
      ewma[:,t] = lam*ewma[:,t-1] + (1-lam)*x[:,t]
    lam in (0,1). Larger lam => longer memory.
    """
    if not (0.0 < float(lam) < 1.0):
        raise ValueError(f"lam must be in (0,1). Got {lam}")

    B, T = x.shape
    out = torch.empty_like(x)
    out[:, 0] = x[:, 0]
    a = float(lam)
    b = 1.0 - a
    for t in range(1, T):
        out[:, t] = a * out[:, t - 1] + b * x[:, t]
    return out


# ----------------------------
# HAR fit (streaming normal equations with intercept)
# ----------------------------
def har_fit_streaming(loader: DataLoader, pos_idx: np.ndarray, lam_ridge: float, device: str) -> Tuple[np.ndarray, float]:
    """
    Fit ridge with intercept:
      y = b + w1*last + w2*mean5 + w3*mean22
    using only positions pos_idx within each window.
    """
    A = np.zeros((4, 4), dtype=np.float64)
    bvec = np.zeros((4,), dtype=np.float64)
    pos = torch.from_numpy(pos_idx).to(torch.long)

    for xb, yb, _tid in loader:
        xb = xb.to(device).float()             # (B,T,3)
        yb = yb.to(device).float().squeeze(-1) # (B,T)

        r = xb[:, :, 0]
        m5 = rolling_mean_inclusive_torch(r, 5)
        m22 = rolling_mean_inclusive_torch(r, 22)

        last = r.index_select(1, pos)
        f5 = m5.index_select(1, pos)
        f22 = m22.index_select(1, pos)
        yt = yb.index_select(1, pos)

        X = torch.stack([last, f5, f22], dim=-1).reshape(-1, 3).detach().cpu().numpy().astype(np.float64)
        y = yt.reshape(-1).detach().cpu().numpy().astype(np.float64)

        z = np.empty((X.shape[0], 4), dtype=np.float64)
        z[:, 0] = 1.0
        z[:, 1:] = X

        A += z.T @ z
        bvec += z.T @ y

    reg = np.zeros((4, 4), dtype=np.float64)
    reg[1:, 1:] = float(lam_ridge) * np.eye(3, dtype=np.float64)

    w_full = np.linalg.solve(A + reg, bvec)  # (4,)
    b0 = float(w_full[0])
    w = w_full[1:].copy()
    return w, b0


# ----------------------------
# Evaluation functions (TEST only)
# ----------------------------
@torch.no_grad()
def eval_mamba(model: MambaTS, loader: DataLoader, pos_idx: np.ndarray, n_sym: int, device: str) -> Dict[str, float]:
    acc = Accum.create(n_sym)
    pos = torch.from_numpy(pos_idx).to(torch.long)

    for xb, yb, tid in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        tid = tid.cpu().numpy().astype(np.int64)

        yhat, _ = model(xb, return_hidden_states=False, return_ssm_states=False)
        yp = yhat.squeeze(-1).index_select(1, pos).detach().cpu().numpy().reshape(-1)
        yt = yb.squeeze(-1).index_select(1, pos).detach().cpu().numpy().reshape(-1)

        tid_rep = np.repeat(tid, pos_idx.size)
        acc.add(yt, yp, tid_rep)

    return acc.finalize()


@torch.no_grad()
def eval_baseline_persist(loader: DataLoader, pos_idx: np.ndarray, n_sym: int, device: str) -> Dict[str, float]:
    acc = Accum.create(n_sym)
    pos = torch.from_numpy(pos_idx).to(torch.long)

    for xb, yb, tid in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        tid = tid.cpu().numpy().astype(np.int64)

        r = xb[:, :, 0]
        yp = r.index_select(1, pos).detach().cpu().numpy().reshape(-1)
        yt = yb.squeeze(-1).index_select(1, pos).detach().cpu().numpy().reshape(-1)

        tid_rep = np.repeat(tid, pos_idx.size)
        acc.add(yt, yp, tid_rep)

    return acc.finalize()


@torch.no_grad()
def eval_baseline_ewma(loader: DataLoader, pos_idx: np.ndarray, lam: float, n_sym: int, device: str) -> Dict[str, float]:
    acc = Accum.create(n_sym)
    pos = torch.from_numpy(pos_idx).to(torch.long)

    for xb, yb, tid in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        tid = tid.cpu().numpy().astype(np.int64)

        r = xb[:, :, 0]
        ew = ewma_inclusive_torch(r, lam=float(lam))
        yp = ew.index_select(1, pos).detach().cpu().numpy().reshape(-1)
        yt = yb.squeeze(-1).index_select(1, pos).detach().cpu().numpy().reshape(-1)

        tid_rep = np.repeat(tid, pos_idx.size)
        acc.add(yt, yp, tid_rep)

    return acc.finalize()


@torch.no_grad()
def eval_baseline_har(
    loader: DataLoader,
    pos_idx: np.ndarray,
    har_w: np.ndarray,
    har_b: float,
    n_sym: int,
    device: str,
) -> Dict[str, float]:
    acc = Accum.create(n_sym)
    pos = torch.from_numpy(pos_idx).to(torch.long)
    w = torch.from_numpy(har_w.astype(np.float32)).to(device).view(1, 1, 3)
    b = float(har_b)

    for xb, yb, tid in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        tid = tid.cpu().numpy().astype(np.int64)

        r = xb[:, :, 0]
        m5 = rolling_mean_inclusive_torch(r, 5)
        m22 = rolling_mean_inclusive_torch(r, 22)

        last = r.index_select(1, pos)
        f5 = m5.index_select(1, pos)
        f22 = m22.index_select(1, pos)

        X = torch.stack([last, f5, f22], dim=-1)
        yp = (X * w).sum(dim=-1) + b

        yp = yp.detach().cpu().numpy().reshape(-1)
        yt = yb.squeeze(-1).index_select(1, pos).detach().cpu().numpy().reshape(-1)

        tid_rep = np.repeat(tid, pos_idx.size)
        acc.add(yt, yp, tid_rep)

    return acc.finalize()


# ----------------------------
# Run discovery + I/O
# ----------------------------
def load_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(runs_root: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        cfg = os.path.join(run_dir, "config.json")
        ckpt = os.path.join(run_dir, "model_best.pt")
        if os.path.exists(cfg) and os.path.exists(ckpt):
            out.append(run_dir)
    return out


def run_tag(seq_len: int, loss_mode: str, burn_in: int) -> str:
    if loss_mode == "burnin":
        return f"seq{seq_len}_burnin{burn_in}"
    return f"seq{seq_len}_{loss_mode}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, help="Folder containing the run directories.")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--har_lambda", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dirs = discover_runs(args.runs_root)
    if not run_dirs:
        raise RuntimeError(f"No runs found under {args.runs_root} (expected config.json + model_best.pt per run).")

    metrics_root = os.path.join(args.runs_root, "metrics")
    ensure_dir(metrics_root)

    for run_dir in run_dirs:
        cfg = load_json_file(os.path.join(run_dir, "config.json"))
        ckpt_path = os.path.join(run_dir, "model_best.pt")
        run_name = os.path.basename(run_dir)

        seq_len = int(cfg["seq_len"])
        loss_mode = str(cfg["loss_mode"])
        burn_in = int(cfg.get("burn_in", 0))

        dataset_csv = str(cfg["dataset_csv"])
        split_train = float(cfg.get("split_train", 0.7))
        split_val = float(cfg.get("split_val", 0.15))
        split_test = float(cfg.get("split_test", 0.15))

        df = pd.read_csv(dataset_csv)
        df["date"] = pd.to_datetime(df["date"], utc=False)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df = add_next_logrange_target(df).dropna(subset=[TARGET_COL]).reset_index(drop=True)

        tr_df, va_df, te_df = split_by_date_3way(df, train=split_train, val=split_val, test=split_test)

        train_tickers = sorted(tr_df["ticker"].unique().tolist())
        ticker_to_id = {t: i for i, t in enumerate(train_tickers)}
        n_sym = len(train_tickers)

        tr_ds = MultiStockWindowDataset(tr_df, seq_len, FEATURE_COLS, TARGET_COL, ticker_to_id)
        te_ds = MultiStockWindowDataset(te_df, seq_len, FEATURE_COLS, TARGET_COL, ticker_to_id)

        def make_loader(ds):
            return DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                pin_memory=(device == "cuda"),
            )

        train_loader = make_loader(tr_ds)  # for HAR fit
        test_loader = make_loader(te_ds)   # for evaluation

        model = MambaTS(
            d_in=int(cfg["d_in"]),
            d_model=int(cfg["d_model"]),
            n_layers=int(cfg["n_layers"]),
            d_state=int(cfg["d_state"]),
            expand=int(cfg["expand"]),
            conv_kernel=int(cfg["conv_kernel"]),
            use_bias=bool(cfg.get("use_bias", False)),
            use_conv_bias=bool(cfg.get("use_conv_bias", True)),
            hidden_act=str(cfg.get("hidden_act", "silu")),
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        pos_full = np.arange(seq_len, dtype=np.int64)
        pos_trainmask = mask_positions(seq_len, loss_mode, burn_in)
        eval_masks = {"full": pos_full, "trainmask": pos_trainmask}

        # Fit HAR per eval_mask on TRAIN
        har_params = {}
        for em_name, pos_idx in eval_masks.items():
            w, b0 = har_fit_streaming(train_loader, pos_idx, lam_ridge=args.har_lambda, device=device)
            har_params[em_name] = (w, b0)

        # Evaluate ONLY TEST
        rows = []
        split = "test"
        for em_name, pos_idx in eval_masks.items():
            m = eval_mamba(model, test_loader, pos_idx, n_sym, device)
            rows.append({
                "run_name": run_name,
                "seq_len": seq_len,
                "loss_mode": loss_mode,
                "burn_in": burn_in,
                "split": split,
                "eval_mask": em_name,
                "method": "mamba",
                **m
            })

            b1 = eval_baseline_persist(test_loader, pos_idx, n_sym, device)
            rows.append({
                "run_name": run_name, "seq_len": seq_len, "loss_mode": loss_mode, "burn_in": burn_in,
                "split": split, "eval_mask": em_name, "method": "baseline_persist",
                **b1
            })

            b2 = eval_baseline_ewma(test_loader, pos_idx, lam=0.2, n_sym=n_sym, device=device)
            rows.append({
                "run_name": run_name, "seq_len": seq_len, "loss_mode": loss_mode, "burn_in": burn_in,
                "split": split, "eval_mask": em_name, "method": "baseline_ewma_lam0p2",
                **b2
            })

            b3 = eval_baseline_ewma(test_loader, pos_idx, lam=0.94, n_sym=n_sym, device=device)
            rows.append({
                "run_name": run_name, "seq_len": seq_len, "loss_mode": loss_mode, "burn_in": burn_in,
                "split": split, "eval_mask": em_name, "method": "baseline_ewma_lam0p94",
                **b3
            })

            hw, hb = har_params[em_name]
            b4 = eval_baseline_har(test_loader, pos_idx, hw, hb, n_sym, device)
            rows.append({
                "run_name": run_name, "seq_len": seq_len, "loss_mode": loss_mode, "burn_in": burn_in,
                "split": split, "eval_mask": em_name, "method": "baseline_har_1_5_22",
                **b4
            })

        out_df = pd.DataFrame(rows)

        tag = run_tag(seq_len, loss_mode, burn_in)
        out_dir = os.path.join(metrics_root, tag)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, "metrics_test.csv")
        out_df.to_csv(out_path, index=False)

        print(f"[saved] {out_path}")

    print(f"[done] all metrics saved under: {metrics_root}")


if __name__ == "__main__":
    main()