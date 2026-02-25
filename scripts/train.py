#train.py

import os
import json
import time
import sys
import argparse
from dataclasses import asdict, dataclass
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
# Dataset
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
        if "ticker" not in df.columns:
            raise ValueError("Multi-stock CSV must contain a 'ticker' column.")
        if "date" not in df.columns:
            raise ValueError("Multi-stock CSV must contain a 'date' column.")

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

        xw = X[i : i + self.seq_len]
        # y is next-step target aligned per timestep (see add_next_logrange_target)
        yw = y[i : i + self.seq_len][:, None]
        return torch.from_numpy(xw), torch.from_numpy(yw), torch.tensor(tid, dtype=torch.long)


# ----------------------------
# Splitting / Target
# ----------------------------
def split_by_date_3way(df: pd.DataFrame, train=0.7, val=0.15, test=0.15):
    if train <= 0 or val <= 0 or test <= 0 or abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("train/val/test must be >0 and sum to 1.")

    dates = np.sort(df["date"].unique())
    n = len(dates)
    if n < 3:
        raise ValueError("Not enough unique dates for a 3-way split.")

    i1 = int(train * n)
    i2 = int((train + val) * n)

    i1 = min(max(1, i1), n - 2)
    i2 = min(max(i1 + 1, i2), n - 1)

    cut1 = dates[i1]  # train: date < cut1
    cut2 = dates[i2]  # val:   cut1 <= date < cut2 ; test: date >= cut2

    train_df = df[df["date"] < cut1].copy()
    val_df = df[(df["date"] >= cut1) & (df["date"] < cut2)].copy()
    test_df = df[df["date"] >= cut2].copy()
    return train_df, val_df, test_df, cut1, cut2


def add_next_logrange_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df[TARGET_COL] = df.groupby("ticker", sort=False)["log_range"].shift(-1)
    return df


# ----------------------------
# Loss / Eval
# ----------------------------
def masked_mse(y_hat: torch.Tensor, y: torch.Tensor, mode: str, burn_in: int = 0) -> torch.Tensor:
    if mode == "full":
        return torch.mean((y_hat - y) ** 2)
    if mode == "last":
        return torch.mean((y_hat[:, -1:, :] - y[:, -1:, :]) ** 2)
    if mode == "burnin":
        if burn_in < 0 or burn_in >= y.size(1):
            raise ValueError("Invalid burn_in for burnin mode.")
        return torch.mean((y_hat[:, burn_in:, :] - y[:, burn_in:, :]) ** 2)
    raise ValueError("mode must be one of: full, last, burnin")


@torch.no_grad()
def eval_loader(model: MambaTS, loader: DataLoader, device: str, loss_mode: str, burn_in: int) -> float:
    model.eval()
    s = 0.0
    for xb, yb, _tid in loader:
        xb, yb = xb.to(device), yb.to(device)
        y_hat, _ = model(xb, return_hidden_states=False, return_ssm_states=False)
        s += float(masked_mse(y_hat, yb, mode=loss_mode, burn_in=burn_in).item())
    return s / max(1, len(loader))


# ----------------------------
# Config / Logging
# ----------------------------
@dataclass
class RunConfig:
    dataset_csv: str
    out_dir: str
    run_name: str
    seed: int

    seq_len: int
    batch_size: int
    epochs: int
    lr: float

    d_in: int
    d_model: int
    n_layers: int
    d_state: int
    expand: int
    conv_kernel: int
    use_bias: bool
    use_conv_bias: bool
    hidden_act: str

    loss_mode: str
    burn_in: int
    grad_clip: float
    num_workers: int

    split_train: float
    split_val: float
    split_test: float


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def append_metrics_csv(path: str, row: Dict) -> None:
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # These make runs more deterministic at the cost of speed in some cases:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Main
# ----------------------------
def main(args=None):
    p = argparse.ArgumentParser()

    # data / output
    p.add_argument("--dataset_csv", type=str, default="dataset_3feat_alltickers.csv")
    p.add_argument("--out_dir", type=str, default="overriedinsweep")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    # sweepable
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--loss_mode", type=str, choices=["full", "last", "burnin"], default="burnin")
    p.add_argument("--burn_in", type=int, default=16)

    # optional overrides
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)

    a = p.parse_args(args)

    run_name = a.run_name or f"mamba_logrange_3feat_T{a.seq_len}_{a.loss_mode}_b{a.burn_in}_seed{a.seed}"

    cfg = RunConfig(
        dataset_csv=a.dataset_csv,
        out_dir=a.out_dir,
        run_name=run_name,
        seed=a.seed,

        seq_len=a.seq_len,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,

        d_in=3,
        d_model=4,
        n_layers=8,
        d_state=4,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",

        loss_mode=a.loss_mode,
        burn_in=a.burn_in if a.loss_mode == "burnin" else 0,
        grad_clip=1.0,
        num_workers=0,

        split_train=0.7,
        split_val=0.15,
        split_test=0.15,
    )

    if cfg.loss_mode == "burnin":
        if not (0 <= cfg.burn_in < cfg.seq_len):
            raise ValueError("burn_in must satisfy 0 <= burn_in < seq_len for burnin mode.")

    os.makedirs(cfg.out_dir, exist_ok=True)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # persist config immediately
    save_json(os.path.join(run_dir, "config.json"), asdict(cfg))

    set_determinism(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load data ----
    df = pd.read_csv(cfg.dataset_csv)

    required = {"ticker", "date", *FEATURE_COLS}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Ensure datetime for split, but keep ordering stable
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # add target and drop last row per ticker (where shift(-1) is NaN)
    df = add_next_logrange_target(df).dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_df, val_df, test_df, cut1, cut2 = split_by_date_3way(
        df, train=cfg.split_train, val=cfg.split_val, test=cfg.split_test
    )

    print("Date cuts:", cut1, "|", cut2)
    print("Rows train/val/test:", len(train_df), len(val_df), len(test_df))

    # ticker_to_id from TRAIN ONLY
    train_tickers = sorted(train_df["ticker"].unique().tolist())
    ticker_to_id = {t: i for i, t in enumerate(train_tickers)}
    save_json(os.path.join(run_dir, "ticker_to_id.json"), ticker_to_id)

    # datasets/loaders
    train_ds = MultiStockWindowDataset(train_df, cfg.seq_len, FEATURE_COLS, TARGET_COL, ticker_to_id)
    val_ds = MultiStockWindowDataset(val_df, cfg.seq_len, FEATURE_COLS, TARGET_COL, ticker_to_id)
    test_ds = MultiStockWindowDataset(test_df, cfg.seq_len, FEATURE_COLS, TARGET_COL, ticker_to_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # model/opt
    model = MambaTS(
        d_in=cfg.d_in,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        expand=cfg.expand,
        conv_kernel=cfg.conv_kernel,
        use_bias=cfg.use_bias,
        use_conv_bias=cfg.use_conv_bias,
        hidden_act=cfg.hidden_act,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    best_ckpt_path = os.path.join(run_dir, "model_best.pt")
    last_ckpt_path = os.path.join(run_dir, "model_last.pt")

    best_val = float("inf")
    global_step = 0

    # ---- train ----
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        tr_sum = 0.0

        for xb, yb, _tid in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            y_hat, _ = model(xb, return_hidden_states=False, return_ssm_states=False)
            loss = masked_mse(y_hat, yb, mode=cfg.loss_mode, burn_in=cfg.burn_in)
            loss.backward()

            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            tr_sum += float(loss.item())
            global_step += 1

        train_loss = tr_sum / max(1, len(train_loader))
        val_loss = eval_loader(model, val_loader, device, cfg.loss_mode, cfg.burn_in)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "best_val": best_val,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )

        # Always save "last"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "best_val": best_val,
                "epoch": epoch,
            },
            last_ckpt_path,
        )

        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
            "is_best": int(is_best),
            "seconds": dt,
            "global_step": global_step,
        }
        append_metrics_csv(metrics_path, row)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | train={train_loss:.6f} | val={val_loss:.6f} "
            f"| best={best_val:.6f} | {dt:.1f}s"
        )

    # ---- final test (load best) ----
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    test_loss = eval_loader(model, test_loader, device, cfg.loss_mode, cfg.burn_in)
    save_json(os.path.join(run_dir, "final.json"), {"test_loss": test_loss, "best_val": best_val})
    print(f"TEST | loss={test_loss:.6f}")
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()