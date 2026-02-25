#analyze_states.py

import argparse, json, os, random, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MambaTS


# ----------------------------
# Shared utilities
# ----------------------------
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_outdirs(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)


def add_next_target(df: pd.DataFrame, group_col: str, src_col: str, tgt_col: str) -> pd.DataFrame:
    df = df.sort_values([group_col, "date"]).reset_index(drop=True)
    df[tgt_col] = df.groupby(group_col, sort=False)[src_col].shift(-1)
    return df


def split_by_date_3way(df: pd.DataFrame, train: float, val: float, test: float):
    if train <= 0 or val <= 0 or test <= 0 or abs(train + val + test - 1.0) > 1e-9:
        raise ValueError("train/val/test must be >0 and sum to 1.")
    dates = np.sort(pd.to_datetime(df["date"]).unique())
    n = len(dates)
    if n < 3:
        raise ValueError("Not enough unique dates for a 3-way split.")
    i1 = int(train * n)
    i2 = int((train + val) * n)
    i1 = min(max(1, i1), n - 2)
    i2 = min(max(i1 + 1, i2), n - 1)
    cut1 = dates[i1]
    cut2 = dates[i2]
    tr = df[df["date"] < cut1].copy()
    va = df[(df["date"] >= cut1) & (df["date"] < cut2)].copy()
    te = df[df["date"] >= cut2].copy()
    return tr, va, te, cut1, cut2


def compute_train_feature_stats(train_df: pd.DataFrame, feats: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for f in feats:
        x = train_df[f].astype(np.float64).to_numpy()
        stats[f] = {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=0))}
    return stats


def load_model_from_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = MambaTS(
        d_in=int(cfg.get("d_in", 3)),
        d_model=int(cfg.get("d_model", 16)),
        n_layers=int(cfg.get("n_layers", 4)),
        d_state=int(cfg.get("d_state", 16)),
        expand=int(cfg.get("expand", 2)),
        conv_kernel=int(cfg.get("conv_kernel", 4)),
        use_bias=bool(cfg.get("use_bias", False)),
        use_conv_bias=bool(cfg.get("use_conv_bias", True)),
        hidden_act=str(cfg.get("hidden_act", "silu")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def extract_ssm_traj(model: MambaTS, x_np: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    _yhat, aux = model(x, return_hidden_states=False, return_ssm_states=True)
    ssm = aux["ssm_states"]  # list[T][L] (1,interm,d_state)

    T = len(ssm)
    L = len(ssm[0])
    ex = ssm[0][0].squeeze(0).detach().cpu().numpy()
    Dflat = ex.reshape(-1).shape[0]

    H = np.empty((L, T, Dflat), dtype=np.float32)
    for t in range(T):
        for l in range(L):
            H[l, t] = ssm[t][l].squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
    return H


@torch.no_grad()
def extract_pred_traj(model: MambaTS, x_np: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    yhat, _ = model(x, return_hidden_states=False, return_ssm_states=False)
    return yhat.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)


def auc_trapz(y: np.ndarray) -> float:
    return float(np.trapz(y, dx=1.0))


def horizon_index(curve: np.ndarray, frac: float) -> Optional[int]:
    if curve.size == 0:
        return None
    thr = float(frac) * float(curve[0])
    idx = np.where(curve <= thr)[0]
    return int(idx[0]) if idx.size > 0 else None


def fit_exp_time_constant(curve: np.ndarray, min_lag: int = 0, max_lag: Optional[int] = None,
                          floor_frac: float = 1e-3) -> float:
    if curve.size < 3:
        return float("nan")
    y0 = float(curve[0])
    if not np.isfinite(y0) or y0 <= 0:
        return float("nan")
    L = curve.size
    hi = L if max_lag is None else min(L, int(max_lag) + 1)
    lo = max(0, int(min_lag))
    x = np.arange(lo, hi, dtype=np.float64)
    y = curve[lo:hi].astype(np.float64)

    floor = max(1e-30, floor_frac * y0)
    m = (y > floor) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")

    x = x[m]
    y = y[m]
    logy = np.log(y + 1e-30)

    xm = x.mean()
    ym = logy.mean()
    denom = ((x - xm) ** 2).sum()
    if denom <= 1e-30:
        return float("nan")
    slope = float(((x - xm) * (logy - ym)).sum() / denom)
    if slope >= 0:
        return float("nan")
    return float(-1.0 / slope)


def plot_lines(curves_mu: np.ndarray, curves_se: np.ndarray, title: str,
               xlabel: str, ylabel: str, outpath: str, yscale_log: bool):
    L, lag_len = curves_mu.shape
    lag = np.arange(lag_len)
    plt.figure(figsize=(9, 4))
    for l in range(L):
        mu = curves_mu[l]
        se = curves_se[l]
        plt.plot(lag, mu, label=f"layer{l}")
        plt.fill_between(lag, mu - se, mu + se, alpha=0.15)
    if yscale_log:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_heatmap(curves_mu_avg_tau: np.ndarray, title: str, outpath: str):
    plt.figure(figsize=(9, 4))
    img = np.log10(curves_mu_avg_tau + 1e-30)
    plt.imshow(img, aspect="auto", origin="lower")
    plt.colorbar(label="log10(mean)")
    plt.yticks(np.arange(img.shape[0]), [f"L{l}" for l in range(img.shape[0])])
    plt.xlabel("lag")
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def heatmap_plot(mat: np.ndarray, xlabels: List[str], ylabels: List[str], title: str, outpath: str):
    plt.figure(figsize=(10, 4.5))
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xticks(np.arange(len(xlabels)), xlabels)
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.xlabel("horizon k")
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ----------------------------
# Shared window sampling
# ----------------------------
def sample_windows(df_split: pd.DataFrame, feats: List[str], T: int, n_windows: int,
                   seed: int, per_ticker_cap: Optional[int]) -> List[Dict]:
    rng = np.random.default_rng(seed)
    tickers = sorted(df_split["ticker"].unique().tolist())
    if not tickers:
        raise ValueError("No tickers in chosen split.")

    groups = {}
    for t in tickers:
        g = df_split[df_split["ticker"] == t].sort_values("date").reset_index(drop=True)
        X = g[feats].to_numpy(np.float32)
        if len(X) >= T:
            groups[t] = X
    valid = [t for t in tickers if t in groups]
    if not valid:
        raise ValueError(f"No tickers with length >= T={T} in chosen split.")

    count = {t: 0 for t in valid}
    out = []
    while len(out) < n_windows:
        t = valid[int(rng.integers(0, len(valid)))]
        if per_ticker_cap is not None and count[t] >= per_ticker_cap:
            continue
        X = groups[t]
        n = len(X) - T + 1
        s = int(rng.integers(0, n))
        out.append({"ticker": t, "start": s, "X": X[s:s + T]})
        count[t] += 1
    return out


def sample_windows_xy(df_split: pd.DataFrame, feats: List[str], ycol: str, T: int, n_windows: int,
                      seed: int, per_ticker_cap: Optional[int]) -> List[Dict]:
    rng = np.random.default_rng(seed)
    tickers = sorted(df_split["ticker"].unique().tolist())
    if not tickers:
        raise ValueError("No tickers in chosen split.")

    groups = {}
    valid = []
    for t in tickers:
        g = df_split[df_split["ticker"] == t].sort_values("date").reset_index(drop=True)
        X = g[feats].to_numpy(np.float32)
        y = g[ycol].to_numpy(np.float32)
        if len(X) >= T and len(y) >= T:
            groups[t] = (X, y)
            valid.append(t)
    if not valid:
        raise ValueError(f"No tickers with length >= T={T} in chosen split.")

    count = {t: 0 for t in valid}
    out = []
    while len(out) < n_windows:
        t = valid[int(rng.integers(0, len(valid)))]
        if per_ticker_cap is not None and count[t] >= per_ticker_cap:
            continue
        X, y = groups[t]
        n = len(X) - T + 1
        s = int(rng.integers(0, n))
        out.append({"ticker": t, "start": s, "X": X[s:s + T], "y": y[s:s + T]})
        count[t] += 1
    return out


# ----------------------------
# Module (1): analyze_ssm_states.py (same logic)
# ----------------------------
@dataclass(frozen=True)
class SSMCfg:
    split_name: str
    feature_name: str
    taus: Tuple[int, ...]
    alphas: Tuple[float, ...]
    n_windows: int
    per_ticker_cap: Optional[int]
    horizon_frac: float
    null_noise_draws: int
    fit_min_lag: int
    fit_max_lag: Optional[int]
    fit_floor_frac: float
    yscale_log: bool


def run_ssm_analysis(outdir: str, model: MambaTS, device: str,
                     feats: List[str], target_col: str, target_src_col: str,
                     tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
                     cuts: Tuple[str, str], T: int, d_in: int, ssmcfg: SSMCfg, seed: int):

    split_df = {"train": tr, "val": va, "test": te}[ssmcfg.split_name].sort_values(["ticker", "date"]).reset_index(drop=True)
    stats = compute_train_feature_stats(tr, feats)
    with open(os.path.join(outdir, "train_feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if ssmcfg.feature_name not in feats:
        raise ValueError("feature_name not in feats")
    feat_idx = feats.index(ssmcfg.feature_name)
    feat_std = float(stats[ssmcfg.feature_name]["std"])
    if feat_std <= 0 or not np.isfinite(feat_std):
        raise ValueError(f"Bad std for {ssmcfg.feature_name}: {feat_std}")

    windows = sample_windows(split_df, feats, T=T, n_windows=ssmcfg.n_windows,
                             seed=seed, per_ticker_cap=ssmcfg.per_ticker_cap)

    H0_list, Y0_list = [], []
    for w in windows:
        X = w["X"]
        H0_list.append(extract_ssm_traj(model, X, device))
        Y0_list.append(extract_pred_traj(model, X, device))
    L = H0_list[0].shape[0]
    Dflat = H0_list[0].shape[2]

    state_ir_summary_rows = []
    curves_store: Dict[Tuple[float, int], np.ndarray] = {}

    for alpha in ssmcfg.alphas:
        eps = float(alpha * feat_std)
        for tau in ssmcfg.taus:
            if tau < 0 or tau >= T:
                raise ValueError(f"tau={tau} out of range for T={T}")
            lag_len = T - tau
            per_win = np.empty((len(windows), L, lag_len), dtype=np.float32)

            for i, w in enumerate(windows):
                X = w["X"]
                Xp = X.copy()
                Xp[tau, feat_idx] += eps
                H0 = H0_list[i]
                Hp = extract_ssm_traj(model, Xp, device)

                for l in range(L):
                    base = H0[l, tau:, :]
                    pert = Hp[l, tau:, :]
                    dh = pert - base
                    num = np.linalg.norm(dh, axis=1)
                    den = np.linalg.norm(base, axis=1) + 1e-12
                    per_win[i, l, :] = (num / den) / (eps + 1e-12)

            curves_store[(alpha, tau)] = per_win
            mu = per_win.mean(axis=0)
            se = per_win.std(axis=0, ddof=0) / np.sqrt(per_win.shape[0])

            plot_lines(
                curves_mu=mu, curves_se=se,
                title=f"SSM relative impulse-response per-eps | split={ssmcfg.split_name} | feat={ssmcfg.feature_name} | alpha={alpha:g} eps={eps:.3g} | tau={tau}",
                xlabel="lag = t - tau",
                ylabel="E[ ||Δh||/||h|| ] / eps",
                outpath=os.path.join(outdir, "plots", f"state_ir_lines_alpha{alpha:g}_tau{tau}.png"),
                yscale_log=ssmcfg.yscale_log
            )

            for l in range(L):
                c = mu[l]
                state_ir_summary_rows.append({
                    "split": ssmcfg.split_name,
                    "feature": ssmcfg.feature_name,
                    "alpha": float(alpha),
                    "eps": float(eps),
                    "tau": int(tau),
                    "layer": int(l),
                    "lag0": float(c[0]),
                    "AUC": float(auc_trapz(c)),
                    "horizon_idx": horizon_index(c, ssmcfg.horizon_frac),
                    "tau_fit": fit_exp_time_constant(
                        c, min_lag=ssmcfg.fit_min_lag, max_lag=ssmcfg.fit_max_lag, floor_frac=ssmcfg.fit_floor_frac
                    ),
                    "T": int(T),
                    "Dflat": int(Dflat),
                    "N_windows": int(len(windows)),
                })

    state_ir_summary = pd.DataFrame(state_ir_summary_rows)
    state_ir_summary.to_csv(os.path.join(outdir, "tables", "state_ir_summary.csv"), index=False)

    for alpha in ssmcfg.alphas:
        tau_mus = []
        for tau in ssmcfg.taus:
            tau_mus.append(curves_store[(alpha, tau)].mean(axis=0))
        min_lag_len = min(m.shape[1] for m in tau_mus)
        avg_mu = np.mean([m[:, :min_lag_len] for m in tau_mus], axis=0)
        plot_heatmap(
            curves_mu_avg_tau=avg_mu,
            title=f"SSM impulse-response heatmap (avg over taus) | split={ssmcfg.split_name} | feat={ssmcfg.feature_name} | alpha={alpha:g} (per-eps)",
            outpath=os.path.join(outdir, "plots", f"state_ir_heatmap_alpha{alpha:g}.png"),
        )

    for alpha in ssmcfg.alphas:
        df_a = state_ir_summary[state_ir_summary["alpha"] == float(alpha)]
        tau_by_layer = []
        for l in range(L):
            vals = df_a[df_a["layer"] == l]["tau_fit"].to_numpy(np.float64)
            vals = vals[np.isfinite(vals)]
            tau_by_layer.append(np.nanmedian(vals) if vals.size else np.nan)
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(L), tau_by_layer, marker="o")
        plt.xticks(np.arange(L), [f"L{l}" for l in range(L)])
        plt.xlabel("layer")
        plt.ylabel("time constant (tau_fit)")
        plt.title(f"Estimated memory time constant vs layer | split={ssmcfg.split_name} | feat={ssmcfg.feature_name} | alpha={alpha:g}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plots", f"state_tau_vs_layer_alpha{alpha:g}.png"), dpi=160)
        plt.close()

    rng = np.random.default_rng(seed + 123)
    null_rows = []
    for alpha in ssmcfg.alphas:
        eps = float(alpha * feat_std)
        for tau in ssmcfg.taus:
            lag_len = T - tau
            per_win_sig = curves_store[(alpha, tau)]
            mu_sig = per_win_sig.mean(axis=0)

            per_win_null = np.zeros((len(windows), L, lag_len), dtype=np.float32)
            for i, w in enumerate(windows):
                X = w["X"]
                H0 = H0_list[i]
                acc = np.zeros((L, lag_len), dtype=np.float64)
                for _ in range(ssmcfg.null_noise_draws):
                    v = rng.standard_normal(d_in).astype(np.float32)
                    v /= (np.linalg.norm(v) + 1e-12)
                    Xn = X.copy()
                    Xn[tau, :] += eps * v
                    Hn = extract_ssm_traj(model, Xn, device)
                    for l in range(L):
                        base = H0[l, tau:, :]
                        pert = Hn[l, tau:, :]
                        dh = pert - base
                        num = np.linalg.norm(dh, axis=1)
                        den = np.linalg.norm(base, axis=1) + 1e-12
                        acc[l] += (num / den) / (eps + 1e-12)
                per_win_null[i] = (acc / float(ssmcfg.null_noise_draws)).astype(np.float32)

            mu_null = per_win_null.mean(axis=0)
            for l in range(L):
                A_sig = auc_trapz(mu_sig[l])
                A_nul = auc_trapz(mu_null[l])
                null_rows.append({
                    "split": ssmcfg.split_name,
                    "feature": ssmcfg.feature_name,
                    "alpha": float(alpha),
                    "eps": float(eps),
                    "tau": int(tau),
                    "layer": int(l),
                    "AUC_signal": float(A_sig),
                    "AUC_null": float(A_nul),
                    "AUC_ratio": float(A_sig / (A_nul + 1e-12)),
                    "null_noise_draws": int(ssmcfg.null_noise_draws),
                })

    null_tbl = pd.DataFrame(null_rows)
    null_tbl.to_csv(os.path.join(outdir, "tables", "state_ir_signal_vs_null.csv"), index=False)

    ratio_by_layer = []
    for l in range(L):
        vals = null_tbl[null_tbl["layer"] == l]["AUC_ratio"].to_numpy(np.float64)
        vals = vals[np.isfinite(vals)]
        ratio_by_layer.append(np.nanmedian(vals) if vals.size else np.nan)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(L), ratio_by_layer, marker="o")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(L), [f"L{l}" for l in range(L)])
    plt.xlabel("layer")
    plt.ylabel("AUC(signal) / AUC(null)")
    plt.title(f"Feature-specific sensitivity (signal vs random-dir null) | split={ssmcfg.split_name} | feat={ssmcfg.feature_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "signal_vs_null_auc_ratio.png"), dpi=160)
    plt.close()

    pred_rows = []
    for alpha in ssmcfg.alphas:
        eps = float(alpha * feat_std)
        for tau in ssmcfg.taus:
            lag_len = T - tau
            per_win = np.empty((len(windows), lag_len), dtype=np.float32)
            for i, w in enumerate(windows):
                X = w["X"]
                Xp = X.copy()
                Xp[tau, feat_idx] += eps
                y0 = Y0_list[i]
                yp = extract_pred_traj(model, Xp, device)
                dy = np.abs(yp[tau:] - y0[tau:]) / (eps + 1e-12)
                per_win[i] = dy.astype(np.float32)

            mu = per_win.mean(axis=0)
            se = per_win.std(axis=0, ddof=0) / np.sqrt(per_win.shape[0])

            lag = np.arange(lag_len)
            plt.figure(figsize=(9, 4))
            plt.plot(lag, mu, label="mean")
            plt.fill_between(lag, mu - se, mu + se, alpha=0.15)
            if ssmcfg.yscale_log:
                plt.yscale("log")
            plt.title(f"Prediction sensitivity per-eps | split={ssmcfg.split_name} | feat={ssmcfg.feature_name} | alpha={alpha:g} eps={eps:.3g} | tau={tau}")
            plt.xlabel("lag = t - tau")
            plt.ylabel("E[ |Δŷ| ] / eps")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "plots", f"pred_sens_lines_alpha{alpha:g}_tau{tau}.png"), dpi=160)
            plt.close()

            pred_rows.append({
                "split": ssmcfg.split_name,
                "feature": ssmcfg.feature_name,
                "alpha": float(alpha),
                "eps": float(eps),
                "tau": int(tau),
                "lag0": float(mu[0]),
                "AUC": float(auc_trapz(mu)),
                "horizon_idx": horizon_index(mu, ssmcfg.horizon_frac),
                "tau_fit": fit_exp_time_constant(mu, min_lag=ssmcfg.fit_min_lag, max_lag=ssmcfg.fit_max_lag, floor_frac=ssmcfg.fit_floor_frac),
                "N_windows": int(len(windows)),
            })

    pd.DataFrame(pred_rows).to_csv(os.path.join(outdir, "tables", "pred_sens_summary.csv"), index=False)

    with open(os.path.join(outdir, "cfg_used.json"), "w") as f:
        json.dump({
            "split_name": ssmcfg.split_name,
            "feature_name": ssmcfg.feature_name,
            "taus": list(ssmcfg.taus),
            "alphas": list(ssmcfg.alphas),
            "n_windows": ssmcfg.n_windows,
            "per_ticker_cap": ssmcfg.per_ticker_cap,
            "horizon_frac": ssmcfg.horizon_frac,
            "null_noise_draws": ssmcfg.null_noise_draws,
            "fit_min_lag": ssmcfg.fit_min_lag,
            "fit_max_lag": ssmcfg.fit_max_lag,
            "fit_floor_frac": ssmcfg.fit_floor_frac,
            "yscale_log": ssmcfg.yscale_log,
            "device": device, "T": int(T), "L": int(L), "d_in": int(d_in),
            "date_cuts": list(cuts),
        }, f, indent=2)


# ----------------------------
# Module (2): probe_real_signal.py (same logic; optimized by caching H per window)
# ----------------------------
def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu.astype(np.float64), sd.astype(np.float64)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    X1 = np.empty((n, d + 1), dtype=np.float64)
    X1[:, 0] = 1.0
    X1[:, 1:] = X
    XtX = X1.T @ X1
    Xty = X1.T @ y
    reg = np.zeros((d + 1, d + 1), dtype=np.float64)
    reg[1:, 1:] = lam * np.eye(d, dtype=np.float64)
    w1 = np.linalg.solve(XtX + reg, Xty)
    return w1[1:].copy(), float(w1[0])


def predict_lin(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    v = float(np.var(y_true))
    r2 = float(1.0 - mse / (v + 1e-12))
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = float(np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2)) + 1e-12)
    corr = float(np.sum(yt * yp) / denom)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "corr": corr}


def build_probe_xy_from_H(
    H_list: List[np.ndarray], windows: List[Dict],
    horizons: List[int], warmup: int, max_samples: int, seed: int
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    L, T, _ = H_list[0].shape

    X_out: Dict[int, Dict[int, List[np.ndarray]]] = {k: {l: [] for l in range(L)} for k in horizons}
    y_out: Dict[int, List[float]] = {k: [] for k in horizons}

    for k in horizons:
        t_lo = int(max(0, warmup))
        t_hi = int(T - k)
        if t_hi - t_lo <= 0:
            raise ValueError(f"Horizon k={k} too large for T={T} with warmup={warmup}.")

        candidates = []
        for i in range(len(windows)):
            for t in range(t_lo, t_hi):
                candidates.append((i, t))

        if len(candidates) > max_samples:
            idx = rng.choice(len(candidates), size=max_samples, replace=False)
            candidates = [candidates[j] for j in idx]

        for (i, t) in candidates:
            y_out[k].append(float(windows[i]["y"][t + k]))
            H = H_list[i]
            for l in range(L):
                X_out[k][l].append(H[l, t].copy())

    X_final: Dict[int, Dict[int, np.ndarray]] = {}
    y_final: Dict[int, np.ndarray] = {}
    for k in horizons:
        y_final[k] = np.asarray(y_out[k], dtype=np.float32)
        X_final[k] = {}
        for l in range(L):
            X_final[k][l] = np.stack(X_out[k][l], axis=0).astype(np.float32)
    return X_final, y_final


def run_probe(outdir: str, model: MambaTS, device: str,
              feats: List[str], target_col: str, target_src_col: str,
              tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
              cuts: Tuple[str, str], T: int, seed: int,
              n_windows_train: int, n_windows_val: int, n_windows_test: int,
              per_ticker_cap: Optional[int], warmup: int,
              horizons: List[int], lambdas: List[float], max_samples_per_split: int):

    win_tr = sample_windows_xy(tr, feats, target_col, T, n_windows_train, seed + 1, per_ticker_cap)
    win_va = sample_windows_xy(va, feats, target_col, T, n_windows_val,   seed + 2, per_ticker_cap)
    win_te = sample_windows_xy(te, feats, target_col, T, n_windows_test,  seed + 3, per_ticker_cap)

    Htr = [extract_ssm_traj(model, w["X"], device) for w in win_tr]
    Hva = [extract_ssm_traj(model, w["X"], device) for w in win_va]
    Hte = [extract_ssm_traj(model, w["X"], device) for w in win_te]

    Xtr, ytr = build_probe_xy_from_H(Htr, win_tr, horizons, warmup, max_samples_per_split, seed + 10)
    Xva, yva = build_probe_xy_from_H(Hva, win_va, horizons, warmup, max_samples_per_split, seed + 20)
    Xte, yte = build_probe_xy_from_H(Hte, win_te, horizons, warmup, max_samples_per_split, seed + 30)

    L = Htr[0].shape[0]
    rows = []
    r2_mat = np.full((L, len(horizons)), np.nan, dtype=np.float64)
    corr_mat = np.full((L, len(horizons)), np.nan, dtype=np.float64)

    for hi, k in enumerate(horizons):
        y_tr = ytr[k].astype(np.float64)
        y_va = yva[k].astype(np.float64)
        y_te = yte[k].astype(np.float64)

        for l in range(L):
            X_tr = Xtr[k][l].astype(np.float64)
            X_va = Xva[k][l].astype(np.float64)
            X_te = Xte[k][l].astype(np.float64)

            mu, sd = standardize_fit(X_tr)
            X_trs = standardize_apply(X_tr, mu, sd)
            X_vas = standardize_apply(X_va, mu, sd)
            X_tes = standardize_apply(X_te, mu, sd)

            best_lam = None
            best_val_r2 = -1e18
            best_w = None
            best_b = 0.0

            for lam in lambdas:
                w, b = ridge_fit_closed_form(X_trs, y_tr, lam)
                pred_va = predict_lin(X_vas, w, b)
                r2 = metrics(y_va, pred_va)["r2"]
                if r2 > best_val_r2:
                    best_val_r2 = r2
                    best_lam = float(lam)
                    best_w = w
                    best_b = b

            pred_tr = predict_lin(X_trs, best_w, best_b)
            pred_va = predict_lin(X_vas, best_w, best_b)
            pred_te = predict_lin(X_tes, best_w, best_b)

            mtr = metrics(y_tr, pred_tr)
            mva = metrics(y_va, pred_va)
            mte = metrics(y_te, pred_te)

            rows.append({
                "horizon_k": int(k),
                "layer": int(l),
                "lambda": float(best_lam),
                "n_train": int(X_tr.shape[0]),
                "n_val": int(X_va.shape[0]),
                "n_test": int(X_te.shape[0]),
                "train_r2": mtr["r2"], "train_corr": mtr["corr"], "train_rmse": mtr["rmse"], "train_mae": mtr["mae"],
                "val_r2":   mva["r2"], "val_corr":   mva["corr"], "val_rmse":   mva["rmse"], "val_mae":   mva["mae"],
                "test_r2":  mte["r2"], "test_corr":  mte["corr"], "test_rmse":  mte["rmse"], "test_mae":  mte["mae"],
            })

            r2_mat[l, hi] = mte["r2"]
            corr_mat[l, hi] = mte["corr"]

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "tables", "probe_real_signal.csv"), index=False)

    xlab = [str(k) for k in horizons]
    ylab = [f"L{l}" for l in range(L)]
    heatmap_plot(r2_mat, xlab, ylab,
                 "Linear probe: TEST R2 for y_{t+k} from h_t (SSM state)",
                 os.path.join(outdir, "plots", "probe_r2_heatmap.png"))
    heatmap_plot(corr_mat, xlab, ylab,
                 "Linear probe: TEST Pearson corr for y_{t+k} from h_t (SSM state)",
                 os.path.join(outdir, "plots", "probe_corr_heatmap.png"))

    mean_r2 = np.nanmean(r2_mat, axis=1)
    best_layer = int(np.nanargmax(mean_r2))
    best_r2 = r2_mat[best_layer]
    best_corr = corr_mat[best_layer]

    plt.figure(figsize=(7.5, 4))
    plt.plot(horizons, best_r2, marker="o")
    plt.xlabel("horizon k")
    plt.ylabel("TEST R2")
    plt.title(f"Best layer by mean R2: L{best_layer} | R2 vs horizon")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "probe_best_layer_r2_vs_horizon.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7.5, 4))
    plt.plot(horizons, best_corr, marker="o")
    plt.xlabel("horizon k")
    plt.ylabel("TEST corr")
    plt.title(f"Best layer by mean R2: L{best_layer} | corr vs horizon")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "probe_best_layer_corr_vs_horizon.png"), dpi=160)
    plt.close()

    with open(os.path.join(outdir, "cfg_probe_used.json"), "w") as f:
        json.dump({
            "T": int(T), "L": int(L),
            "n_windows_train": int(n_windows_train),
            "n_windows_val": int(n_windows_val),
            "n_windows_test": int(n_windows_test),
            "per_ticker_cap": per_ticker_cap,
            "warmup": int(warmup),
            "horizons": list(horizons),
            "lambdas": list(lambdas),
            "max_samples_per_split": int(max_samples_per_split),
            "seed": int(seed),
            "device": device,
            "date_cut1": cuts[0],
            "date_cut2": cuts[1],
        }, f, indent=2)


# ----------------------------
# Module (3): init-context test (same logic)
# ----------------------------
@torch.no_grad()
def run_ctxK(model: MambaTS, device: str, tick_df: pd.DataFrame, feats: List[str], ycol: str, start: int, K: int, T: int):
    X = tick_df[feats].to_numpy(np.float32)
    y = tick_df[ycol].to_numpy(np.float32)
    if start + T > len(X):
        raise ValueError("start out of range")

    ctx = X[max(0, start - K):start] if K > 0 else X[start:start]
    Xw = X[start:start + T]
    xcat = np.concatenate([ctx, Xw], axis=0).astype(np.float32)

    xt = torch.from_numpy(xcat).unsqueeze(0).to(device)
    yhat, aux = model(xt, return_hidden_states=False, return_ssm_states=True)
    yhat = yhat.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32)

    ssm = aux["ssm_states"]
    tot = len(ssm)
    L = len(ssm[0])
    Dflat = int(np.prod(ssm[0][0].squeeze(0).shape))

    H = np.empty((L, tot, Dflat), dtype=np.float32)
    for t in range(tot):
        for l in range(L):
            H[l, t] = ssm[t][l].squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)

    return yhat[-T:], H[:, -T:, :], y[start:start + T].astype(np.float32)


def run_init_context_test(outdir: str, model: MambaTS, device: str,
                          test_df: pd.DataFrame, feats: List[str], ycol: str,
                          KS: List[int], seed: int, T: int, init_max_windows: Optional[int]):

    TEST = test_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    groups = {t: g.sort_values("date").reset_index(drop=True) for t, g in TEST.groupby("ticker", sort=False)}
    tickers = sorted(groups.keys())

    starts_all = []
    for t in tickers:
        n = len(groups[t])
        if n >= T:
            for s in range(0, n - T + 1):
                starts_all.append((t, s))

    if init_max_windows is not None and len(starts_all) > init_max_windows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(starts_all), size=init_max_windows, replace=False)
        starts = [starts_all[i] for i in idx]
    else:
        starts = starts_all

    N = len(starts)
    L = int(model.n_layers) if hasattr(model, "n_layers") else None

    mse_by_pos = {K: np.empty((N, T), dtype=np.float32) for K in KS}
    rel_state = {K: None for K in KS if K != 0}

    # infer L from first run (K=0)
    y0, H0, ytrue = run_ctxK(model, device, groups[starts[0][0]], feats, ycol, starts[0][1], 0, T)
    Lloc = H0.shape[0]
    for K in KS:
        if K != 0:
            rel_state[K] = np.empty((N, Lloc, T), dtype=np.float32)

    for i, (tick, s) in enumerate(starts):
        g = groups[tick]
        y0, H0, ytrue = run_ctxK(model, device, g, feats, ycol, s, 0, T)
        mse_by_pos[0][i] = (y0 - ytrue) ** 2
        base_norm = np.linalg.norm(H0, axis=-1) + 1e-12  # (L,T)

        for K in KS:
            if K == 0:
                continue
            yk, Hk, _ = run_ctxK(model, device, g, feats, ycol, s, K, T)
            mse_by_pos[K][i] = (yk - ytrue) ** 2
            rel_state[K][i] = np.linalg.norm(Hk - H0, axis=-1) / base_norm

        if (i + 1) % 50 == 0:
            print(f"[initctx] {i+1}/{N} windows done")

    rows_mse = []
    for K in KS:
        E = mse_by_pos[K]
        mu = E.mean(axis=0)
        se = E.std(axis=0, ddof=0) / np.sqrt(E.shape[0])
        for t in range(T):
            rows_mse.append({"K": K, "pos": t, "mse_mu": float(mu[t]), "mse_se": float(se[t]), "N": int(E.shape[0])})
    pd.DataFrame(rows_mse).to_csv(os.path.join(outdir, "tables", "mse_by_pos.csv"), index=False)

    rows_rel = []
    for K in KS:
        if K == 0:
            continue
        D = rel_state[K]
        mu = D.mean(axis=0)
        se = D.std(axis=0, ddof=0) / np.sqrt(D.shape[0])
        for l in range(mu.shape[0]):
            for t in range(T):
                rows_rel.append({
                    "K": K, "layer": l, "pos": t,
                    "rel_mu": float(mu[l, t]),
                    "rel_se": float(se[l, t]),
                    "N": int(D.shape[0]),
                })
    pd.DataFrame(rows_rel).to_csv(os.path.join(outdir, "tables", "rel_state_by_pos_layer.csv"), index=False)

    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump({"T": int(T), "KS": KS, "N_windows": int(N)}, f, indent=2)

    x = np.arange(T)

    plt.figure(figsize=(9, 4))
    for K in KS:
        med = np.median(mse_by_pos[K], axis=0)
        plt.plot(x, med, label=f"K={K}", linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("timestep in window")
    plt.ylabel("MSE")
    plt.title("TEST: MSE by timestep under different initial-context lengths")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "mse_by_pos.png"), dpi=160)
    plt.close()

    for K in KS:
        if K == 0:
            continue
        D = rel_state[K]
        mu = D.mean(axis=0)
        se = D.std(axis=0, ddof=0) / np.sqrt(N)
        plt.figure(figsize=(9, 4))
        for l in range(mu.shape[0]):
            plt.plot(x, mu[l], label=f"layer{l}")
            plt.fill_between(x, mu[l] - se[l], mu[l] + se[l], alpha=0.10)
        plt.yscale("log")
        plt.xlabel("timestep in window")
        plt.ylabel("E[ ||h_K-h_0|| / ||h_0|| ]")
        plt.title(f"TEST: Relative SSM deviation vs zero-init | K={K}")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plots", f"rel_state_layers_K{K}.png"), dpi=160)
        plt.close()

    plt.figure(figsize=(9, 4))
    for K in KS:
        if K == 0:
            continue
        D = rel_state[K]
        per_win = D.mean(axis=1)
        mu = per_win.mean(axis=0)
        se = per_win.std(axis=0, ddof=0) / np.sqrt(N)
        plt.plot(x, mu, label=f"K={K}")
        plt.fill_between(x, mu - se, mu + se, alpha=0.12)
    plt.yscale("log")
    plt.xlabel("timestep in window")
    plt.ylabel("E_layer[ ||h_K-h_0|| / ||h_0|| ]")
    plt.title("TEST: Layer-avg relative SSM deviation vs zero-init (all modes)")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plots", "rel_state_layeravg_allmodes.png"), dpi=160)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--feats", nargs="+", default=["log_range", "log_ret_close", "log_ret_volume"])
    ap.add_argument("--target_col", default="target_log_range_next")
    ap.add_argument("--target_src_col", default="log_range")

    ap.add_argument("--split_train", type=float, default=0.7)
    ap.add_argument("--split_val", type=float, default=0.15)
    ap.add_argument("--split_test", type=float, default=0.15)

    # (1) SSM impulse/sensitivity analysis params (same defaults)
    ap.add_argument("--split_name", choices=["train", "val", "test"], default="test")
    ap.add_argument("--feature_name", default="log_range")
    ap.add_argument("--taus", nargs="+", type=int, default=[0, 16, 32, 64])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--n_windows", type=int, default=60)
    ap.add_argument("--per_ticker_cap", type=int, default=0)
    ap.add_argument("--horizon_frac", type=float, default=0.1)
    ap.add_argument("--null_noise_draws", type=int, default=1)
    ap.add_argument("--fit_min_lag", type=int, default=0)
    ap.add_argument("--fit_max_lag", type=int, default=0)
    ap.add_argument("--fit_floor_frac", type=float, default=1e-3)
    ap.add_argument("--yscale_log", action="store_true")

    # (2) Probe params
    ap.add_argument("--do_probe", action="store_true")
    ap.add_argument("--n_windows_train", type=int, default=200)
    ap.add_argument("--n_windows_val", type=int, default=120)
    ap.add_argument("--n_windows_test", type=int, default=120)
    ap.add_argument("--probe_warmup", type=int, default=16)
    ap.add_argument("--probe_horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--probe_lambdas", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    ap.add_argument("--probe_max_samples_per_split", type=int, default=200000)

    # (3) Init-context params
    ap.add_argument("--do_initctx", action="store_true")
    ap.add_argument("--init_KS", nargs="+", type=int, default=[0, 4, 16, 32, 64])
    ap.add_argument("--init_max_windows", type=int, default=100)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    ensure_outdirs(args.outdir)
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, train_cfg = load_model_from_ckpt(args.ckpt, device)
    T = int(train_cfg.get("seq_len", 128))
    d_in = int(train_cfg.get("d_in", len(args.feats)))

    df = pd.read_csv(args.csv)
    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError("CSV must contain ticker,date.")
    for c in args.feats:
        if c not in df.columns:
            raise ValueError(f"Missing feature col: {c}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = add_next_target(df, "ticker", args.target_src_col, args.target_col)
    df = df.dropna(subset=list(args.feats) + [args.target_col]).reset_index(drop=True)

    tr, va, te, cut1, cut2 = split_by_date_3way(df, args.split_train, args.split_val, args.split_test)
    cuts = (str(cut1), str(cut2))

    per_cap = None if args.per_ticker_cap <= 0 else int(args.per_ticker_cap)
    fit_max = None if args.fit_max_lag <= 0 else int(args.fit_max_lag)

    ssmcfg = SSMCfg(
        split_name=args.split_name,
        feature_name=args.feature_name,
        taus=tuple(int(x) for x in args.taus),
        alphas=tuple(float(x) for x in args.alphas),
        n_windows=int(args.n_windows),
        per_ticker_cap=per_cap,
        horizon_frac=float(args.horizon_frac),
        null_noise_draws=int(args.null_noise_draws),
        fit_min_lag=int(args.fit_min_lag),
        fit_max_lag=fit_max,
        fit_floor_frac=float(args.fit_floor_frac),
        yscale_log=bool(args.yscale_log),
    )

    run_ssm_analysis(
        outdir=args.outdir, model=model, device=device,
        feats=list(args.feats), target_col=args.target_col, target_src_col=args.target_src_col,
        tr=tr, va=va, te=te, cuts=cuts, T=T, d_in=d_in, ssmcfg=ssmcfg, seed=args.seed
    )

    if args.do_probe:
        run_probe(
            outdir=args.outdir, model=model, device=device,
            feats=list(args.feats), target_col=args.target_col, target_src_col=args.target_src_col,
            tr=tr, va=va, te=te, cuts=cuts, T=T, seed=args.seed,
            n_windows_train=int(args.n_windows_train),
            n_windows_val=int(args.n_windows_val),
            n_windows_test=int(args.n_windows_test),
            per_ticker_cap=per_cap,
            warmup=int(args.probe_warmup),
            horizons=[int(k) for k in args.probe_horizons],
            lambdas=[float(x) for x in args.probe_lambdas],
            max_samples_per_split=int(args.probe_max_samples_per_split),
        )

    if args.do_initctx:
        init_max = None if args.init_max_windows <= 0 else int(args.init_max_windows)
        run_init_context_test(
            outdir=args.outdir, model=model, device=device,
            test_df=te, feats=list(args.feats), ycol=args.target_col,
            KS=[int(k) for k in args.init_KS],
            seed=args.seed, T=T, init_max_windows=init_max
        )

    print("Done. Saved to:", args.outdir)


if __name__ == "__main__":
    main()