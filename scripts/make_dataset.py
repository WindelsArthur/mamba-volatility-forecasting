#make_dataset.py

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

DEFAULT_REPO_ID = "guloyy/sp500_csv"

FEATURE_COLS = ["log_range", "log_ret_close", "log_ret_volume"]
REQUIRED_COLS = ["date", "high", "low", "close", "volume"]


@dataclass
class TickerStats:
    ticker: str
    status: str  # kept / dropped
    reason: str
    raw_rows: int
    feat_rows: int
    canonical_rows: int
    matched_rows: int
    coverage: float
    first_date: Optional[str]
    last_date: Optional[str]


def setup_logger(log_path: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("dataset_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def list_tickers_in_repo(repo_id: str, logger: logging.Logger) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    tickers = sorted(
        f[:-4] for f in files
        if f.lower().endswith(".csv") and "/" not in f  # only top-level CSVs
    )
    logger.info("Discovered %d tickers (csv files) in repo %s", len(tickers), repo_id)
    return tickers


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # basic validation
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # ensure numeric; enforce positivity for logs
    must_be_pos = ["high", "low", "close", "volume"]
    for c in must_be_pos:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= 0, c] = np.nan

    log_high = np.log(df["high"])
    log_low = np.log(df["low"])
    log_close = np.log(df["close"])
    log_vol = np.log(df["volume"])

    df["log_range"] = log_high - log_low
    df["log_ret_close"] = log_close - log_close.shift(1)
    df["log_ret_volume"] = log_vol - log_vol.shift(1)

    out = df[["date"] + FEATURE_COLS].dropna().reset_index(drop=True)

    # final sanity: ensure finite
    arr = out[FEATURE_COLS].to_numpy(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values present after feature engineering.")

    return out


def load_ticker(repo_id: str, ticker: str) -> pd.DataFrame:
    path = hf_hub_download(repo_id=repo_id, filename=f"{ticker}.csv", repo_type="dataset")
    raw = pd.read_csv(path)
    feat = make_features(raw)
    feat.insert(0, "ticker", ticker)
    return feat, len(raw)


def align_to_canonical(
    feat: pd.DataFrame, canonical_dates: pd.Index
) -> Tuple[pd.DataFrame, int, float]:
    # Inner join to canonical grid: only keep dates that exist in canonical
    aligned = feat.merge(
        pd.DataFrame({"date": canonical_dates.astype(str)}),
        on="date",
        how="inner",
        validate="many_to_one",
    )
    matched = len(aligned)
    coverage = matched / max(len(canonical_dates), 1)
    return aligned, matched, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    ap.add_argument("--out_path", default="dataset_3feat_alltickers.csv")
    ap.add_argument("--out_stats_path", default="dataset_3feat_alltickers_stats.csv")
    ap.add_argument("--ref_ticker", default="AAPL", help="Defines canonical date grid after feature engineering.")
    ap.add_argument(
        "--min_coverage",
        type=float,
        default=0.98,
        help="Minimum fraction of canonical dates a ticker must match to be kept (0-1).",
    )
    ap.add_argument("--log_path", default="dataset_3feat_build.log")
    args = ap.parse_args()

    logger = setup_logger(args.log_path)
    logger.info("Repo: %s", args.repo_id)
    logger.info("Output dataset: %s", args.out_path)
    logger.info("Output stats: %s", args.out_stats_path)
    logger.info("Reference ticker: %s", args.ref_ticker)
    logger.info("Min coverage: %.4f", args.min_coverage)

    tickers = list_tickers_in_repo(args.repo_id, logger)
    if args.ref_ticker not in tickers:
        raise RuntimeError(f"REF_TICKER={args.ref_ticker} not found in repo.")

    # Build canonical grid from ref ticker AFTER FE
    logger.info("Building canonical date grid from %s ...", args.ref_ticker)
    ref_feat, ref_raw_rows = load_ticker(args.repo_id, args.ref_ticker)
    canonical_dates = pd.Index(ref_feat["date"].astype(str).tolist())
    logger.info(
        "Canonical grid size: %d dates (ref raw rows=%d, ref feat rows=%d)",
        len(canonical_dates), ref_raw_rows, len(ref_feat),
    )

    kept_frames: List[pd.DataFrame] = []
    stats_rows: List[TickerStats] = []
    reasons = Counter()

    total = len(tickers)
    processed = 0

    for i, t in enumerate(tickers, start=1):
        processed += 1
        if i % 25 == 0 or i == total:
            logger.info("Progress: %d/%d", i, total)

        try:
            feat, raw_rows = load_ticker(args.repo_id, t)
            feat_rows = len(feat)

            aligned, matched, coverage = align_to_canonical(feat, canonical_dates)

            first_date = aligned["date"].iloc[0] if len(aligned) else None
            last_date = aligned["date"].iloc[-1] if len(aligned) else None

            # Coverage filter
            if coverage < args.min_coverage:
                reason = f"coverage_below_threshold({coverage:.4f}<{args.min_coverage:.4f})"
                reasons[reason] += 1
                stats_rows.append(
                    TickerStats(
                        ticker=t,
                        status="dropped",
                        reason=reason,
                        raw_rows=raw_rows,
                        feat_rows=feat_rows,
                        canonical_rows=len(canonical_dates),
                        matched_rows=matched,
                        coverage=coverage,
                        first_date=first_date,
                        last_date=last_date,
                    )
                )
                logger.info("DROP %-6s | %s", t, reason)
                continue

            # Finite check (belt-and-suspenders)
            arr = aligned[FEATURE_COLS].to_numpy(np.float64)
            if not np.isfinite(arr).all():
                reason = "non_finite_after_alignment"
                reasons[reason] += 1
                stats_rows.append(
                    TickerStats(
                        ticker=t,
                        status="dropped",
                        reason=reason,
                        raw_rows=raw_rows,
                        feat_rows=feat_rows,
                        canonical_rows=len(canonical_dates),
                        matched_rows=matched,
                        coverage=coverage,
                        first_date=first_date,
                        last_date=last_date,
                    )
                )
                logger.info("DROP %-6s | %s", t, reason)
                continue

            kept_frames.append(aligned.sort_values(["ticker", "date"]).reset_index(drop=True))
            reasons["kept"] += 1
            stats_rows.append(
                TickerStats(
                    ticker=t,
                    status="kept",
                    reason="ok",
                    raw_rows=raw_rows,
                    feat_rows=feat_rows,
                    canonical_rows=len(canonical_dates),
                    matched_rows=matched,
                    coverage=coverage,
                    first_date=first_date,
                    last_date=last_date,
                )
            )
            logger.info("KEEP %-6s | coverage=%.4f matched=%d", t, coverage, matched)

        except Exception as e:
            reason = f"exception:{type(e).__name__}"
            reasons[reason] += 1
            stats_rows.append(
                TickerStats(
                    ticker=t,
                    status="dropped",
                    reason=str(e)[:300],  # keep readable
                    raw_rows=0,
                    feat_rows=0,
                    canonical_rows=len(canonical_dates),
                    matched_rows=0,
                    coverage=0.0,
                    first_date=None,
                    last_date=None,
                )
            )
            logger.info("DROP %-6s | %s | %s", t, reason, str(e))

    if not kept_frames:
        raise RuntimeError("No tickers were kept. Consider lowering --min_coverage or changing --ref_ticker.")

    out = pd.concat(kept_frames, axis=0, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    out.to_csv(args.out_path, index=False)

    stats_df = pd.DataFrame([asdict(s) for s in stats_rows]).sort_values(["status", "ticker"])
    stats_df.to_csv(args.out_stats_path, index=False)

    # Summary stats
    kept = int((stats_df["status"] == "kept").sum())
    dropped = int((stats_df["status"] == "dropped").sum())
    unique_kept = out["ticker"].nunique()

    cov_kept = stats_df.loc[stats_df["status"] == "kept", "coverage"]
    cov_drop = stats_df.loc[stats_df["status"] == "dropped", "coverage"]

    logger.info("========================================")
    logger.info("DONE")
    logger.info("Tickers total: %d | kept: %d | dropped: %d", total, kept, dropped)
    logger.info("Unique tickers in output: %d", unique_kept)
    logger.info("Output rows: %d", len(out))
    logger.info("Date range in output: %s .. %s", out["date"].min(), out["date"].max())

    if len(cov_kept):
        logger.info(
            "Coverage (kept): min=%.4f p25=%.4f median=%.4f p75=%.4f max=%.4f",
            float(cov_kept.min()),
            float(cov_kept.quantile(0.25)),
            float(cov_kept.median()),
            float(cov_kept.quantile(0.75)),
            float(cov_kept.max()),
        )
    if len(cov_drop):
        logger.info(
            "Coverage (dropped): min=%.4f median=%.4f max=%.4f",
            float(cov_drop.min()),
            float(cov_drop.median()),
            float(cov_drop.max()),
        )

    # Reasons breakdown (top)
    logger.info("Reason breakdown (top 15):")
    for k, v in reasons.most_common(15):
        logger.info("  %s: %d", k, v)

    # Show a quick preview to stdout (not only logs)
    print("\nSaved:", args.out_path)
    print("Saved stats:", args.out_stats_path)
    print("Kept tickers:", unique_kept)
    print("Rows:", len(out))
    print(out.head(10))


if __name__ == "__main__":
    main()