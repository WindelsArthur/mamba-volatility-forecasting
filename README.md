# Analyzing State-Space Dynamics in Mamba for Financial Time Series Forecasting

[Read the full paper here](paper/Mamba_Volatility_Forecasting.pdf)

This repository contains the code and research for adapting the Mamba architecture to next-day financial volatility (log-range) forecasting. Beyond evaluating predictive performance against traditional econometric models (HAR, EWMA), this project rigorously probes the internal selective state-space (SSM) representations of the Mamba model to understand how it processes noisy financial data. 

This project was developed for the Deep Learning course at ETH Zurich.

## Repository Structure

* `paper/`: Contains the final research paper detailing methodology, experiments, and state dynamics analysis.
* `scripts/`: The core execution pipeline (data generation, training, metric computation, and state analysis).
* `src/`: Core model architecture (`MambaTS`).
* `requirements.txt`: Strict dependency versioning for exact reproducibility.

## Environment Setup

Clone the repository and install the required dependencies. The requirements are strictly pinned to ensure reproducibility.

```bash
git clone [https://github.com/WindelsArthur/mamba-volatility-forecasting.git](https://github.com/WindelsArthur/mamba-volatility-forecasting.git)
cd mamba-volatility-forecasting
pip install -r requirements.txt
```

## Reproducing the Pipeline
The research pipeline is divided into four sequential scripts located in the scripts/ directory.

### 1. Data Preparation

Downloads S&P 500 daily OHLCV data, engineers log-features, and aligns them to a canonical trading calendar.

```bash
python scripts/make_dataset.py \
  --repo_id "guloyy/sp500_csv" \
  --out_path "data/dataset_3feat_alltickers.csv" \
  --ref_ticker "AAPL"
```

### 2. Model Training

Trains the MambaTS model using a causal, many-to-many teacher-forced regime.

```bash
python scripts/train.py \
  --dataset_csv "data/dataset_3feat_alltickers.csv" \
  --out_dir "checkpoints" \
  --seq_len 64 \
  --loss_mode "burnin" \
  --burn_in 16 \
  --epochs 4
```

### 3. Evaluation and Baselines

Fits the HAR baseline on the training split and evaluates Mamba, HAR, EWMA, and Last-Value heuristics on the out-of-sample test split.

```bash
python scripts/compute_metrics.py \
  --runs_root "checkpoints" \
  --batch_size 2048
```

### 4. State Dynamics Analysis

Extracts the internal SSM hidden trajectories under teacher-forcing to reproduce the representation analyses discussed in the paper (impulse responses, linear probing, and initialization context).

```bash
python scripts/analyze_states.py \
  --csv "data/dataset_3feat_alltickers.csv" \
  --ckpt "checkpoints/YOUR_RUN_NAME/model_best.pt" \
  --outdir "analysis_run" \
  --do_probe \
  --do_initctx \
  --yscale_log```
Generated plots and tables will be saved in analysis_run/plots/ and analysis_run/tables/.


## Authors
* Arthur Windels
* Sacha Liechti
