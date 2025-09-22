# VAE for Joint Distribution of S&P 500 Daily Returns (Pyro/PyTorch)

This framework fetches historical S&P 500 prices, computes daily returns, and trains a VAE (PyTorch + Pyro) to model the joint distribution of daily returns.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# Fetch S&P 500 data from Yahoo Finance
python -m vae_framework.cli fetch --start 2005-01-01 --end 2024-12-31 --save-dir artifacts/data/raw --limit-tickers 500

# Preprocess into aligned daily returns panel
python -m vae_framework.cli preprocess --raw-dir artifacts/data/raw --out-file artifacts/data/returns.parquet --start 2005-01-01 --end 2024-12-31 --min-days 1000 --max-missing-frac 0.05

# Train VAE on day-level return vectors
python -m vae_framework.cli train --data-file artifacts/data/returns.parquet --epochs 50 --latent-dim 16 --hidden-dims 512 256 --batch-size 128 --lr 1e-3 --val-frac 0.1 --standardize

# Sample synthetic return vectors
python -m vae_framework.cli sample --checkpoint artifacts/checkpoints/best.pt --num-samples 10 --out-file artifacts/samples/returns.csv
```

Artifacts are stored under `artifacts/` by default.

## Notes
- Uses `yfinance.tickers_sp500()` to get tickers, and `Adj Close` for returns.
- Not all years need be included—control with `--start`/`--end` and `--min-days`.
- Decoder is a diagonal Gaussian over returns; guide is amortized Gaussian.

License: MIT


