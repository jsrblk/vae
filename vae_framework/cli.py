from __future__ import annotations

import argparse
import os

import pandas as pd

from .data import get_sp500_tickers, fetch_prices, compute_daily_returns_from_csv_dir
from .preprocess import filter_panel, compute_standardization, apply_standardization, save_panel
from .train import TrainConfig, train, load_and_sample
from .utils import ensure_dir


def cmd_fetch(args: argparse.Namespace) -> None:
	tickers = get_sp500_tickers(limit=args.limit_tickers)
	ensure_dir(args.save_dir)
	fetch_prices(tickers, start=args.start, end=args.end, save_dir=args.save_dir, interval="1d", auto_adjust=False)
	print(f"Saved raw CSVs to {args.save_dir}")


def cmd_preprocess(args: argparse.Namespace) -> None:
	panel = compute_daily_returns_from_csv_dir(args.raw_dir, start=args.start, end=args.end)
	panel = filter_panel(panel, min_days=args.min_days, max_missing_frac=args.max_missing_frac, start=args.start, end=args.end)
	if args.standardize:
		stats = compute_standardization(panel)
		panel_std = apply_standardization(panel, stats)
		save_panel(panel_std, args.out_file, stats)
	else:
		save_panel(panel.fillna(0.0), args.out_file, None)
	print(f"Saved processed panel to {args.out_file}")


def cmd_train(args: argparse.Namespace) -> None:
	ensure_dir(args.checkpoint_dir)
	cfg = TrainConfig(
		data_file=args.data_file,
		checkpoint_dir=args.checkpoint_dir,
		latent_dim=args.latent_dim,
		hidden_dims=args.hidden_dims,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		val_frac=args.val_frac,
	)
	best = train(cfg)
	print(f"Best checkpoint: {best}")


def cmd_sample(args: argparse.Namespace) -> None:
	from .train import load_and_sample
	samples = load_and_sample(args.checkpoint, args.num_samples)
	ensure_dir(os.path.dirname(args.out_file) or ".")
	pd.DataFrame(samples.cpu().numpy()).to_csv(args.out_file, index=False)
	print(f"Saved samples to {args.out_file}")


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="VAE framework for S&P500 returns")
	sub = p.add_subparsers(dest="cmd", required=True)

	pf = sub.add_parser("fetch", help="Fetch raw prices")
	pf.add_argument("--start", type=str, required=True)
	pf.add_argument("--end", type=str, required=True)
	pf.add_argument("--save-dir", type=str, default="artifacts/data/raw")
	pf.add_argument("--limit-tickers", type=int, default=None)
	pf.set_defaults(func=cmd_fetch)

	pp = sub.add_parser("preprocess", help="Build returns panel")
	pp.add_argument("--raw-dir", type=str, default="artifacts/data/raw")
	pp.add_argument("--out-file", type=str, default="artifacts/data/returns.parquet")
	pp.add_argument("--start", type=str, default=None)
	pp.add_argument("--end", type=str, default=None)
	pp.add_argument("--min-days", type=int, default=252)
	pp.add_argument("--max-missing-frac", type=float, default=0.1)
	pp.add_argument("--standardize", action="store_true")
	pp.set_defaults(func=cmd_preprocess)

	pt = sub.add_parser("train", help="Train VAE")
	pt.add_argument("--data-file", type=str, default="artifacts/data/returns.parquet")
	pt.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints")
	pt.add_argument("--latent-dim", type=int, default=16)
	pt.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
	pt.add_argument("--epochs", type=int, default=50)
	pt.add_argument("--batch-size", type=int, default=128)
	pt.add_argument("--lr", type=float, default=1e-3)
	pt.add_argument("--val-frac", type=float, default=0.1)
	pt.set_defaults(func=cmd_train)

	ps = sub.add_parser("sample", help="Sample synthetic returns")
	ps.add_argument("--checkpoint", type=str, required=True)
	ps.add_argument("--num-samples", type=int, default=10)
	ps.add_argument("--out-file", type=str, default="artifacts/samples/returns.csv")
	ps.set_defaults(func=cmd_sample)

	return p


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()


