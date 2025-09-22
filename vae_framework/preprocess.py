from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PanelStats:
	mean: pd.Series
	std: pd.Series


def filter_panel(
	ret: pd.DataFrame,
	min_days: int = 252,
	max_missing_frac: float = 0.1,
	start: Optional[str] = None,
	end: Optional[str] = None,
) -> pd.DataFrame:
	if start:
		ret = ret[ret.index >= pd.to_datetime(start)]
	if end:
		ret = ret[ret.index <= pd.to_datetime(end)]
	# Drop columns with too many missing
	ok_mask = ret.isna().mean(axis=0) <= max_missing_frac
	ret = ret.loc[:, ok_mask]
	# Drop rows with too many missing (optional). Here we forward-fill then drop remaining NA rows.
	ret = ret.ffill().bfill()
	# Ensure enough length
	if len(ret) < min_days:
		# Return as-is; caller can decide
		return ret
	return ret


def compute_standardization(ret: pd.DataFrame) -> PanelStats:
	mean = ret.mean(axis=0)
	std = ret.std(axis=0).replace(0.0, 1.0)
	return PanelStats(mean=mean, std=std)


def apply_standardization(ret: pd.DataFrame, stats: PanelStats) -> pd.DataFrame:
	ret_std = (ret - stats.mean) / stats.std
	ret_std = ret_std.replace([np.inf, -np.inf], 0.0).fillna(0.0)
	return ret_std


def save_panel(ret: pd.DataFrame, out_file: str, stats: Optional[PanelStats] = None) -> None:
	meta = {}
	if stats is not None:
		meta = {"mean": stats.mean.to_dict(), "std": stats.std.to_dict()}
	ret.attrs["stats"] = meta
	ret.to_parquet(out_file, engine="pyarrow")


def load_panel(path: str) -> Tuple[pd.DataFrame, Optional[PanelStats]]:
	ret = pd.read_parquet(path, engine="pyarrow")
	meta = ret.attrs.get("stats", None)
	if isinstance(meta, dict) and "mean" in meta and "std" in meta:
		stats = PanelStats(mean=pd.Series(meta["mean"]), std=pd.Series(meta["std"]))
	else:
		stats = None
	return ret, stats


