from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from .model import VAE
from .preprocess import load_panel
from .utils import ensure_dir


class ReturnsDataset(Dataset):
	def __init__(self, panel: pd.DataFrame):
		# rows: dates, cols: tickers; treat each row as sample
		self.columns = list(panel.columns)
		self.tensor = torch.from_numpy(panel.astype("float32").values)

	def __len__(self) -> int:
		return self.tensor.size(0)

	def __getitem__(self, idx: int) -> torch.Tensor:
		return self.tensor[idx]


@dataclass
class TrainConfig:
	data_file: str
	checkpoint_dir: str
	latent_dim: int
	hidden_dims: List[int]
	epochs: int = 50
	batch_size: int = 128
	lr: float = 1e-3
	val_frac: float = 0.1
	seed: Optional[int] = 42
	device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(cfg: TrainConfig) -> str:
	panel, stats = load_panel(cfg.data_file)
	# Ensure no remaining NaNs
	panel = panel.fillna(0.0)
	dataset = ReturnsDataset(panel)
	val_len = int(len(dataset) * cfg.val_frac)
	train_len = len(dataset) - val_len
	train_ds, val_ds = random_split(dataset, [train_len, val_len]) if val_len > 0 else (dataset, None)
	train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds is not None else None

	input_dim = dataset.tensor.size(1)
	model = VAE(input_dim=input_dim, hidden_dims=cfg.hidden_dims, latent_dim=cfg.latent_dim)
	model.to(cfg.device)

	optimizer = ClippedAdam({"lr": cfg.lr})
	elbo = Trace_ELBO()
	svi = SVI(model.model, model.guide, optimizer, loss=elbo)

	best_val = float("inf")
	ensure_dir(cfg.checkpoint_dir)
	best_path = os.path.join(cfg.checkpoint_dir, "best.pt")

	for epoch in range(1, cfg.epochs + 1):
		model.train()
		total_loss = 0.0
		count = 0
		for batch in train_loader:
			batch = batch.to(cfg.device)
			loss = svi.step(batch)
			total_loss += loss
			count += batch.size(0)
		train_elbo = total_loss / max(1, count)

		val_elbo = None
		if val_loader is not None:
			model.eval()
			with torch.no_grad():
				vtot = 0.0
				vcount = 0
				for vb in val_loader:
					vb = vb.to(cfg.device)
					loss = svi.evaluate_loss(vb)
					vtot += loss
					vcount += vb.size(0)
				val_elbo = vtot / max(1, vcount)

		# Save best
		metric = val_elbo if val_elbo is not None else train_elbo
		if metric < best_val:
			best_val = metric
			torch.save({
				"state_dict": model.state_dict(),
				"input_dim": input_dim,
				"hidden_dims": cfg.hidden_dims,
				"latent_dim": cfg.latent_dim,
			}, best_path)

	return best_path


@torch.no_grad()

def load_and_sample(checkpoint_path: str, num_samples: int, device: Optional[str] = None) -> torch.Tensor:
	ckpt = torch.load(checkpoint_path, map_location="cpu")
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model = VAE(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"], latent_dim=ckpt["latent_dim"])
	model.load_state_dict(ckpt["state_dict"])
	model.to(device)
	return model.sample(num_samples, device=torch.device(device))


