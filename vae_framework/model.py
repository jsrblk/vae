from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class Encoder(nn.Module):
	def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
		super().__init__()
		layers: List[nn.Module] = []
		prev = input_dim
		for h in hidden_dims:
			layers += [nn.Linear(prev, h), nn.ReLU()]
			prev = h
		self.net = nn.Sequential(*layers)
		self.mu = nn.Linear(prev, latent_dim)
		self.logvar = nn.Linear(prev, latent_dim)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.net(x)
		mu = self.mu(h)
		logvar = self.logvar(h)
		return mu, logvar


class Decoder(nn.Module):
	def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
		super().__init__()
		layers: List[nn.Module] = []
		prev = latent_dim
		for h in hidden_dims:
			layers += [nn.Linear(prev, h), nn.ReLU()]
			prev = h
		self.net = nn.Sequential(*layers)
		self.mu = nn.Linear(prev, output_dim)
		self.logvar = nn.Linear(prev, output_dim)

	def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.net(z)
		mu = self.mu(h)
		logvar = self.logvar(h)
		return mu, logvar


class VAE(PyroModule):
	def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
		super().__init__()
		self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
		self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
		self.latent_dim = latent_dim
		self.input_dim = input_dim

	def model(self, x: torch.Tensor):
		pyro.module("decoder", self.decoder)
		with pyro.plate("batch", x.size(0)):
			z = pyro.sample("z", dist.Normal(torch.zeros(x.size(0), self.latent_dim, device=x.device), torch.ones(x.size(0), self.latent_dim, device=x.device)).to_event(1))
			mu_x, logvar_x = self.decoder(z)
			sigma_x = (0.5 * logvar_x).exp()
			pyro.sample("x", dist.Normal(mu_x, sigma_x).to_event(1), obs=x)

	def guide(self, x: torch.Tensor):
		pyro.module("encoder", self.encoder)
		with pyro.plate("batch", x.size(0)):
			mu_z, logvar_z = self.encoder(x)
			sigma_z = (0.5 * logvar_z).exp()
			pyro.sample("z", dist.Normal(mu_z, sigma_z).to_event(1))

	@torch.no_grad()
	def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
		z = torch.randn(num_samples, self.latent_dim, device=device)
		mu_x, logvar_x = self.decoder(z)
		return mu_x


