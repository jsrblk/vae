import os
import json
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
	ensure_dir(os.path.dirname(path) or ".")
	with open(path, "w") as f:
		json.dump(obj, f, indent=2)


def set_seed(seed: Optional[int]) -> None:
	if seed is None:
		return
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


