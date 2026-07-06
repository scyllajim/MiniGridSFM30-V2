from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from minigridsfm30.graph_builder import sample_to_heterodata


class Case30OPFDataset(Dataset):
    def __init__(
        self,
        path: str,
        only_feasible: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.path = Path(path)
        self.mode = "processed" if self.path.suffix == ".pt" else "raw"

        if self.mode == "processed":
            self.raw = torch.load(self.path, map_location="cpu", weights_only=False)
            graphs = self.raw["graphs"]

            if max_samples is not None:
                graphs = graphs[:max_samples]

            self.graphs = graphs
            self.samples = None

        else:
            with open(self.path, "rb") as f:
                self.raw = pickle.load(f)

            samples = self.raw["samples"]

            if only_feasible:
                samples = [s for s in samples if s.get("feasible", False)]

            if max_samples is not None:
                samples = samples[:max_samples]

            self.samples = samples
            self.graphs = None

    def __len__(self):
        if self.mode == "processed":
            return len(self.graphs)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "processed":
            return self.graphs[idx]
        return sample_to_heterodata(self.samples[idx])

    @property
    def info(self):
        if self.mode == "processed":
            return {
                "path": str(self.path),
                "mode": "processed",
                "case": self.raw.get("case"),
                "n_used": len(self.graphs),
                "raw_info": self.raw.get("raw_info", {}),
            }

        return {
            "path": str(self.path),
            "mode": "raw",
            "case": self.raw.get("case"),
            "n_requested": self.raw.get("n_requested"),
            "n_success": self.raw.get("n_success"),
            "n_failed": self.raw.get("n_failed"),
            "n_saved": self.raw.get("n_saved"),
            "n_used": len(self.samples),
            "settings": self.raw.get("settings", {}),
        }
