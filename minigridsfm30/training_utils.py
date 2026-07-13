from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        requested = "cuda:0"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{requested}', but CUDA is not available."
            )

        device = torch.device(requested)
        index = 0 if device.index is None else device.index

        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device index {index}, but "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )

        return torch.device(f"cuda:{index}")

    raise ValueError(
        f"Unsupported --device value: {requested!r}. "
        "Use auto, cpu, cuda, or cuda:N."
    )


def is_finite_number(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def assert_finite_tensor(name: str, value: Any) -> None:
    if torch.is_tensor(value) and not torch.isfinite(value).all():
        bad = int((~torch.isfinite(value)).sum().item())
        raise FloatingPointError(
            f"Non-finite tensor detected: {name}; bad_values={bad}; "
            f"shape={tuple(value.shape)}"
        )


def validate_batch_finite(batch) -> None:
    for node_type in batch.node_types:
        store = batch[node_type]
        for attr in ("x", "y"):
            if hasattr(store, attr):
                assert_finite_tensor(f"{node_type}.{attr}", getattr(store, attr))

    for attr in ("res_cost", "feasible"):
        if hasattr(batch, attr):
            assert_finite_tensor(attr, getattr(batch, attr))


def validate_output_finite(output: dict[str, Any]) -> None:
    for key, value in output.items():
        if key == "x_dict" and isinstance(value, dict):
            for node_type, tensor in value.items():
                assert_finite_tensor(f"output.x_dict.{node_type}", tensor)
        else:
            assert_finite_tensor(f"output.{key}", value)


def validate_metrics_finite(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if not is_finite_number(value):
            raise FloatingPointError(
                f"Non-finite metric detected: {key}={value}"
            )


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
