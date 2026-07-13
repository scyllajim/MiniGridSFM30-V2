import pytest
import torch

from minigridsfm30.training_utils import (
    assert_finite_tensor,
    resolve_device,
)


def test_resolve_cpu_device():
    assert str(resolve_device("cpu")) == "cpu"


def test_resolve_invalid_device():
    with pytest.raises(ValueError):
        resolve_device("not-a-device")


def test_finite_tensor_check_accepts_finite_values():
    assert_finite_tensor("x", torch.tensor([0.0, 1.0]))


def test_finite_tensor_check_rejects_nan():
    with pytest.raises(FloatingPointError):
        assert_finite_tensor("x", torch.tensor([float("nan")]))
