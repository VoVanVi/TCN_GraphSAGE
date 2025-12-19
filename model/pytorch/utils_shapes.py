import torch


def ensure_shape(tensor: torch.Tensor, expected: tuple, name: str = "tensor") -> None:
    """Lightweight runtime check used for debugging shapes."""
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match expected {expected}")