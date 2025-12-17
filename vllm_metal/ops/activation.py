# SPDX-License-Identifier: Apache-2.0
"""Activation functions for Metal backend."""

import torch
from torch.nn import functional


def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """Fused SiLU activation and element-wise multiplication.

    Computes: out = silu(x[..., :d]) * x[..., d:]
    where d = x.shape[-1] // 2

    This is commonly used in LLaMA/Mistral FFN layers.

    Args:
        out: Output tensor [*, d]
        x: Input tensor [*, 2*d]
    """
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]

    # SiLU = x * sigmoid(x)
    out.copy_(functional.silu(gate) * up)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """Fused GELU activation and element-wise multiplication.

    Computes: out = gelu(x[..., :d]) * x[..., d:]
    where d = x.shape[-1] // 2

    Args:
        out: Output tensor [*, d]
        x: Input tensor [*, 2*d]
    """
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]

    out.copy_(functional.gelu(gate) * up)


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """Fused GELU (tanh approximation) activation and multiplication.

    Computes: out = gelu_tanh(x[..., :d]) * x[..., d:]
    where d = x.shape[-1] // 2

    Uses the tanh approximation of GELU:
    gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        out: Output tensor [*, d]
        x: Input tensor [*, 2*d]
    """
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]

    # GELU with tanh approximation
    out.copy_(functional.gelu(gate, approximate="tanh") * up)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GELU activation with the 'new' approximation.

    This is the approximation used in GPT-2 and some other models.

    Args:
        x: Input tensor

    Returns:
        GELU activated tensor
    """
    return functional.gelu(x, approximate="tanh")


def gelu_fast(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU approximation.

    Uses: 0.5 * x * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

    Args:
        x: Input tensor

    Returns:
        GELU activated tensor
    """
    return (
        0.5 * x * (1.0 + torch.tanh(x * 0.7978845608028654 * (1.0 + 0.044715 * x * x)))
    )


def quickgelu(x: torch.Tensor) -> torch.Tensor:
    """Quick GELU approximation.

    Uses: x * sigmoid(1.702 * x)

    Args:
        x: Input tensor

    Returns:
        GELU activated tensor
    """
    return x * torch.sigmoid(1.702 * x)


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """ReLU squared activation.

    Computes: relu(x)^2

    Used in some transformer variants.

    Args:
        x: Input tensor

    Returns:
        Activated tensor
    """
    return functional.relu(x).square()
