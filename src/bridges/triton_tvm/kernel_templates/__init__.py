"""Kernel templates for common GPU operations.

Each module in this package provides hand-optimized ``@triton.jit``
kernels for a specific family of operations.  The kernels are pure
Triton — no vendor libraries — with compile-time tile parameters
that are selected by :mod:`src.bridges.triton_tvm.expert_rules`.

Available modules:

* **normalization** — ``layer_norm``, ``softmax``, ``rms_norm``
  (numerically stable, row-wise reduction kernels)
* **attention** — ``attention_h100``, ``attention_mi300x``, ``attention_gaudi``,
  ``attention_apple``
  (vendor-optimized FlashAttention-2 kernels with online softmax)
* **matmul** — ``matmul_h100``, ``matmul_mi300x``, ``matmul_gaudi``, ``matmul_apple``
  (vendor-optimized matrix multiplication with compile-time tile parameters)
"""

from __future__ import annotations

from .attention import (
    attention_apple,
    attention_gaudi,
    attention_h100,
    attention_mi300x,
)
from .matmul import matmul_apple, matmul_gaudi, matmul_h100, matmul_mi300x
from .normalization import layer_norm, rms_norm, softmax

__all__ = [
    "attention_apple",
    "attention_gaudi",
    "attention_h100",
    "attention_mi300x",
    "layer_norm",
    "matmul_apple",
    "matmul_gaudi",
    "matmul_h100",
    "matmul_mi300x",
    "rms_norm",
    "softmax",
]
