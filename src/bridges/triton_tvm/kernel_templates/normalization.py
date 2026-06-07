"""Hand-optimized normalization Triton kernels.

All kernels are pure ``@triton.jit`` — no vendor-specific libraries.
Tile parameters (``BLOCK_N``, ``num_warps``) are compile-time constants
selected by the expert-rules system in :mod:`src.bridges.triton_tvm.expert_rules`.

Numerical stability
-------------------
* **LayerNorm** — mean and variance computed via two-pass ``tl.sum``
  (``x``, then ``(x - mean)²``).  Normalization uses ``tl.rsqrt(var + eps)``.
* **Softmax** — subtracts the row-maximum *before* exponentiation to
  prevent overflow of ``exp(large_value)``.
* **RMSNorm** — single reduction of ``x²`` followed by ``tl.rsqrt``.

Vendor-agnostic tile shapes
---------------------------
``BLOCK_N`` must be a power of two ≥ the hidden dimension *N*.  The
expert-rules system selects a vendor-appropriate value by capping to the
nearest power-of-two that fits the target's shared-memory / register budget.

Usage
-----
::

    import triton
    from src.bridges.triton_tvm.kernel_templates import layer_norm

    grid = (num_rows,)
    layer_norm[grid](X, Y, W, B, X.stride(0), N, eps=1e-5, BLOCK_N=1024)
"""

from __future__ import annotations

import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------


@triton.jit
def layer_norm(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    stride: tl.int32,
    n: tl.int32,
    eps: tl.float32,
    BLOCK_N: tl.constexpr,  # noqa: N803
):
    """Layer normalisation with learned scale and shift.

    Computes::

        mean   = sum(x) / N
        var    = sum((x - mean)²) / N
        y      = (x - mean) / sqrt(var + eps) * weight + bias

    Each program handles one row of the input.  The row is identified by
    ``tl.program_id(0)``.

    Args:
        x_ptr:   Pointer to input tensor (rows x *N*).
        y_ptr:   Pointer to output tensor (same shape as *x_ptr*).
        weight_ptr: Learned scale vector of length *N*.
        bias_ptr:   Learned shift vector of length *N*.
        stride:  Stride between consecutive rows (i.e. *N* or the row
                 stride of the tensor).
        n:       Hidden dimension (number of columns per row).
        eps:     Small constant added to variance before ``rsqrt``.
        BLOCK_N: Compile-time tile size (power of two ≥ *n*).
    """
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride

    cols = tl.arange(0, BLOCK_N)
    mask = cols < n

    # Load row
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean
    mean = tl.sum(x, axis=0) / n

    # Variance
    x_shifted = x - mean
    var = tl.sum(x_shifted * x_shifted, axis=0) / n

    # Normalise
    inv_std = tl.rsqrt(var + eps)
    y = x_shifted * inv_std

    # Scale and shift
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * w + b

    tl.store(y_ptr + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


@triton.jit
def softmax(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    stride: tl.int32,
    n: tl.int32,
    BLOCK_N: tl.constexpr,  # noqa: N803
):
    """Numerically stable row-wise softmax.

    Computes::

        m   = max(x)               # row-wise
        e   = exp(x - m)           # subtract max before exp
        s   = sum(e)               # row-wise
        y   = e / s

    Each program handles one row.  The row is identified by
    ``tl.program_id(0)``.

    Args:
        x_ptr:   Pointer to input tensor (rows x *N*).
        y_ptr:   Pointer to output tensor (same shape).
        stride:  Stride between consecutive rows.
        n:       Number of columns per row.
        BLOCK_N: Compile-time tile size (power of two ≥ *n*).
    """
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride

    cols = tl.arange(0, BLOCK_N)
    mask = cols < n

    # Load
    x = tl.load(x_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)

    # Numerically stable softmax
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(y_ptr + cols, y, mask=mask)


# ---------------------------------------------------------------------------
# RMS Normalization (Llama / Mistral style)
# ---------------------------------------------------------------------------


@triton.jit
def rms_norm(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    stride: tl.int32,
    n: tl.int32,
    eps: tl.float32,
    BLOCK_N: tl.constexpr,  # noqa: N803
):
    """Root Mean Square normalisation (Llama / Mistral style).

    Computes::

        mean_sq  = mean(x²)
        inv_rms  = 1 / sqrt(mean_sq + eps)
        y        = x * inv_rms * weight

    Unlike LayerNorm there is no centering step — RMSNorm normalises
    by the root-mean-square of the activations only.  This is the
    normalisation used by the Llama and Mistral model families.

    Each program handles one row.

    Args:
        x_ptr:   Pointer to input tensor (rows x *N*).
        y_ptr:   Pointer to output tensor (same shape).
        weight_ptr: Learned scale vector of length *N*.
        stride:  Stride between consecutive rows.
        n:       Hidden dimension.
        eps:     Small constant for numerical stability.
        BLOCK_N: Compile-time tile size (power of two ≥ *n*).
    """
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride

    cols = tl.arange(0, BLOCK_N)
    mask = cols < n

    # Load row
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean of squares
    mean_sq = tl.sum(x * x, axis=0) / n

    # RMS normalisation (no centering)
    inv_rms = tl.rsqrt(mean_sq + eps)
    y = x * inv_rms

    # Scale
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * w

    tl.store(y_ptr + cols, y, mask=mask)
