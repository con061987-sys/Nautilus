"""2-D convolution benchmark kernel — Triton implicit-GEMM conv.

Implements NCHW conv as a single fused kernel using the
im2col trick. Each program computes a (BLOCK_M, BLOCK_N) tile
of the output, where M = N*OH*OW and N = F. The implementation
exercises direct multi-index arithmetic rather than tl.dot to
keep the kernel simple and avoid the TF32/FP16 contract.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, C, H, W, F, KH, KW, OH, OW,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr,
):
    """Compute BLOCK_M output spatial points and BLOCK_F output channels."""
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    # Decompose flattened output index [n, f, oh, ow] -> linear m
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # (BLOCK_M,)
    of = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)        # (BLOCK_F,)

    m_mask = om < N * OH * OW
    f_mask = of < F

    n_idx = om // (OH * OW)
    rem = om % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)
    bias = tl.load(B_ptr + of, mask=f_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    for c in range(0, C):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh_idx * stride_h + kh
                iw = ow_idx * stride_w + kw
                in_bounds = (ih < H) & (iw < W) & m_mask
                x_off = ((n_idx * C + c) * H + ih) * W + iw
                x_val = tl.load(X_ptr + x_off, mask=in_bounds, other=0.0).to(tl.float32)

                w_off = ((of * C + c) * KH + kh) * KW + kw
                w_mask = (tl.arange(0, BLOCK_F) < F)[:, None]
                w_val = tl.load(W_ptr + w_off, mask=f_mask, other=0.0).to(tl.float32)

                # x_val: (BLOCK_M,), w_val: (BLOCK_F,) -> outer product into (BLOCK_M, BLOCK_F)
                acc += x_val[:, None] * w_val[None, :]

    y_off = ((n_idx[:, None] * F + of[None, :]) * OH + oh_idx[:, None]) * OW + ow_idx[:, None]
    out_mask = m_mask[:, None] & f_mask[None, :]
    tl.store(Y_ptr + y_off, acc, mask=out_mask)


def reference_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
) -> torch.Tensor:
    """Torch reference: NCHW conv via torch.nn.functional.conv2d."""
    return torch.nn.functional.conv2d(x, weight, bias=bias, stride=stride)
