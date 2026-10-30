# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Any, Literal

import torch
from absl import logging

from emerging_optimizers import triton_kernels, utils

__all__ = [
    "msign",
    "newton_schulz",
    "newton_schulz_tp",
    "orthogonal_weight_decay_direction",
]

_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        # optimized for a quintic iteration.
        # Source: https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        # Numbers from: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L44
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        # Polar Express iteration from: https://arxiv.org/abs/2505.16932
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ],
    "aol": [
        # from https://github.com/thib-s/flash-newton-schulz/blob/main/newton_schulz_triton.py#L511
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}


_MSIGN_COEFFICIENTS = [
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
]


@torch.no_grad()
def _small_msign(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute a matrix-sign approximation with the Muon coefficient schedule."""
    if matrix.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if matrix.dtype != torch.float32:
        raise ValueError(f"Input tensor must be in float32, got {matrix.dtype}")

    transpose = matrix.size(-2) > matrix.size(-1)
    output = matrix.mT if transpose else matrix
    output = torch.nn.functional.normalize(output, p=2, dim=(-2, -1), eps=1e-7)
    output = output.to(torch.bfloat16)

    for index in range(steps):
        a, b, c = _MSIGN_COEFFICIENTS[min(index, len(_MSIGN_COEFFICIENTS) - 1)]
        output = newton_schulz_step(output, a, b, c)

    return output.mT if transpose else output


@torch.compile  # type: ignore[misc]
@torch.no_grad()
def _large_msign(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute matrix sign through the shared, optionally accelerated Muon path."""
    with utils.fp32_matmul_precision("medium"):
        return newton_schulz(
            matrix,
            steps=steps,
            coefficient_type="custom",
            custom_coefficient_sets=_MSIGN_COEFFICIENTS,
            use_syrk=True,
        )


@torch.no_grad()
def msign(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute the matrix-sign approximation used by the Muon optimizer."""
    return _small_msign(matrix, steps)


def _is_tsyrk_compatible(x: torch.Tensor) -> bool:
    """Return whether ``x`` and its SYRK output satisfy TMA descriptor alignment."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or any(dim <= 0 for dim in x.shape):
        return False
    if not (x.is_contiguous() or x.mT.is_contiguous()):
        return False

    elem_bytes = x.element_size()
    n = x.shape[0]

    # tsyrk_ex stores an NxN row-major result, which must also have an aligned row stride.
    if (n * elem_bytes) % 16 != 0:
        return False

    # Match tsyrk_ex's choice of the contiguous view used to construct a_desc.
    descriptor_input = x.mT if x.mT.is_contiguous() else x
    if descriptor_input.stride(-1) != 1:
        return False
    return all(
        (stride * elem_bytes) % 16 == 0 for stride in descriptor_input.stride()[:-1]
    )


def orthogonal_weight_decay_direction(
    matrix: torch.Tensor, use_syrk: bool = True, alpha: float = 1.0
) -> torch.Tensor:
    """Return ``(W W^T - alpha I) W`` for a matrix or a batch of matrices.

    ``alpha=1`` preserves the original orthogonal weight-decay direction.
    The smaller Gram matrix is formed to reduce both FLOPs and temporary memory.
    CUDA BF16 inputs use the existing Triton SYRK implementation when available;
    the final product is a general matrix multiplication and therefore uses
    addmm/baddbmm rather than a symmetric kernel.
    """
    if (
        not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or alpha < 0.0
    ):
        raise ValueError(
            "Orthogonal weight decay alpha must be finite and non-negative, "
            f"got {alpha}."
        )
    alpha = float(alpha)
    if matrix.ndim < 2:
        raise ValueError(
            "Orthogonal weight decay expects a matrix or batch of matrices, "
            f"got shape {tuple(matrix.shape)}."
        )
    if matrix.size(-2) == 0 or matrix.size(-1) == 0:
        raise ValueError("Orthogonal weight decay does not support empty matrices.")

    rows, cols = matrix.shape[-2:]
    gram_input = matrix if rows <= cols else matrix.mT
    can_use_syrk = False
    if use_syrk and gram_input.is_cuda and gram_input.dtype == torch.bfloat16:
        if gram_input.ndim == 2:
            can_use_syrk = bool(
                triton_kernels.HAS_TRITON_340  # type: ignore[attr-defined]
                and _is_tsyrk_compatible(gram_input)
            )
        else:
            can_use_syrk = bool(
                getattr(triton_kernels, "HAS_TRITON_BATCHED_SYRK", False)
            )

    gram = (
        triton_kernels.tsyrk_ex(gram_input)  # type: ignore[attr-defined]
        if can_use_syrk
        else gram_input @ gram_input.mT
    )

    if matrix.ndim == 2:
        if rows <= cols:
            return torch.addmm(matrix, gram, matrix, beta=-alpha)
        return torch.addmm(matrix, matrix, gram, beta=-alpha)

    batch_shape = matrix.shape[:-2]
    matrix_3d = matrix.reshape(-1, rows, cols)
    gram_3d = gram.reshape(-1, gram.size(-2), gram.size(-1))
    if rows <= cols:
        direction = torch.baddbmm(matrix_3d, gram_3d, matrix_3d, beta=-alpha)
    else:
        direction = torch.baddbmm(matrix_3d, matrix_3d, gram_3d, beta=-alpha)
    return direction.reshape(batch_shape + (rows, cols))


def distributed_normalize_p2(
    x: torch.Tensor, eps: float, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Normalize a tensor in a distributed way."""
    # Keep leading dimensions independent when a batch of matrices is passed.
    # This is equivalent to the scalar reduction for a 2D input.
    x_sq_sum = (x * x).sum(dim=(-2, -1))
    torch.distributed.all_reduce(
        x_sq_sum, op=torch.distributed.ReduceOp.SUM, group=group
    )
    scale = torch.sqrt(x_sq_sum).clamp_min(eps)
    return x / scale.reshape(x.shape[:-2] + (1, 1))


def newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str = "polar_express",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    eps: float = 1e-7,
    transpose: bool | None = None,
    tp_group: torch.distributed.ProcessGroup | None = None,
    use_syrk: bool = False,
) -> torch.Tensor:
    """Use Newton-Schulz iteration to compute the zeroth power / orthogonalization of x.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of x. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero and minimize variance.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the
    slope at zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce :math:`UV^T` but rather something like :math:`US'V^T`
    where :math:`S'` is diagonal with noisy values around 1, which turns out not to hurt model performance
    at all relative to :math:`UV^T`, where :math:`USV^T = G` is the SVD.


    Parameter ``coefficient_type`` can be one of the following
      - "simple": Single fixed coefficient set.
      - "quintic": Quintic iteration with optimized coefficients.
      - "polar_express": Polar Express iteration with optimized coefficients.
      - "aol": AOL coefficient schedule.
      - "custom": Custom coefficient sets.

    Arguments:
        x: The tensor to be orthogonalized.
        steps: Number of Newton-Schulz iterations.
        coefficient_type: Type of coefficient set to use for the Newton-Schulz iteration.
        custom_coefficient_sets: Custom coefficient sets to use for the Newton-Schulz iteration.
        eps: Small constant to avoid division by zero.
        transpose: Whether to transpose the tensor to perform whitening on the smaller dimension.
            If None, will be determined based on the size of the tensor.
        tp_group: The process group for communication if input is distributed.
        use_syrk: Whether to prefer a Triton SYRK kernel. The 2D path uses TensorDescriptor/TMA,
            while tensors with leading batch dimensions use a strided batched kernel. Falls back
            to PyTorch when the applicable kernel or BF16 input is unavailable.

    Returns:
        The orthogonalization of x.
    """
    # Muon is not for 1d parameters
    if x.ndim < 2:
        raise ValueError(
            "Input tensor x must have at least 2 dimensions since Muon is not for 1d parameters."
        )
    if x.dtype != torch.float32:
        raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")

    # transpose tensor to perform whitening on the smaller dimension
    if transpose is None:
        transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    # Ensure spectral norm is at most 1
    if tp_group is not None:
        X = distributed_normalize_p2(x, eps, tp_group)
    else:
        X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)

    if coefficient_type in _COEFFICIENT_SETS:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    elif coefficient_type == "custom":
        if custom_coefficient_sets is None:
            raise ValueError(
                "custom_coefficient_sets must be provided when coefficient_type is 'custom'."
            )
        coefficient_sets = custom_coefficient_sets
    else:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")

    if steps % len(coefficient_sets) != 0:
        raise ValueError(
            f"steps ({steps}) must be multiple of len(coefficient_sets) ({len(coefficient_sets)})."
        )

    ns_step_fn = newton_schulz_step
    if torch.get_float32_matmul_precision() == "medium":
        # PyTorch doesn't provide FP32 I/O with BF16 compute for this path, so use BF16 I/O explicitly.
        X = X.to(torch.bfloat16)
        logging.log_first_n(
            logging.INFO, "Using BF16 I/O kernels for Newton-Schulz iteration.", 1
        )

    if use_syrk:
        if X.ndim >= 3:
            if X.dtype != torch.bfloat16:
                logging.log_first_n(
                    logging.WARNING,
                    "Batched Triton SYRK requires BF16 input; falling back to PyTorch Newton-Schulz.",
                    1,
                )
            elif getattr(triton_kernels, "HAS_TRITON_BATCHED_SYRK", False):
                # The batched kernel uses strided pointer arithmetic and does
                # not require the 2D TensorDescriptor/TMA API.
                ns_step_fn = newton_schulz_step_tsyrk
            else:
                logging.log_first_n(
                    logging.WARNING,
                    "Batched Triton SYRK is unavailable; falling back to PyTorch Newton-Schulz.",
                    1,
                )
        elif triton_kernels.HAS_TRITON_340:  # type: ignore[attr-defined]
            if X.dtype == torch.bfloat16:
                if _is_tsyrk_compatible(X):
                    ns_step_fn = newton_schulz_step_tsyrk
                else:
                    logging.log_first_n(
                        logging.WARNING,
                        f"Triton SYRK tensor descriptor alignment is unsupported for "
                        f"shape={tuple(X.shape)}, stride={X.stride()}; "
                        "falling back to PyTorch Newton-Schulz.",
                        1,
                    )
            else:
                logging.log_first_n(
                    logging.WARNING,
                    "Triton SYRK requires BF16 input; falling back to PyTorch Newton-Schulz.",
                    1,
                )
        else:
            logging.log_first_n(
                logging.WARNING,
                "Triton TensorDescriptor API is unavailable; falling back to PyTorch Newton-Schulz.",
                1,
            )

    for i in range(steps):
        a, b, c = coefficient_sets[i % len(coefficient_sets)]
        X = ns_step_fn(X, a, b, c, tp_group=tp_group)

    # Convert back to FP32. This is a noop if X is already in FP32.
    X = X.to(torch.float32)

    # undo transpose if necessary
    if transpose:
        X = X.mT
    return X


def newton_schulz_tp(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group: torch.distributed.ProcessGroup,
    partition_dim: int | None = None,
    mode: Literal["duplicated", "distributed"] = "duplicated",
    use_syrk: bool = True,
) -> torch.Tensor:
    """Tensor Parallel Newton-Schulz iteration.

    This function uses partition_dim to determine along which dimension the input tensor is sharded. Transpose is
    set based on the partition_dim. If partition_dim is None, the input tensor is not sharded and the function will
    fall back to the non-TP path.

    Warning:
        If partition_dim is the smaller dim of the input tensor, `distributed` mode will run Newton-Schulz along the
        long dimension which wastes compute.
        Although we reuse the partition_dim name, the default value is None which means no partition instead of -1.

    Note:
        This function is designed to provide tensor parallel support for most common use of Newton-Schulz.
        Many arguments, e.g. custom coefficient sets and custom eps, are not supported.
        It prefers the Triton SYRK kernel and falls back to PyTorch matrix operations when unavailable.

    ``mode`` can be one of the following:
        - "duplicated": The input tensor is duplicated and orthogonalized on each rank.
        - "distributed": The input tensor is partitioned along the partition_dim and orthogonalized on each rank.

    Args:
        x: The tensor to be orthogonalized. Must has partition_dim and tensor_model_parallel set by TransformerEngine.
        steps: Number of Newton-Schulz iterations.
        coefficient_type: Type of coefficient set to use for the Newton-Schulz iteration.
        partition_dim: The dimension to partition the tensor.
        tp_group: The process group for communication if input is distributed.
        mode: The mode to use for the Newton-Schulz iteration.
        use_syrk: Whether to use the custom Triton SYRK kernels.
    """
    if partition_dim is None:
        # Fallback path for non TP params.
        return newton_schulz(x, steps, coefficient_type, use_syrk=use_syrk)

    kwargs: Any = {
        "steps": steps,
        "coefficient_type": coefficient_type,
        "use_syrk": use_syrk,
    }

    if mode == "duplicated":
        x_shards = [torch.empty_like(x) for _ in range(tp_group.size())]
        torch.distributed.all_gather(x_shards, x, tp_group)
        global_x = torch.cat(x_shards, dim=partition_dim)

        orthogonalized_x = newton_schulz(global_x, tp_group=None, **kwargs)
        output = orthogonalized_x.chunk(tp_group.size(), dim=partition_dim)[
            tp_group.rank()
        ]
    elif mode == "distributed":
        if partition_dim == 0:
            transpose = True
        elif partition_dim == 1:
            transpose = False
        else:
            raise ValueError(f"Invalid partition_dim: {partition_dim}")
        output = newton_schulz(x, **kwargs, transpose=transpose, tp_group=tp_group)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return output


def newton_schulz_step(
    X: torch.Tensor,
    a: float,
    b: float,
    c: float,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Perform one Newton-Schulz step using PyTorch matrix operations."""
    A = X @ X.mT
    if tp_group is not None:
        torch.distributed.all_reduce(
            A, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )
    if X.ndim == 2:
        B = torch.addmm(A, A, A, alpha=c, beta=b)
        X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    else:
        batch_shape = X.shape[:-2]
        X_3d = X.reshape(-1, X.size(-2), X.size(-1))
        A_3d = A.reshape(-1, A.size(-2), A.size(-1))
        B_3d = torch.baddbmm(A_3d, A_3d, A_3d, beta=b, alpha=c)
        X = torch.baddbmm(X_3d, B_3d, X_3d, beta=a).reshape(batch_shape + X.shape[-2:])
    return X


def newton_schulz_step_tsyrk(
    X: torch.Tensor,
    a: float,
    b: float,
    c: float,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Perform a single Newton-Schulz iteration step.

    This function performs a single Newton-Schulz iteration step using the Triton kernel for extended syrk.

    Arguments:
        X: The tensor to be orthogonalized. Must be bfloat16.
        a: The a coefficient.
        b: The b coefficient.
        c: The c coefficient.
        tp_group: The process group to use for the all-reduce.

    Returns:
        The orthogonalization of X.
    """
    if X.ndim == 2:
        assert triton_kernels.HAS_TRITON_340, (  # type: ignore[attr-defined]
            "Triton version doesn't support tensor descriptor API. Minimum required version is 3.4.0."
        )
    else:
        assert getattr(triton_kernels, "HAS_TRITON_BATCHED_SYRK", False), (
            "Batched Triton SYRK is unavailable."
        )
    A = triton_kernels.tsyrk_ex(X)  # type: ignore[attr-defined]
    if tp_group is not None:
        torch.distributed.all_reduce(
            A, op=torch.distributed.ReduceOp.SUM, group=tp_group
        )
    B = triton_kernels.tsyrk_ex(A, A, alpha=c, beta=b)  # type: ignore[attr-defined]
    if X.ndim == 2:
        X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    else:
        batch_shape = X.shape[:-2]
        X_3d = X.reshape(-1, X.size(-2), X.size(-1))
        B_3d = B.reshape(-1, B.size(-2), B.size(-1))
        X = torch.baddbmm(X_3d, B_3d, X_3d, beta=a).reshape(batch_shape + X.shape[-2:])
    return X
