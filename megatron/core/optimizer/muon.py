# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron muon optimizer wrapper to handle tensor-parallel."""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, List, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig
from .orca_adamw import ORCAAdamW

try:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers import utils as emerging_utils
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        newton_schulz_tp,
        orthogonal_weight_decay_direction,
    )
    from emerging_optimizers import mixin as opt_mixin

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object


logger = logging.getLogger(__name__)


ORCA_OPTIMIZERS = frozenset(
    {
        "orca",
        "orca_layerwise",
    }
)

_LEGACY_ORCA_METADATA_KEYS = ("period6muon", "period6muon_layerwise")

def is_orca_optimizer(name: str) -> bool:
    """Return whether ``name`` selects one of the ORCA variants."""
    return name in ORCA_OPTIMIZERS


@dataclass
class MuonLogicalMatrixTask:
    """One independently orthogonalized matrix backed by an optimizer update tensor.

    ``read`` materializes the logical 2D matrix only when its bucket is executed.
    This matters for fused QKV weights: a component spanning multiple query groups
    is not contiguous in the fused parameter layout. ``write`` scatters the result
    back into the original update tensor, so batching never changes the persistent
    parameter, FP32 master-weight, or momentum-buffer layout.
    """

    shape: tuple[int, int]
    dtype: torch.dtype
    device: torch.device
    role: str
    qkv_mode: Optional[str]
    tp_group: Optional[torch.distributed.ProcessGroup]
    partition_dim: Optional[int]
    read: Callable[[], torch.Tensor]
    write: Callable[[torch.Tensor], None]

    def bucket_key(self) -> tuple[Hashable, ...]:
        """Return the properties that must agree for a safe batched NS call."""
        return (
            self.role,
            self.qkv_mode,
            self.shape,
            self.dtype,
            self.device.type,
            self.device.index,
            id(self.tp_group),
            self.partition_dim,
        )


def _resolve_fc1_split_shapes(
    p: torch.Tensor,
    tensor: torch.Tensor,
    default_split_shapes: tuple[int, int] | None,
) -> tuple[int, int]:
    """Resolve the local SwiGLU gate/up sizes for one FC1 parameter.

    Dense, shared-expert, and routed-expert FC1 weights can have different
    output widths in the same model. Prefer metadata recorded from the local
    parameter shape; the optimizer-wide value is only a compatibility fallback.
    """
    param_name = getattr(p, 'param_name', '<unnamed>')
    split_shapes = getattr(p, 'fc1_split_shapes', None)
    if split_shapes is None and default_split_shapes is not None:
        candidate = tuple(int(size) for size in default_split_shapes)
        if len(candidate) == 2 and sum(candidate) == tensor.size(0):
            split_shapes = candidate

    if split_shapes is None:
        if tensor.size(0) % 2 != 0:
            raise ValueError(
                f"Cannot split SwiGLU FC1 parameter {param_name!r} with shape "
                f"{tuple(tensor.shape)} into equal gate/up matrices."
            )
        local_ffn_size = tensor.size(0) // 2
        split_shapes = (local_ffn_size, local_ffn_size)

    split_shapes = tuple(int(size) for size in split_shapes)
    if (
        len(split_shapes) != 2
        or any(size <= 0 for size in split_shapes)
        or sum(split_shapes) != tensor.size(0)
    ):
        raise ValueError(
            f"Invalid SwiGLU FC1 split shapes {split_shapes} for parameter "
            f"{param_name!r} with shape {tuple(tensor.shape)}."
        )
    return split_shapes


def _map_orca_qkv_matrices(
    tensor: torch.Tensor,
    qkv_split_shapes: tuple[int, int, int],
    matrix_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    qk_matrix_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    v_matrix_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Apply ``matrix_fn`` using ORCA's fused-QKV matrix boundaries.

    Q is split by attention head, K by query group, and all V query groups
    are merged into one matrix. Q and K have the same matrix shape and are
    processed together as one batch, exactly as in ORCA.
    """
    q_per_group, k_per_group, v_per_group = qkv_split_shapes
    rows, hidden = tensor.shape
    rows_per_group = sum(qkv_split_shapes)
    if (
        min(qkv_split_shapes) <= 0
        or rows % rows_per_group != 0
        or q_per_group % k_per_group != 0
    ):
        raise ValueError(
            f"Cannot apply ORCA QKV split to shape {tuple(tensor.shape)} "
            f"with split shapes {qkv_split_shapes}."
        )

    num_groups = rows // rows_per_group
    q_heads_per_group = q_per_group // k_per_group
    grouped = tensor.view(num_groups, rows_per_group, hidden)
    q, k, v = torch.split(grouped, list(qkv_split_shapes), dim=1)

    q_matrices = q.reshape(num_groups * q_heads_per_group, k_per_group, hidden)
    k_matrices = k.reshape(num_groups, k_per_group, hidden)
    v_matrix = v.reshape(num_groups * v_per_group, hidden)
    qk_matrices = torch.cat((q_matrices, k_matrices), dim=0)
    qk = (qk_matrix_fn or matrix_fn)(qk_matrices)
    q_count = q_matrices.size(0)
    q = qk[:q_count].reshape(num_groups, q_per_group, hidden)
    k = qk[q_count:].reshape(num_groups, k_per_group, hidden)
    v = (v_matrix_fn or matrix_fn)(v_matrix).reshape(num_groups, v_per_group, hidden)
    return torch.cat((q, k, v), dim=1).reshape_as(tensor)


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        qkv_split_mode: str = "component",  # "component", "group", "head", or "orca"
        # FC1 split support for gated linear units (SwiGLU)
        split_fc1: bool = False,
        is_fc1_fn: Callable[[torch.Tensor], bool] | None = None,
        fc1_split_shapes: tuple[int, int] | None = None,  # (gate_dim, up_dim)
        # MoE expert split support for GroupedMLP
        split_moe_experts: bool = False,
        is_grouped_moe_fn: Callable[[torch.Tensor], bool] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "polar_express",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        use_triton_optimize: bool = False,
        optimizer_max_batch_size: int = 32,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if optimizer_max_batch_size < 1:
            raise ValueError(
                "optimizer_max_batch_size must be at least 1, got "
                f"{optimizer_max_batch_size}"
            )

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                mode="duplicated" if mode == "blockwise" else mode,
                use_syrk=use_triton_optimize,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode
        self.use_triton_optimize = bool(use_triton_optimize)
        self.optimizer_max_batch_size = int(optimizer_max_batch_size)
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_split_mode = qkv_split_mode
        # FC1 split for gated linear units
        self.split_fc1 = split_fc1
        self.is_fc1_fn = is_fc1_fn
        self.fc1_split_shapes = fc1_split_shapes
        # MoE expert split for GroupedMLP
        self.split_moe_experts = split_moe_experts
        self.is_grouped_moe_fn = is_grouped_moe_fn

        # https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/fe29e5670fc0dadf1f10ab267a0edfa6e1b89fb3/emerging_optimizers/orthogonalized_optimizers/muon.py#L71
        if use_decoupled_weight_decay:
            self.weight_decay_method = "decoupled"
        else:
            raise NotImplementedError

        super().__init__(
            params,
            lr,
            momentum_beta,
            weight_decay,
            use_nesterov=use_nesterov,
            weight_decay_method=self.weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
            log_per_module_update_rms=False,  # Will be set later via config
        )

    def _tp_metadata(
        self, p: torch.Tensor
    ) -> tuple[Optional[torch.distributed.ProcessGroup], Optional[int]]:
        """Resolve the process group and matrix partition metadata for ``p``."""
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None
        return tp_group, partition_dim

    @staticmethod
    def _is_te_grouped_expert(p: torch.Tensor) -> bool:
        """Identify one independently stored TE GroupedLinear expert weight."""
        if getattr(p, "is_te_grouped_expert", False):
            return True
        name = getattr(p, "param_name", "")
        return (
            "shared" not in name
            and (
                "experts.linear_fc1.weight" in name
                or "experts.linear_fc2.weight" in name
            )
        )

    @staticmethod
    def _make_logical_task(
        tensor: torch.Tensor,
        *,
        shape: tuple[int, int],
        role: str,
        qkv_mode: Optional[str],
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
        read: Callable[[], torch.Tensor],
        write: Callable[[torch.Tensor], None],
    ) -> MuonLogicalMatrixTask:
        return MuonLogicalMatrixTask(
            shape=shape,
            dtype=tensor.dtype,
            device=tensor.device,
            role=role,
            qkv_mode=qkv_mode,
            tp_group=tp_group,
            partition_dim=partition_dim,
            read=read,
            write=write,
        )

    def _task_for_matrix_view(
        self,
        update: torch.Tensor,
        matrix_view: torch.Tensor,
        *,
        role: str,
        qkv_mode: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        partition_dim: Optional[int] = None,
    ) -> MuonLogicalMatrixTask:
        """Create a task for a 2D logical matrix view of one update tensor."""
        shape = (matrix_view.size(-2), matrix_view.size(-1))
        return self._make_logical_task(
            update,
            shape=shape,
            role=role,
            qkv_mode=qkv_mode,
            tp_group=tp_group,
            partition_dim=partition_dim,
            read=lambda view=matrix_view: view,
            write=lambda value, view=matrix_view: self._copy_task_result(view, value),
        )

    @staticmethod
    def _copy_task_result(view: torch.Tensor, value: torch.Tensor) -> None:
        """Write a task result without attaching optimizer scratch to autograd."""
        with torch.no_grad():
            view.copy_(value)

    def _logical_matrix_tasks(
        self, p: torch.Tensor, update: torch.Tensor
    ) -> Optional[list[MuonLogicalMatrixTask]]:
        """Describe batchable logical matrices without changing parameter layout.

        The first implementation deliberately batches only local (TP-size one)
        matrices. TP collectives retain the existing per-parameter path. QKV modes
        other than ``component`` also retain their existing matrix boundaries and
        execution path.
        """
        tp_group, partition_dim = self._tp_metadata(p)
        if get_pg_size(tp_group) != 1 or partition_dim is not None:
            return None

        # Legacy GroupedMLP stores all experts in one fused parameter and has
        # its own reshape/orthogonalization semantics below. Do not let the
        # generic 2D task path bypass that mapping.
        if (
            self.split_moe_experts
            and self.is_grouped_moe_fn is not None
            and self.is_grouped_moe_fn(p)
        ):
            return None

        if (
            self.split_qkv
            and self.is_qkv_fn is not None
            and self.is_qkv_fn(p)
        ):
            if self.qkv_split_mode != "component":
                return None
            if self.qkv_split_shapes is None:
                raise ValueError("qkv_split_shapes must be provided for QKV splitting")
            split_shapes = tuple(int(size) for size in self.qkv_split_shapes)
            rows_per_group = sum(split_shapes)
            if (
                len(split_shapes) != 3
                or min(split_shapes) <= 0
                or update.dim() != 2
                or update.size(0) % rows_per_group != 0
            ):
                raise ValueError(
                    f"Cannot apply component QKV split to shape {tuple(update.shape)} "
                    f"with split shapes {split_shapes}."
                )

            num_query_groups = update.size(0) // rows_per_group
            hidden = update.size(1)
            fused_view = update.view(num_query_groups, rows_per_group, hidden)
            component_views = torch.split(fused_view, list(split_shapes), dim=1)
            tasks = []
            for component_index, (component_view, component_rows) in enumerate(
                zip(component_views, split_shapes)
            ):
                matrix_shape = (num_query_groups * component_rows, hidden)
                # K and V intentionally share a role and therefore a bucket. Q
                # retains its full component boundary and a separate semantic role.
                role = "qkv_q_component" if component_index == 0 else "qkv_kv_component"
                tasks.append(
                    self._make_logical_task(
                        update,
                        shape=matrix_shape,
                        role=role,
                        qkv_mode=self.qkv_split_mode,
                        tp_group=tp_group,
                        partition_dim=partition_dim,
                        read=lambda view=component_view, shape=matrix_shape: view.reshape(shape),
                        write=lambda value, view=component_view: self._copy_task_result(
                            view, value.view_as(view)
                        ),
                    )
                )
            return tasks

        is_te_expert = self._is_te_grouped_expert(p)
        if (
            self.split_fc1
            and self.is_fc1_fn is not None
            and self.is_fc1_fn(p)
        ):
            gate_dim, up_dim = _resolve_fc1_split_shapes(p, update, self.fc1_split_shapes)
            gate, up = torch.split(update, [gate_dim, up_dim], dim=0)
            role = "te_expert_fc1_gate_up" if is_te_expert else "fc1_gate_up"
            return [
                self._task_for_matrix_view(
                    update, gate, role=role, tp_group=tp_group, partition_dim=partition_dim
                ),
                self._task_for_matrix_view(
                    update, up, role=role, tp_group=tp_group, partition_dim=partition_dim
                ),
            ]

        if is_te_expert:
            name = getattr(p, "param_name", "")
            role = (
                "te_expert_fc1_fused"
                if "linear_fc1" in name
                else "te_expert_fc2"
            )
            return [
                self._task_for_matrix_view(
                    update, update, role=role, tp_group=tp_group, partition_dim=partition_dim
                )
            ]

        # Ordinary 2D weights are independent logical matrices as well. They
        # share the generic role so same-shaped layers can use one batched NS
        # call, while fused QKV/GLU/expert paths above retain explicit roles.
        if update.dim() == 2:
            return [
                self._task_for_matrix_view(
                    update, update, role="matrix", tp_group=tp_group, partition_dim=partition_dim
                )
            ]
        return None

    def _run_logical_matrix_tasks(
        self,
        tasks: list[MuonLogicalMatrixTask],
        operation: Optional[
            Callable[
                [torch.Tensor, Optional[torch.distributed.ProcessGroup], Optional[int]],
                torch.Tensor,
            ]
        ] = None,
    ) -> None:
        """Bucket and execute independent logical matrices in bounded batches.

        ``operation`` defaults to Muon's scaled Newton-Schulz transform.  The
        same bounded runner is also used by ORCA's orthogonal weight decay,
        whose operation returns the updated matrix rather than an update
        direction.
        """
        if operation is None:
            operation = lambda matrix, tp_group, partition_dim: self.scaled_orthogonalize_fn(
                matrix, tp_group, partition_dim
            )

        buckets: dict[tuple[Hashable, ...], list[MuonLogicalMatrixTask]] = defaultdict(list)
        for task in tasks:
            buckets[task.bucket_key()].append(task)

        max_batch_size = self.optimizer_max_batch_size
        for bucket in buckets.values():
            for start in range(0, len(bucket), max_batch_size):
                chunk = bucket[start : start + max_batch_size]
                if len(chunk) == 1:
                    task = chunk[0]
                    result = operation(task.read(), task.tp_group, task.partition_dim)
                    task.write(result)
                    continue

                # Copy one source at a time. Fused QKV components may require a
                # materializing reshape; avoiding a list of all such tensors
                # keeps peak input staging close to one bounded batch.
                matrices = torch.empty(
                    (len(chunk), *chunk[0].shape),
                    dtype=chunk[0].dtype,
                    device=chunk[0].device,
                )
                for matrix_index, task in enumerate(chunk):
                    matrices[matrix_index].copy_(task.read())
                result = operation(matrices, chunk[0].tp_group, chunk[0].partition_dim)
                for task, matrix_result in zip(chunk, result.unbind(0)):
                    task.write(matrix_result)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Update momentum per parameter, then batch compatible logical matrices.

        ``p.grad`` is reused as transient write-back storage after gradient norm
        and zero counting have completed in the Megatron wrapper. Persistent state
        remains one momentum tensor per FP32 optimizer parameter, so batching does
        not add a model-sized temporary or change checkpoint layouts.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self.log_per_module_update_rms:
            self.per_module_update_rms.clear()
        if self.log_per_module_grad_rms:
            self.per_module_grad_rms.clear()

        pending_updates: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []
        logical_tasks: list[MuonLogicalMatrixTask] = []
        fallback_updates: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() == 1:
                    raise ValueError(f"{self.__class__.__name__} does not support 1D parameters")
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                exp_avg = state["momentum_buffer"]

                effective_weight_decay = group.get("wd_mult", 1.0) * group["weight_decay"]
                self._apply_weight_decay_inplace(
                    p, grad, group["lr"], effective_weight_decay
                )
                exp_avg.lerp_(grad, 1 - group["momentum_beta"])

                # Use the gradient allocation as scratch, not the persistent
                # momentum tensor. This bounds extra memory to the current
                # matrix batch instead of retaining one update per parameter.
                if self.use_nesterov:
                    grad.lerp_(exp_avg, group["momentum_beta"])
                else:
                    grad.copy_(exp_avg)
                update = grad

                param_name = getattr(p, "param_name", None)
                if self.log_per_module_grad_rms and param_name:
                    self.per_module_grad_rms[param_name] = torch.sqrt(
                        torch.mean(update**2)
                    ).item()

                group_kwargs = {k: v for k, v in group.items() if k != "params"}
                pending_updates.append((p, update, group_kwargs))
                tasks = self._logical_matrix_tasks(p, update)
                if tasks is None:
                    fallback_updates.append((p, update, group_kwargs))
                else:
                    logical_tasks.extend(tasks)

        with emerging_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            self._run_logical_matrix_tasks(logical_tasks)
            for p, update, group_kwargs in fallback_updates:
                update.copy_(self.orthogonalize(p, update, **group_kwargs))

        for p, update, group_kwargs in pending_updates:
            param_name = getattr(p, "param_name", None)
            if self.log_per_module_update_rms and param_name:
                self.per_module_update_rms[param_name] = torch.sqrt(
                    torch.mean(update**2)
                ).item()
            p.add_(update, alpha=-group_kwargs["lr"])

        return loss

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        tp_group, partition_dim = self._tp_metadata(p)

        if self.split_moe_experts and self.is_grouped_moe_fn is not None and self.is_grouped_moe_fn(p):  # type: ignore[misc]
            # Split GroupedMLP weight1/weight2 by experts
            # weight1: [hidden_size, num_experts * ffn_hidden_size] (or partitioned version)
            # weight2: [num_experts * ffn_hidden_size, hidden_size] (or partitioned version)
            num_local_experts = getattr(p, 'num_local_experts', None)
            if num_local_experts is None or num_local_experts <= 1:
                # If num_local_experts is not set or is 1, fall back to default behavior
                grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
            else:
                grad_shape = grad.shape
                param_name = getattr(p, 'param_name', '')
                batch_local_experts = (
                    partition_dim is None and get_pg_size(tp_group) == 1
                )

                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f'MoE expert split for {param_name}, grad shape {grad_shape}, num_experts {num_local_experts}',
                )

                if 'weight1' in param_name:
                    # weight1: [hidden_size, num_experts * ffn_per_expert]
                    # Need to account for gated linear units which double the output size
                    is_gated = getattr(p, 'is_gated', False)
                    ffn_multiplier = 2 if is_gated else 1
                    ffn_dim_per_expert = grad_shape[1] // (num_local_experts * ffn_multiplier)

                    if batch_local_experts:
                        if self.split_fc1 and is_gated:
                            # [H, E, 2F] -> [E * 2, H, F]. Each gate/up matrix
                            # remains independent inside the batched update.
                            grad_batched = (
                                grad.view(
                                    grad_shape[0],
                                    num_local_experts,
                                    ffn_multiplier,
                                    ffn_dim_per_expert,
                                )
                                .permute(1, 2, 0, 3)
                                .reshape(
                                    num_local_experts * ffn_multiplier,
                                    grad_shape[0],
                                    ffn_dim_per_expert,
                                )
                            )
                            grad_batched = self.scaled_orthogonalize_fn(
                                grad_batched, tp_group, partition_dim
                            )
                            grad = (
                                grad_batched.view(
                                    num_local_experts,
                                    ffn_multiplier,
                                    grad_shape[0],
                                    ffn_dim_per_expert,
                                )
                                .permute(2, 0, 1, 3)
                                .reshape(grad_shape)
                            )
                        else:
                            # [H, E, F] -> [E, H, F]
                            grad_batched = (
                                grad.view(
                                    grad_shape[0],
                                    num_local_experts,
                                    ffn_dim_per_expert * ffn_multiplier,
                                )
                                .permute(1, 0, 2)
                                .contiguous()
                            )
                            grad_batched = self.scaled_orthogonalize_fn(
                                grad_batched, tp_group, partition_dim
                            )
                            grad = grad_batched.permute(1, 0, 2).reshape(grad_shape)
                    else:
                        # Preserve the existing per-expert path for tensor-parallel
                        # parameters until batched TP collectives are supported.
                        grad_reshaped = grad.view(
                            grad_shape[0],
                            num_local_experts,
                            ffn_dim_per_expert * ffn_multiplier,
                        )
                        expert_grads = []
                        for expert_idx in range(num_local_experts):
                            expert_grad = grad_reshaped[:, expert_idx, :]
                            if self.split_fc1 and is_gated:
                                gate_grad, up_grad = torch.split(
                                    expert_grad,
                                    [ffn_dim_per_expert, ffn_dim_per_expert],
                                    dim=1,
                                )
                                gate_grad_orth = self.scaled_orthogonalize_fn(
                                    gate_grad, tp_group, partition_dim
                                )
                                up_grad_orth = self.scaled_orthogonalize_fn(
                                    up_grad, tp_group, partition_dim
                                )
                                expert_grad_orth = torch.cat(
                                    [gate_grad_orth, up_grad_orth], dim=1
                                )
                            else:
                                expert_grad_orth = self.scaled_orthogonalize_fn(
                                    expert_grad, tp_group, partition_dim
                                )
                            expert_grads.append(expert_grad_orth)
                        grad = torch.stack(expert_grads, dim=1).view(grad_shape)

                elif 'weight2' in param_name:
                    # weight2: [num_experts * ffn_hidden_size_per_partition, hidden_size]
                    # Note: in TP case, the first dimension is already partitioned
                    ffn_dim_per_expert = grad_shape[0] // num_local_experts

                    # Reshape to separate experts: [num_experts, ffn_per_expert, hidden_size]
                    grad_reshaped = grad.view(num_local_experts, ffn_dim_per_expert, grad_shape[1])

                    if batch_local_experts:
                        grad = self.scaled_orthogonalize_fn(
                            grad_reshaped, tp_group, partition_dim
                        ).view(grad_shape)
                    else:
                        expert_grads = []
                        for expert_idx in range(num_local_experts):
                            expert_grad = grad_reshaped[expert_idx, :, :]
                            expert_grad_orth = self.scaled_orthogonalize_fn(
                                expert_grad, tp_group, partition_dim
                            )
                            expert_grads.append(expert_grad_orth)
                        grad = torch.stack(expert_grads, dim=0).view(grad_shape)
                else:
                    # Unknown parameter name, fall back to default
                    grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        elif self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            # split grouped attention parameters (e.g., QKV, GQA, etc.)
            assert self.qkv_split_shapes is not None, (
                "qkv_split_shapes must be provided for QKV splitting"
            )
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {self.qkv_split_shapes}',
            )
            if self.qkv_split_mode == "orca":
                return _map_orca_qkv_matrices(
                    grad,
                    tuple(self.qkv_split_shapes),
                    lambda matrix: self.scaled_orthogonalize_fn(
                        matrix, tp_group, None
                    ),
                )

            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            grad_view = grad.view(num_query_groups, sum(self.qkv_split_shapes), -1)

            q_dim_per_group, kv_channels, _ = self.qkv_split_shapes  # v_dim not used (same as k_dim/kv_channels)
            heads_per_group = q_dim_per_group // kv_channels

            if self.qkv_split_mode == "group":
                # Group mode: process each query group independently, with Q/K/V split within each group
                # This aligns with split_qkv_init which initializes each query group independently
                group_updates = []
                for g in range(num_query_groups):
                    # Split this group into Q/K/V
                    qkv_comps = torch.split(grad_view[g], list(self.qkv_split_shapes), dim=0)
                    # Apply Newton-Schulz to each component
                    comp_updates = [
                        self.scaled_orthogonalize_fn(comp, tp_group, partition_dim)
                        for comp in qkv_comps
                    ]
                    # Concatenate Q/K/V updates within this group
                    group_updates.append(torch.cat(comp_updates, dim=0))
                # Stack all groups and reshape
                grad = torch.stack(group_updates, dim=0).view(grad_shape)
            elif self.qkv_split_mode == "head":
                # Head mode: process each attention head independently for Q/K/V
                group_updates = []
                for g in range(num_query_groups):
                    # Split this group into Q/K/V
                    qkv_comps = torch.split(grad_view[g], list(self.qkv_split_shapes), dim=0)
                    q_grad, k_grad, v_grad = qkv_comps

                    # Q: split into individual heads and process each
                    # q_grad shape: [heads_per_group * kv_channels, hidden_dim]
                    q_grad_heads = q_grad.view(heads_per_group, kv_channels, -1)

                    q_head_updates = []
                    for h in range(heads_per_group):
                        uh = self.scaled_orthogonalize_fn(q_grad_heads[h], tp_group, partition_dim)
                        q_head_updates.append(uh)

                    # Merge Q head updates
                    q_grad_updated = torch.stack(q_head_updates, dim=0).reshape(q_dim_per_group, -1)

                    # K and V: process directly
                    k_grad_updated = self.scaled_orthogonalize_fn(k_grad, tp_group, partition_dim)
                    v_grad_updated = self.scaled_orthogonalize_fn(v_grad, tp_group, partition_dim)

                    # Concatenate Q/K/V updates within this group
                    group_updates.append(torch.cat([q_grad_updated, k_grad_updated, v_grad_updated], dim=0))

                # Stack all groups and reshape
                grad = torch.stack(group_updates, dim=0).view(grad_shape)
            else:  # component mode (original logic)
                # Component mode: merge all groups' Q together, all K together, all V together
                qkv_grads = torch.split(grad_view, list(self.qkv_split_shapes), dim=1)
                qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

                # Apply Newton-Schulz and scales to each component, concat back
                qkv_grads = [
                    self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                        num_query_groups, -1, grad_shape[-1]
                    )
                    for g in qkv_grads
                ]
                grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        elif self.split_fc1 and self.is_fc1_fn is not None and self.is_fc1_fn(p):
            # Split FC1 (gate and up) for gated linear units (SwiGLU)
            grad_shape = grad.shape
            gate_dim, up_dim = _resolve_fc1_split_shapes(p, grad, self.fc1_split_shapes)
            log_single_rank(
                logger,
                logging.DEBUG,
                f'fc1 split grad shape {grad_shape}, split shapes {(gate_dim, up_dim)}',
            )

            # Split gate and up along dim=0
            gate_grad, up_grad = torch.split(grad, [gate_dim, up_dim], dim=0)

            # Apply Newton-Schulz to each component
            gate_grad = self.scaled_orthogonalize_fn(gate_grad, tp_group, partition_dim)
            up_grad = self.scaled_orthogonalize_fn(up_grad, tp_group, partition_dim)

            # Concatenate back
            grad = torch.cat([gate_grad, up_grad], dim=0)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class ORCA(TensorParallelMuon):
    """Muon with scheduled orthogonal weight decay during the initial phase.

    Q and K are handled head-wise, V is handled as one matrix, SwiGLU is
    split into gate/up, and legacy GroupedMLP weights are batched by expert.
    """

    metadata_key = "orca"

    def __init__(
        self,
        *args: Any,
        total_steps: int,
        alpha1: float = 0.25,
        owdwd: float = 1.0,
        c: float = 0.1,
        wd_schedule: str = "stable_linear",
        wd_stable_ratio: float = 1.0,
        wd_end_scale: float = 0.0,
        router_owd: bool = True,
        shared_expert_gate_owd: bool = True,
        initial_step: int = 0,
        **kwargs: Any,
    ) -> None:
        if total_steps is None:
            raise ValueError("ORCA requires total_steps to be set.")
        super().__init__(*args, **kwargs)
        self.total_steps = int(total_steps)
        self.alpha1 = float(alpha1)
        self.owdwd = float(owdwd)
        self.c = float(c)
        self.wd_schedule = str(wd_schedule)
        self.wd_stable_ratio = float(wd_stable_ratio)
        self.wd_end_scale = float(wd_end_scale)
        self.router_owd = bool(router_owd)
        self.shared_expert_gate_owd = bool(shared_expert_gate_owd)
        self.period_step = int(initial_step)
        self.special_steps = int(self.total_steps * self.alpha1)
        self.last_phase = None
        self._validate_orca_hparams()

    @property
    def c_square(self) -> float:
        """Return the OWD target coefficient derived from ``c``."""
        return self.c * self.c

    def _validate_orca_hparams(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("ORCA requires total_steps to be positive.")
        if not math.isfinite(self.alpha1) or not 0.0 <= self.alpha1 <= 1.0:
            raise ValueError(f"ORCA alpha1 must be in [0, 1], got {self.alpha1}.")
        if not math.isfinite(self.owdwd) or self.owdwd < 0.0:
            raise ValueError(f"ORCA owdwd must be non-negative, got {self.owdwd}.")
        if not math.isfinite(self.c) or self.c < 0.0:
            raise ValueError(
                "ORCA c must be finite and non-negative, "
                f"got {self.c}."
            )
        try:
            c_square = self.c_square
        except OverflowError as exc:
            raise ValueError(f"ORCA c^2 must be finite, got {self.c}.") from exc
        if not math.isfinite(c_square):
            raise ValueError(f"ORCA c^2 must be finite, got {self.c_square}.")
        if self.wd_schedule != "stable_linear":
            raise ValueError(
                f"ORCA wd_schedule must be 'stable_linear', got {self.wd_schedule!r}."
            )
        if (
            not math.isfinite(self.wd_stable_ratio)
            or not 0.0 <= self.wd_stable_ratio <= 1.0
        ):
            raise ValueError(
                "ORCA wd_stable_ratio must be in [0, 1], "
                f"got {self.wd_stable_ratio}."
            )
        if not math.isfinite(self.wd_end_scale) or self.wd_end_scale < 0.0:
            raise ValueError(
                "ORCA wd_end_scale must be finite and non-negative, "
                f"got {self.wd_end_scale}."
            )
        if self.period_step < 0:
            raise ValueError(
                f"ORCA initial_step must be non-negative, got {self.period_step}."
            )

    def _use_owd_phase(self) -> bool:
        return self.period_step < self.special_steps

    def current_phase(self) -> str:
        return "orca_owd_muon" if self._use_owd_phase() else "orca_muon"

    @staticmethod
    def _owd_schedule_scale_for_steps(
        period_step: int,
        special_steps: int,
        wd_stable_ratio: float,
        wd_end_scale: float,
    ) -> float:
        if special_steps <= 0:
            return 0.0
        progress = min(
            max(float(period_step) / float(max(special_steps - 1, 1)), 0.0),
            1.0,
        )
        if progress <= wd_stable_ratio or wd_stable_ratio >= 1.0:
            return 1.0
        decay_progress = (progress - wd_stable_ratio) / (1.0 - wd_stable_ratio)
        return 1.0 + decay_progress * (wd_end_scale - 1.0)

    def _owd_schedule_scale(self) -> float:
        return self._owd_schedule_scale_for_steps(
            self.period_step,
            self.special_steps,
            self.wd_stable_ratio,
            self.wd_end_scale,
        )

    def _owd_phase_for_parameter(self, p: torch.Tensor) -> bool:
        """Return whether OWD is active for this parameter at this step."""
        return self._use_owd_phase()

    def _owd_coefficient_for_parameter(self, p: torch.Tensor, lr: float) -> float:
        """Return the OWD coefficient for one parameter."""
        return self._owd_coefficient(lr)

    def _orca_logical_matrix_tasks(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup],
        partition_dim: Optional[int],
    ) -> Optional[list[MuonLogicalMatrixTask]]:
        """Build tasks using ORCA's Q/K/V and expert boundaries.

        ORCA's QKV mapping is intentionally different from regular Muon:
        Q is split by attention head, K by query group, and V is one matrix for
        all query groups. Each returned task remains an independent logical
        matrix; the bucket runner may combine tasks from other layers/params.
        """
        if (
            self.split_moe_experts
            and self.is_grouped_moe_fn is not None
            and self.is_grouped_moe_fn(p)
        ):
            num_experts = int(getattr(p, "num_local_experts", 0))
            name = getattr(p, "param_name", "")
            if num_experts <= 0:
                raise ValueError(f"Missing num_local_experts for GroupedMLP parameter {name!r}.")
            tasks: list[MuonLogicalMatrixTask] = []
            if "weight1" in name:
                hidden, fused_ffn = update.shape
                components = 2 if getattr(p, "is_gated", False) and self.split_fc1 else 1
                divisor = num_experts * components
                if fused_ffn % divisor != 0:
                    raise ValueError(
                        f"Cannot split GroupedMLP weight1 {name!r} with shape "
                        f"{tuple(update.shape)} into {num_experts} experts and "
                        f"{components} components."
                    )
                ffn = fused_ffn // divisor
                grouped = update.view(hidden, num_experts, components, ffn)
                for expert_index in range(num_experts):
                    for component_index in range(components):
                        matrix_view = grouped[:, expert_index, component_index, :]
                        tasks.append(
                            self._task_for_matrix_view(
                                update,
                                matrix_view,
                                role="orca_moe_weight1",
                                qkv_mode="orca",
                                tp_group=tp_group,
                                partition_dim=partition_dim,
                            )
                        )
                return tasks
            if "weight2" in name:
                fused_ffn, hidden = update.shape
                if fused_ffn % num_experts != 0:
                    raise ValueError(
                        f"Cannot split GroupedMLP weight2 {name!r} with shape "
                        f"{tuple(update.shape)} into {num_experts} experts."
                    )
                grouped = update.view(num_experts, fused_ffn // num_experts, hidden)
                for expert_index in range(num_experts):
                    tasks.append(
                        self._task_for_matrix_view(
                            update,
                            grouped[expert_index],
                            role="orca_moe_weight2",
                            qkv_mode="orca",
                            tp_group=tp_group,
                            partition_dim=partition_dim,
                        )
                    )
                return tasks
            raise ValueError(f"Unrecognized GroupedMLP parameter {name!r}.")

        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            if self.qkv_split_shapes is None:
                raise ValueError("ORCA requires qkv_split_shapes for fused QKV weights.")
            q_per_group, k_per_group, v_per_group = tuple(
                int(size) for size in self.qkv_split_shapes
            )
            rows_per_group = q_per_group + k_per_group + v_per_group
            if (
                min(q_per_group, k_per_group, v_per_group) <= 0
                or update.dim() != 2
                or update.size(0) % rows_per_group != 0
                or q_per_group % k_per_group != 0
            ):
                raise ValueError(
                    f"Cannot apply ORCA QKV split to shape {tuple(update.shape)} "
                    f"with split shapes {tuple(self.qkv_split_shapes)}."
                )
            groups = update.size(0) // rows_per_group
            hidden = update.size(1)
            grouped = update.view(groups, rows_per_group, hidden)
            q, k, v = torch.split(
                grouped, [q_per_group, k_per_group, v_per_group], dim=1
            )
            heads_per_group = q_per_group // k_per_group
            tasks = []
            for group_index in range(groups):
                for head_index in range(heads_per_group):
                    matrix_view = q[
                        group_index,
                        head_index * k_per_group : (head_index + 1) * k_per_group,
                        :,
                    ]
                    tasks.append(
                        self._task_for_matrix_view(
                            update,
                            matrix_view,
                            role="orca_qk",
                            qkv_mode="orca",
                            tp_group=tp_group,
                            partition_dim=None,
                        )
                    )
                tasks.append(
                    self._task_for_matrix_view(
                        update,
                        k[group_index],
                        role="orca_qk",
                        qkv_mode="orca",
                        tp_group=tp_group,
                        partition_dim=None,
                    )
                )
            v_shape = (groups * v_per_group, hidden)
            tasks.append(
                self._make_logical_task(
                    update,
                    shape=v_shape,
                    role="orca_v",
                    qkv_mode="orca",
                    tp_group=tp_group,
                    partition_dim=None,
                    read=lambda view=v, shape=v_shape: view.reshape(shape),
                    write=lambda value, view=v: self._copy_task_result(
                        view, value.reshape_as(view)
                    ),
                )
            )
            return tasks

        if self.split_fc1 and self.is_fc1_fn is not None and self.is_fc1_fn(p):
            gate_dim, up_dim = _resolve_fc1_split_shapes(p, update, self.fc1_split_shapes)
            if gate_dim != up_dim:
                raise ValueError(
                    f"ORCA requires equal SwiGLU gate/up sizes, got "
                    f"{(gate_dim, up_dim)} for {getattr(p, 'param_name', '<unnamed>')!r}."
                )
            gate, up = torch.split(update, [gate_dim, up_dim], dim=0)
            return [
                self._task_for_matrix_view(
                    update,
                    gate,
                    role="orca_fc1_gate_up",
                    qkv_mode="orca",
                    tp_group=tp_group,
                    partition_dim=partition_dim,
                ),
                self._task_for_matrix_view(
                    update,
                    up,
                    role="orca_fc1_gate_up",
                    qkv_mode="orca",
                    tp_group=tp_group,
                    partition_dim=partition_dim,
                ),
            ]

        # TE experts are already one parameter per expert. Reuse the generic
        # task classification for these and ordinary 2D matrices. It is safe
        # here because the ORCA-specific fused layouts were handled above.
        return super()._logical_matrix_tasks(p, update)

    def _logical_matrix_tasks(
        self, p: torch.Tensor, update: torch.Tensor
    ) -> Optional[list[MuonLogicalMatrixTask]]:
        tp_group, partition_dim = self._tp_metadata(p)
        if get_pg_size(tp_group) != 1 or partition_dim is not None:
            return None
        return self._orca_logical_matrix_tasks(p, update, tp_group, partition_dim)

    def _map_logical_matrices(
        self,
        p: torch.Tensor,
        tensor: torch.Tensor,
        matrix_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply a matrix operation using ORCA's logical parameter splits."""
        if (
            self.split_moe_experts
            and self.is_grouped_moe_fn is not None
            and self.is_grouped_moe_fn(p)
        ):
            num_experts = int(getattr(p, "num_local_experts", 0))
            name = getattr(p, "param_name", "")
            if num_experts <= 0:
                raise ValueError(f"Missing num_local_experts for GroupedMLP parameter {name!r}.")
            if "weight1" in name:
                hidden, fused_ffn = tensor.shape
                components = 2 if getattr(p, "is_gated", False) and self.split_fc1 else 1
                divisor = num_experts * components
                if fused_ffn % divisor != 0:
                    raise ValueError(
                        f"Cannot split GroupedMLP weight1 {name!r} with shape "
                        f"{tuple(tensor.shape)} into {num_experts} experts and "
                        f"{components} components."
                    )
                ffn = fused_ffn // divisor
                matrices = (
                    tensor.view(hidden, num_experts, components, ffn)
                    .permute(1, 2, 0, 3)
                    .reshape(num_experts * components, hidden, ffn)
                )
                mapped = matrix_fn(matrices)
                return (
                    mapped.view(num_experts, components, hidden, ffn)
                    .permute(2, 0, 1, 3)
                    .reshape_as(tensor)
                )
            if "weight2" in name:
                fused_ffn, hidden = tensor.shape
                if fused_ffn % num_experts != 0:
                    raise ValueError(
                        f"Cannot split GroupedMLP weight2 {name!r} with shape "
                        f"{tuple(tensor.shape)} into {num_experts} experts."
                    )
                matrices = tensor.view(num_experts, fused_ffn // num_experts, hidden)
                return matrix_fn(matrices).reshape_as(tensor)
            raise ValueError(f"Unrecognized GroupedMLP parameter {name!r}.")

        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            if self.qkv_split_shapes is None:
                raise ValueError("ORCA requires qkv_split_shapes for fused QKV weights.")
            return _map_orca_qkv_matrices(
                tensor, tuple(self.qkv_split_shapes), matrix_fn
            )

        if self.split_fc1 and self.is_fc1_fn is not None and self.is_fc1_fn(p):
            gate_dim, up_dim = _resolve_fc1_split_shapes(p, tensor, self.fc1_split_shapes)
            if gate_dim != up_dim:
                raise ValueError(
                    f"ORCA requires equal SwiGLU gate/up sizes, got "
                    f"{(gate_dim, up_dim)} for {getattr(p, 'param_name', '<unnamed>')!r}."
                )
            gate, up = torch.split(tensor, [gate_dim, up_dim], dim=0)
            return matrix_fn(torch.stack((gate, up))).reshape_as(tensor)

        return matrix_fn(tensor)

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, "expert_tp", False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None

        def orthogonalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
            return self.scaled_orthogonalize_fn(matrix, tp_group, None)

        return self._map_logical_matrices(p, grad, orthogonalize_matrix)

    def _owd_coefficient(self, lr: float) -> float:
        """Return the OWD coefficient for the current ORCA step."""
        return lr * self.owdwd * self._owd_schedule_scale()

    def _owd_enabled_for_parameter(self, p: torch.Tensor) -> bool:
        """Whether OWD applies to this parameter in the current phase."""
        if self.owdwd == 0.0:
            return False
        if getattr(p, "is_router", False) and not self.router_owd:
            return False
        if (
            getattr(p, "is_shared_expert_gate", False)
            and not self.shared_expert_gate_owd
        ):
            return False
        return self._owd_coefficient_for_parameter(p, 1.0) != 0.0

    def _owd_logical_matrix_tasks(
        self, p: torch.Tensor
    ) -> Optional[list[MuonLogicalMatrixTask]]:
        """Return logical matrices that should receive OWD for one parameter."""
        return self._logical_matrix_tasks(p, p)

    def _map_owd_matrices(
        self, p: torch.Tensor, tensor: torch.Tensor, coefficient: float
    ) -> torch.Tensor:
        """Apply the OWD update to all logical matrices of one parameter."""
        return self._map_logical_matrices(
            p,
            tensor,
            lambda matrix: self._owd_updated_matrix(matrix, coefficient),
        )

    def _owd_updated_matrix(self, matrix: torch.Tensor, coefficient: float) -> torch.Tensor:
        """Return one OWD-updated matrix, or a batch of independent matrices."""
        compute_matrix = (
            matrix.to(torch.bfloat16)
            if matrix.is_cuda and self.fp32_matmul_prec == "medium"
            else matrix
        )
        direction = orthogonal_weight_decay_direction(
            compute_matrix,
            use_syrk=self.use_triton_optimize,
            alpha=self.c_square,
        ).to(matrix.dtype)
        return torch.add(matrix, direction, alpha=-coefficient)

    def _apply_orthogonal_weight_decay(self, p: torch.Tensor, lr: float) -> None:
        if not self._owd_enabled_for_parameter(p):
            return
        coefficient = self._owd_coefficient_for_parameter(p, lr)
        if coefficient == 0.0:
            return

        updated = self._map_owd_matrices(p, p, coefficient)
        p.copy_(updated)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Batch ORCA OWD, then batch the regular Muon updates.

        OWD is evaluated after ordinary decoupled weight decay but before
        momentum, exactly as in the original ORCA order. OWD tasks are
        independent across parameters, so compatible local logical matrices can
        share the same bounded batch runner. TP-partitioned or otherwise
        incompatible parameters retain the per-parameter fallback.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.log_per_module_update_rms:
            self.per_module_update_rms.clear()
        if self.log_per_module_grad_rms:
            self.per_module_grad_rms.clear()

        self.last_phase = self.current_phase()
        pending_params: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []
        owd_tasks_by_coefficient: dict[float, list[MuonLogicalMatrixTask]] = defaultdict(list)
        owd_fallback: list[tuple[torch.Tensor, float]] = []

        # Apply ordinary weight decay first and collect the parameters whose
        # OWD can be run in a compatible logical-matrix batch.
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() == 1:
                    raise ValueError("ORCA does not support 1D parameters")
                grad = p.grad
                if grad is None:
                    continue

                effective_weight_decay = group.get("wd_mult", 1.0) * group["weight_decay"]
                self._apply_weight_decay_inplace(
                    p, grad, group["lr"], effective_weight_decay
                )
                pending_params.append(
                    (p, grad, {k: v for k, v in group.items() if k != "params"})
                )

                if not self._owd_phase_for_parameter(p) or not self._owd_enabled_for_parameter(p):
                    continue
                coefficient = self._owd_coefficient_for_parameter(p, group["lr"])
                if coefficient == 0.0:
                    continue
                tasks = self._owd_logical_matrix_tasks(p)
                if tasks is None:
                    owd_fallback.append((p, group["lr"]))
                else:
                    owd_tasks_by_coefficient[coefficient].extend(tasks)

        with emerging_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for coefficient, tasks in owd_tasks_by_coefficient.items():
                self._run_logical_matrix_tasks(
                    tasks,
                    operation=lambda matrix, _tp_group, _partition_dim, coefficient=coefficient: self._owd_updated_matrix(
                        matrix, coefficient
                    ),
                )
            for p, lr in owd_fallback:
                self._apply_orthogonal_weight_decay(p, lr)

        pending_updates: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []
        logical_tasks: list[MuonLogicalMatrixTask] = []
        fallback_updates: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []

        # Momentum and Muon's post-momentum orthogonalization retain their
        # existing ordering and batching behavior.
        for p, grad, group_kwargs in pending_params:
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            exp_avg = state["momentum_buffer"]

            exp_avg.lerp_(grad, 1 - group_kwargs["momentum_beta"])
            if self.use_nesterov:
                grad.lerp_(exp_avg, group_kwargs["momentum_beta"])
            else:
                grad.copy_(exp_avg)
            update = grad

            param_name = getattr(p, "param_name", None)
            if self.log_per_module_grad_rms and param_name:
                self.per_module_grad_rms[param_name] = torch.sqrt(
                    torch.mean(update**2)
                ).item()

            pending_updates.append((p, update, group_kwargs))
            tasks = self._logical_matrix_tasks(p, update)
            if tasks is None:
                fallback_updates.append((p, update, group_kwargs))
            else:
                logical_tasks.extend(tasks)

        with emerging_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            self._run_logical_matrix_tasks(logical_tasks)
            for p, update, group_kwargs in fallback_updates:
                update.copy_(self.orthogonalize(p, update, **group_kwargs))

        for p, update, group_kwargs in pending_updates:
            param_name = getattr(p, "param_name", None)
            if self.log_per_module_update_rms and param_name:
                self.per_module_update_rms[param_name] = torch.sqrt(
                    torch.mean(update**2)
                ).item()
            p.add_(update, alpha=-group_kwargs["lr"])

        self.period_step += 1
        return loss

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict[self.metadata_key] = {
            "total_steps": self.total_steps,
            "alpha1": self.alpha1,
            "special_steps": self.special_steps,
            "period_step": self.period_step,
            "last_phase": self.last_phase,
            "owdwd": self.owdwd,
            "c": self.c,
            "c_square": self.c_square,
            "wd_schedule": self.wd_schedule,
            "wd_stable_ratio": self.wd_stable_ratio,
            "wd_end_scale": self.wd_end_scale,
            "router_owd": self.router_owd,
            "shared_expert_gate_owd": self.shared_expert_gate_owd,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        metadata = state_dict.pop(self.metadata_key, None)
        loaded_legacy_metadata = False
        if metadata is None:
            # A checkpoint from either ORCA variant can reuse its momentum
            # state, including state saved before the public rename.
            alternate_keys = [
                key for key in ORCA_OPTIMIZERS if key != self.metadata_key
            ] + list(_LEGACY_ORCA_METADATA_KEYS)
            for metadata_key in alternate_keys:
                loaded_legacy_metadata = metadata_key in _LEGACY_ORCA_METADATA_KEYS
                if metadata_key == self.metadata_key:
                    continue
                metadata = state_dict.pop(metadata_key, None)
                if metadata is not None:
                    break
        super().load_state_dict(state_dict)
        if metadata is None:
            return
        self.total_steps = int(metadata.get("total_steps", self.total_steps))
        self.alpha1 = float(metadata.get("alpha1", self.alpha1))
        self.special_steps = int(
            metadata.get("special_steps", int(self.total_steps * self.alpha1))
        )
        self.period_step = int(metadata.get("period_step", self.period_step))
        self.last_phase = None if loaded_legacy_metadata else metadata.get(
            "last_phase", self.last_phase
        )
        self.owdwd = float(metadata.get("owdwd", self.owdwd))
        if "c" in metadata:
            self.c = float(metadata["c"])
        elif "c_square" in metadata:
            self.c = math.sqrt(float(metadata["c_square"]))
        elif "owd_alpha" in metadata:
            # Checkpoint compatibility: legacy metadata stored the target
            # coefficient directly, so recover its non-negative square root.
            self.c = math.sqrt(float(metadata["owd_alpha"]))
        self.wd_schedule = str(metadata.get("wd_schedule", self.wd_schedule))
        self.wd_stable_ratio = float(
            metadata.get("wd_stable_ratio", self.wd_stable_ratio)
        )
        self.wd_end_scale = float(metadata.get("wd_end_scale", self.wd_end_scale))
        self.router_owd = bool(metadata.get("router_owd", self.router_owd))
        self.shared_expert_gate_owd = bool(
            metadata.get("shared_expert_gate_owd", self.shared_expert_gate_owd)
        )
        self._validate_orca_hparams()


class ORCALayerwise(ORCA):
    """ORCA Muon with a linearly increasing, layer-specific OWD phase.

    Parameters without a ``orca_layer_number`` marker retain the configured
    scalar ``alpha1`` schedule. Transformer-layer parameters use the global
    1-indexed layer number and are interpolated between ``alpha_start`` and
    ``alpha_end``.
    """

    metadata_key = "orca_layerwise"

    def __init__(
        self,
        *args: Any,
        alpha_start: float = 0.15,
        alpha_end: float = 0.40,
        num_layers: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.num_layers = int(num_layers) if num_layers is not None else None
        self._validate_layerwise_hparams()

    def _validate_layerwise_hparams(self) -> None:
        if not math.isfinite(self.alpha_start) or not 0.0 <= self.alpha_start <= 1.0:
            raise ValueError(
                f"ORCALayerwise alpha_start must be in [0, 1], got {self.alpha_start}."
            )
        if not math.isfinite(self.alpha_end) or not 0.0 <= self.alpha_end <= 1.0:
            raise ValueError(
                f"ORCALayerwise alpha_end must be in [0, 1], got {self.alpha_end}."
            )
        if self.alpha_start > self.alpha_end:
            raise ValueError(
                "ORCALayerwise alpha_start must not exceed alpha_end."
            )
        if self.num_layers is None:
            raise ValueError(
                "ORCALayerwise requires num_layers to be set."
            )
        if self.num_layers <= 0:
            raise ValueError(
                f"ORCALayerwise num_layers must be positive, got {self.num_layers}."
            )

    def _layer_alpha(self, p: torch.Tensor) -> float:
        layer_number = getattr(p, "orca_layer_number", None)
        if layer_number is None:
            return self.alpha1
        layer_number = int(layer_number)
        if self.num_layers <= 1:
            return self.alpha_start
        layer_index = min(max(layer_number - 1, 0), self.num_layers - 1)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * (
            float(layer_index) / float(self.num_layers - 1)
        )

    def _parameter_special_steps(self, p: torch.Tensor) -> int:
        return int(self.total_steps * self._layer_alpha(p))

    def _owd_phase_for_parameter(self, p: torch.Tensor) -> bool:
        return self.period_step < self._parameter_special_steps(p)

    def _owd_coefficient_for_parameter(self, p: torch.Tensor, lr: float) -> float:
        special_steps = self._parameter_special_steps(p)
        scale = self._owd_schedule_scale_for_steps(
            self.period_step,
            special_steps,
            self.wd_stable_ratio,
            self.wd_end_scale,
        )
        return lr * self.owdwd * scale

    def current_phase(self) -> str:
        # Use the latest layer (or an unmarked fallback parameter)'s boundary
        # for the optimizer-level phase label.
        max_special_steps = max(int(self.total_steps * self.alpha_end), self.special_steps)
        return (
            "orca_layerwise_owd_muon"
            if self.period_step < max_special_steps
            else "orca_muon"
        )

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict[self.metadata_key].update(
            {
                "alpha_start": self.alpha_start,
                "alpha_end": self.alpha_end,
                "num_layers": self.num_layers,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = dict(state_dict)
        metadata = state_dict.get(self.metadata_key)
        if metadata is None:
            metadata = state_dict.get("period6muon_layerwise")
        super().load_state_dict(state_dict)
        if metadata is not None:
            self.alpha_start = float(metadata.get("alpha_start", self.alpha_start))
            self.alpha_end = float(metadata.get("alpha_end", self.alpha_end))
            stored_num_layers = metadata.get("num_layers", self.num_layers)
            self.num_layers = (
                int(stored_num_layers) if stored_num_layers is not None else None
            )
            self._validate_layerwise_hparams()


def set_orca_step(optimizer: MegatronOptimizer, step: int) -> None:
    """Synchronize ORCA's phase when a checkpoint is loaded without optimizer state."""
    step = int(step)
    if step < 0:
        raise ValueError(f"ORCA step must be non-negative, got {step}.")

    pending = [optimizer]
    visited = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, (ORCA, ORCAAdamW)):
            current.period_step = step
            current.last_phase = None
            continue
        if hasattr(current, "chained_optimizers"):
            pending.extend(current.chained_optimizers)
        else:
            inner_optimizer = getattr(current, "__dict__", {}).get("optimizer")
            if inner_optimizer is not None:
                pending.append(inner_optimizer)


def override_orca_schedule(
    optimizer: MegatronOptimizer,
    *,
    total_steps: int,
    alpha1: float,
    owdwd: float,
    wd_schedule: str,
    wd_stable_ratio: float,
    wd_end_scale: float,
    router_owd: bool,
    shared_expert_gate_owd: bool,
    layerwise_alpha_start: Optional[float] = None,
    layerwise_alpha_end: Optional[float] = None,
    num_layers: Optional[int] = None,
    c: Optional[float] = None,
) -> None:
    """Override ORCA schedule metadata after loading a full optimizer state.

    A normal optimizer restore intentionally takes these values from the
    checkpoint.  Resume branches that change only the ORCA schedule can use
    this helper after loading to retain momentum and scheduler state while
    applying the new schedule from the command line.
    """
    total_steps = int(total_steps)
    alpha1 = float(alpha1)
    owdwd = float(owdwd)
    if c is not None:
        c = float(c)
    wd_schedule = str(wd_schedule)
    wd_stable_ratio = float(wd_stable_ratio)
    wd_end_scale = float(wd_end_scale)
    router_owd = bool(router_owd)
    shared_expert_gate_owd = bool(shared_expert_gate_owd)
    if layerwise_alpha_start is not None:
        layerwise_alpha_start = float(layerwise_alpha_start)
    if layerwise_alpha_end is not None:
        layerwise_alpha_end = float(layerwise_alpha_end)
    if num_layers is not None:
        num_layers = int(num_layers)

    pending = [optimizer]
    visited = set()
    found = False
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, (ORCA, ORCAAdamW)):
            current.total_steps = total_steps
            current.alpha1 = alpha1
            current.special_steps = int(total_steps * alpha1)
            current.owdwd = owdwd
            if c is not None:
                current.c = c
            current.wd_schedule = wd_schedule
            current.wd_stable_ratio = wd_stable_ratio
            current.wd_end_scale = wd_end_scale
            current.router_owd = router_owd
            current.shared_expert_gate_owd = shared_expert_gate_owd
            if isinstance(current, ORCALayerwise):
                if layerwise_alpha_start is not None:
                    current.alpha_start = layerwise_alpha_start
                if layerwise_alpha_end is not None:
                    current.alpha_end = layerwise_alpha_end
                if num_layers is not None:
                    current.num_layers = num_layers
            elif isinstance(current, ORCAAdamW) and current.layerwise:
                if layerwise_alpha_start is not None:
                    current.layerwise_alpha_start = layerwise_alpha_start
                if layerwise_alpha_end is not None:
                    current.layerwise_alpha_end = layerwise_alpha_end
                if num_layers is not None:
                    current.num_layers = num_layers
            current._validate_orca_hparams()
            if isinstance(current, ORCALayerwise):
                current._validate_layerwise_hparams()
            found = True
            continue
        if hasattr(current, "chained_optimizers"):
            pending.extend(current.chained_optimizers)
        else:
            inner_optimizer = getattr(current, "__dict__", {}).get("optimizer")
            if inner_optimizer is not None:
                pending.append(inner_optimizer)

    if not found:
        raise RuntimeError(
            "ORCA schedule override requested, but no ORCA optimizer was found."
        )


def _use_muon_for_parameter(
    param: torch.Tensor, config: OptimizerConfig, use_orca: bool
) -> bool:
    """Choose Muon or the AdamW fallback for one model parameter."""
    use_muon = not getattr(param, 'is_embedding_or_output_parameter', False) and param.dim() != 1
    if not use_orca:
        if getattr(param, 'is_router', False) and config.muon_router_adamw:
            return False
        if (
            getattr(param, 'is_shared_expert_gate', False)
            and config.muon_shared_expert_gate_adamw
        ):
            return False
        return use_muon
    if getattr(param, 'is_router', False):
        return not config.muon_router_adamw and config.orca_router_muon
    if getattr(param, 'is_shared_expert_gate', False):
        return (
            not config.muon_shared_expert_gate_adamw
            and config.orca_shared_expert_gate_muon
        )
    return use_muon


def _orca_special_parameter_strategy(
    use_muon: bool, use_owd: bool, owdwd: float
) -> str:
    if use_owd and owdwd > 0.0:
        return "muon+orca" if use_muon else "adamw+orca"
    if not use_muon:
        return "adamw"
    return "muon"


def _get_transformer_block(model_chunk: MegatronModule) -> Optional[Any]:
    """Find a decoder/encoder block below Megatron's wrappers."""
    current: Any = model_chunk
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        for block_name in ("decoder", "encoder"):
            block = getattr(current, block_name, None)
            if block is not None and hasattr(block, "layers"):
                return block
        current = getattr(current, "module", None)
    return None


def _is_nonshared_expert_name(name: str) -> bool:
    """Identify routed expert parameters without matching shared experts."""
    return "experts" in name and "shared" not in name


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers is not installed."
    use_orca = is_orca_optimizer(config.optimizer)
    optimizer_name = config.optimizer
    use_orca_layerwise = config.optimizer == "orca_layerwise"

    if use_orca and not layer_wise_distributed_optimizer:
        raise ValueError(
            "orca requires the LayerWiseDistributedOptimizer (dist Muon) path."
        )

    # dist-optim is not supported due to strong coupling with how DDP init grad buffer
    # in thoery we can put some weight to use non-dist-muon and rest to dist-adam
    # but there are strong dependency and assumption in DDP that prevent it
    if config.use_distributed_optimizer:
        raise Exception('muon with dist optimizer is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

    if use_orca and (
        get_pg_size(pg_collection.tp) != 1 or get_pg_size(pg_collection.expt_tp) != 1
    ):
        raise ValueError("orca currently requires TP=1 and expert TP=1.")

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    special_param_counts = {'router': 0, 'shared_expert_gate': 0}

    qkv_split_shapes = None
    fc1_split_shapes = None
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        layer_param_numbers = {}
        transformer_block = _get_transformer_block(model_chunk)
        if transformer_block is not None:
            for layer in transformer_block.layers:
                layer_number = getattr(layer, 'layer_number', None)
                if layer_number is None:
                    if use_orca_layerwise:
                        raise ValueError(
                            "layerwise ORCA Muon requires every Transformer layer "
                            "to expose its global layer_number."
                        )
                    continue
                for layer_param in layer.parameters():
                    layer_param_numbers[id(layer_param)] = int(layer_number)
        # derive fc1 split shapes for gated linear units (SwiGLU)
        try:
            if model_chunk.config.gated_linear_unit:
                ffn_hidden_size = model_chunk.config.ffn_hidden_size
                fc1_split_shapes = [ffn_hidden_size, ffn_hidden_size]  # gate, up
        except Exception:
            pass
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # Store parameter name for logging
            param.param_name = name
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if _is_nonshared_expert_name(name):
                param.expert_tp = True
                param.is_nonshared_expert = True
            # add flag for qkv parameter
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            # add flag for fc1 parameter (gated linear units like SwiGLU)
            if (
                getattr(model_chunk.config, 'gated_linear_unit', False)
                and 'linear_fc1.weight' in name
                and len(param.shape) == 2
            ):
                if param.shape[0] % 2 != 0:
                    raise ValueError(
                        f"SwiGLU FC1 parameter {name!r} has odd output size {param.shape[0]}."
                    )
                param.is_fc1 = True
                local_ffn_size = param.shape[0] // 2
                param.fc1_split_shapes = (local_ffn_size, local_ffn_size)
            # TE GroupedLinear keeps one Parameter per local expert (weight0,
            # weight1, ...). Mark those parameters so the optimizer can bucket
            # same-shaped experts across local experts and layers without
            # changing TE's forward/backward parameter representation.
            if (
                'shared' not in name
                and (
                    'experts.linear_fc1.weight' in name
                    or 'experts.linear_fc2.weight' in name
                )
                and len(param.shape) == 2
            ):
                param.is_te_grouped_expert = True
            # add flag for GroupedMLP weight1/weight2 (MoE experts)
            if 'experts.weight1' in name or 'experts.weight2' in name:
                param.is_grouped_moe = True
                # Store MoE configuration for expert splitting
                try:
                    param.num_local_experts = (
                        model_chunk.config.num_moe_experts
                        // model_chunk.config.expert_model_parallel_size
                    )
                    param.moe_ffn_hidden_size = model_chunk.config.moe_ffn_hidden_size
                    param.is_gated = model_chunk.config.gated_linear_unit
                except Exception:
                    # If config not available, disable expert splitting for this param
                    param.is_grouped_moe = False
            if name.endswith('router.weight'):
                param.is_router = True
                special_param_counts['router'] += 1
            if name.endswith('shared_experts.gate_weight'):
                param.is_shared_expert_gate = True
                special_param_counts['shared_expert_gate'] += 1
            # Preserve the global Transformer layer number for layerwise ORCA.
            layer_number = layer_param_numbers.get(id(param))
            if layer_number is not None:
                param.orca_layer_number = layer_number
            # TODO(deyuf): might not be sufficient for future algorithm. revisit this conditioning
            if _use_muon_for_parameter(param, config, use_orca):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    if use_orca:
        if (
            use_orca_layerwise and config.orca_num_layers is None
        ):
            # The training setup normally supplies the global count through
            # OptimizerConfig. Direct factory callers may only provide the
            # model's TransformerConfig, so infer it before constructing both
            # the Muon and AdamW fallback optimizers.
            for model_chunk in model_chunks:
                transformer_block = _get_transformer_block(model_chunk)
                model_config = getattr(model_chunk, "config", None)
                inferred_num_layers = getattr(model_config, "num_layers", None)
                if inferred_num_layers is None and transformer_block is not None:
                    inferred_num_layers = getattr(
                        getattr(transformer_block, "config", None), "num_layers", None
                    )
                if inferred_num_layers is not None:
                    config.orca_num_layers = int(inferred_num_layers)
                    break
            if config.orca_num_layers is None:
                raise ValueError(
                    "layerwise ORCA Muon requires the global number of Transformer layers. "
                    "Set --num-layers or OptimizerConfig.orca_num_layers."
                )
        router_use_muon = not config.muon_router_adamw and config.orca_router_muon
        shared_gate_use_muon = (
            not config.muon_shared_expert_gate_adamw
            and config.orca_shared_expert_gate_muon
        )
        router_strategy = _orca_special_parameter_strategy(
            router_use_muon,
            config.orca_router_owd,
            config.orca_owdwd,
        )
        shared_gate_strategy = _orca_special_parameter_strategy(
            shared_gate_use_muon,
            config.orca_shared_expert_gate_owd,
            config.orca_owdwd,
        )
        log_single_rank(
            logger,
            logging.INFO,
            "ORCA special parameter strategies: "
            f"router={router_strategy} ({special_param_counts['router']} tensors), "
            f"shared_expert_gate={shared_gate_strategy} "
            f"({special_param_counts['shared_expert_gate']} tensors)",
        )
    elif config.muon_router_adamw or config.muon_shared_expert_gate_adamw:
        router_strategy = "adamw" if config.muon_router_adamw else "muon"
        shared_gate_strategy = (
            "adamw" if config.muon_shared_expert_gate_adamw else "muon"
        )
        log_single_rank(
            logger,
            logging.INFO,
            "Muon special parameter strategies: "
            f"router={router_strategy} ({special_param_counts['router']} tensors), "
            f"shared_expert_gate={shared_gate_strategy} "
            f"({special_param_counts['shared_expert_gate']} tensors)",
        )

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )

    muon_kwargs = dict(
        lr=config.lr,
        momentum_beta=config.muon_momentum,
        use_nesterov=config.muon_use_nesterov,
        weight_decay=config.weight_decay,
        fp32_matmul_prec=config.muon_fp32_matmul_prec,
        coefficient_type=config.muon_coefficient_type,
        num_ns_steps=config.muon_num_ns_steps,
        use_triton_optimize=config.use_triton_optimize,
        optimizer_max_batch_size=config.muon_optimizer_max_batch_size,
        scale_mode=config.muon_scale_mode,
        split_qkv=True if use_orca else config.muon_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        qkv_split_mode=config.muon_qkv_split_mode,
        split_fc1=True if use_orca else config.muon_split_fc1,
        is_fc1_fn=lambda p: getattr(p, 'is_fc1', False),
        fc1_split_shapes=tuple(fc1_split_shapes) if fc1_split_shapes is not None else None,
        split_moe_experts=True if use_orca else config.muon_split_moe_experts,
        is_grouped_moe_fn=lambda p: getattr(p, 'is_grouped_moe', False),
        extra_scale_factor=config.muon_extra_scale_factor,
        pg_collection=pg_collection,
        mode=config.muon_tp_mode,
    )
    if use_orca:
        orca_optimizer_cls = (
            ORCALayerwise if use_orca_layerwise else ORCA
        )
        optimizer_kwargs = dict(
            total_steps=config.orca_total_steps,
            alpha1=config.orca_alpha1,
            owdwd=config.orca_owdwd,
            c=config.orca_c,
            wd_schedule=config.orca_wd_schedule,
            wd_stable_ratio=config.orca_wd_stable_ratio,
            wd_end_scale=config.orca_wd_end_scale,
            router_owd=config.orca_router_owd,
            shared_expert_gate_owd=config.orca_shared_expert_gate_owd,
            **muon_kwargs,
        )
        if use_orca_layerwise:
            optimizer_kwargs.update(
                alpha_start=config.orca_layerwise_alpha_start,
                alpha_end=config.orca_layerwise_alpha_end,
                num_layers=config.orca_num_layers,
            )
        optimizer = orca_optimizer_cls(linear_param_groups, **optimizer_kwargs)
    else:
        optimizer = TensorParallelMuon(linear_param_groups, **muon_kwargs)

    # Enable per-module logging if configured
    optimizer.log_per_module_update_rms = config.log_per_module_update_rms
    if hasattr(optimizer, 'log_per_module_grad_rms'):
        optimizer.log_per_module_grad_rms = config.log_per_module_grad_rms

    # set config here to:
    # 1. get adam for rest of layer
    # 2. avoid ChainedOptimizer check fail that assert all optimizers are same kind
    # side effect is muon optimizer will have wrong name str, i.e. config.optimizer == 'adam'
    # TODO(deyuf): allow user to select optimizer mix and relax ChainedOptimizer design
    config.optimizer = 'adam'
    config.orca_layerwise = use_orca_layerwise
    config.orca_adam_owd = use_orca and (
        (
            (config.muon_router_adamw or not config.orca_router_muon)
            and config.orca_router_owd
        )
        or (
            (config.muon_shared_expert_gate_adamw or not config.orca_shared_expert_gate_muon)
            and config.orca_shared_expert_gate_owd
        )
    )

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    # TODO(boxiangw): Improve usability after optimizer refactor
    # TODO(boxiangw): support precision aware optimizer
    def muon_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    # need to wrap into megatron mix precision optimizer. (only support bf16 w/o loss scale now)
    if config.fp16:
        raise Exception('muon with fp16 is not supported.')

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, muon_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, muon_init_state_fn)

    optimizers.append(optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config, model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult, use_gloo_process_groups
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        optimizer_display_name = optimizer_name if use_orca else "Muon"
        log_single_rank(
            logger,
            logging.INFO,
            f'Using LayerWiseDistributedOptimizer for {optimizer_display_name}',
        )
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=[muon_init_state_fn, adam_init_state_fn],
        )
    return ChainedOptimizer(optimizers)
