# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Evaluation-time logging of logical weight-matrix norms."""

import logging
import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import unwrap_model


logger = logging.getLogger(__name__)

_LAST_LOGGED_ITERATION = None


def split_qkv_weight(
    weight: torch.Tensor,
    num_attention_heads: int,
    num_query_groups: int,
    kv_channels: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a fused GQA QKV weight into component matrices and per-head matrices.

    Megatron lays out a fused QKV weight by query group. Within each group, the rows are
    ``[Q heads..., K head, V head]``.
    """
    if weight.ndim != 2:
        raise ValueError(f"QKV weight must be 2D, got shape {tuple(weight.shape)}")
    if num_attention_heads % num_query_groups != 0:
        raise ValueError(
            f"num_attention_heads ({num_attention_heads}) must be divisible by "
            f"num_query_groups ({num_query_groups})"
        )

    heads_per_group = num_attention_heads // num_query_groups
    q_rows_per_group = heads_per_group * kv_channels
    rows_per_group = q_rows_per_group + 2 * kv_channels
    expected_rows = num_query_groups * rows_per_group
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"Fused QKV weight has {weight.shape[0]} rows, expected {expected_rows} from "
            f"num_attention_heads={num_attention_heads}, num_query_groups={num_query_groups}, "
            f"kv_channels={kv_channels}. Weight-matrix norm logging does not support this "
            "QKV layout."
        )

    input_size = weight.shape[1]
    grouped = weight.reshape(num_query_groups, rows_per_group, input_size)
    q_grouped, k_heads, v_heads = torch.split(
        grouped, [q_rows_per_group, kv_channels, kv_channels], dim=1
    )
    q_heads = q_grouped.reshape(num_attention_heads, kv_channels, input_size)

    components = {
        "q": q_heads.reshape(num_attention_heads * kv_channels, input_size),
        "k": k_heads.reshape(num_query_groups * kv_channels, input_size),
        "v": v_heads.reshape(num_query_groups * kv_channels, input_size),
    }
    heads = {"q": q_heads, "k": k_heads, "v": v_heads}
    return components, heads


def split_gated_weight(
    weight: torch.Tensor, dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a gated linear weight into equally-sized gate and up matrices."""
    if weight.ndim != 2:
        raise ValueError(f"Gated weight must be 2D, got shape {tuple(weight.shape)}")
    if weight.shape[dim] % 2 != 0:
        raise ValueError(
            f"Cannot split gated weight shape {tuple(weight.shape)} evenly along dim {dim}"
        )
    return torch.chunk(weight, 2, dim=dim)


@torch.no_grad()
def matrix_norms(
    matrices: torch.Tensor, power_iterations: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 Frobenius norms and approximate spectral norms.

    ``matrices`` may be one matrix ``[m, n]`` or a batch ``[..., m, n]``. Spectral norms
    are estimated with deterministic bilateral power iteration so logging does not perturb
    the training RNG state.
    """
    if matrices.ndim < 2:
        raise ValueError(
            f"Expected matrices with ndim >= 2, got shape {tuple(matrices.shape)}"
        )
    if power_iterations < 1:
        raise ValueError(f"power_iterations must be at least 1, got {power_iterations}")

    work = matrices.detach().to(dtype=torch.float32)
    fnorm = torch.linalg.vector_norm(work, dim=(-2, -1))

    vector_shape = (*work.shape[:-2], work.shape[-1], 1)
    # An isolated generator avoids changing training RNG state while also avoiding the
    # convergence failures that a fixed all-ones vector can have for structured matrices.
    generator = torch.Generator(device=work.device)
    generator.manual_seed(0)
    v = torch.randn(
        vector_shape, dtype=work.dtype, device=work.device, generator=generator
    )
    v = F.normalize(v, dim=-2)
    for _ in range(power_iterations):
        u = F.normalize(work @ v, dim=-2)
        v = F.normalize(work.transpose(-2, -1) @ u, dim=-2)

    spectral_norm = torch.linalg.vector_norm(work @ v, dim=-2).squeeze(-1)
    return fnorm, spectral_norm


def _record_matrices(
    metrics: Dict[str, float],
    names: Sequence[str],
    matrices: torch.Tensor,
    power_iterations: int,
) -> None:
    if matrices.ndim == 2:
        matrices = matrices.unsqueeze(0)
    if matrices.shape[0] != len(names):
        raise ValueError(
            f"Received {matrices.shape[0]} matrices but {len(names)} metric names"
        )

    fnorms, spectral_norms = matrix_norms(matrices, power_iterations)
    for name, fnorm, spectral_norm in zip(
        names, fnorms.tolist(), spectral_norms.tolist()
    ):
        metrics[f"weight-fnorm/{name}"] = fnorm
        metrics[f"weight-spectral-norm/{name}"] = spectral_norm


def _record_matrix(
    metrics: Dict[str, float],
    name: str,
    matrix: torch.Tensor,
    power_iterations: int,
) -> None:
    _record_matrices(metrics, [name], matrix, power_iterations)


def _record_mlp(
    metrics: Dict[str, float],
    prefix: str,
    mlp,
    gated: bool,
    power_iterations: int,
) -> None:
    fc1 = getattr(getattr(mlp, "linear_fc1", None), "weight", None)
    if fc1 is not None:
        if gated:
            gate, up = split_gated_weight(fc1, dim=0)
            _record_matrix(metrics, f"{prefix}/gate", gate, power_iterations)
            _record_matrix(metrics, f"{prefix}/up", up, power_iterations)
        else:
            _record_matrix(metrics, f"{prefix}/fc1", fc1, power_iterations)

    fc2 = getattr(getattr(mlp, "linear_fc2", None), "weight", None)
    if fc2 is not None:
        _record_matrix(metrics, f"{prefix}/down", fc2, power_iterations)


def _grouped_linear_weights(grouped_linear) -> List[torch.Tensor]:
    """Return GroupedLinear weights in GEMM/expert order across TE implementations."""
    indexed_weights = []
    for name, parameter in grouped_linear.named_parameters(recurse=True):
        if parameter.ndim != 2 or "weight" not in name:
            continue
        match = re.search(r"(?:^|\.)(\d+)\.weight$", name)
        if match is None:
            match = re.search(r"weight(\d+)$", name)
        if match is not None:
            indexed_weights.append((int(match.group(1)), parameter))

    indexed_weights.sort(key=lambda item: item[0])
    return [parameter for _, parameter in indexed_weights]


def _record_experts(
    metrics: Dict[str, float],
    prefix: str,
    moe,
    power_iterations: int,
) -> None:
    experts = moe.experts
    local_expert_indices = list(getattr(moe, "local_expert_indices", []))
    if not local_expert_indices:
        local_expert_indices = list(range(getattr(experts, "num_local_experts", 0)))
    num_local_experts = len(local_expert_indices)
    gated = bool(moe.config.gated_linear_unit)

    if hasattr(experts, "local_experts"):
        local_experts = list(experts.local_experts)
        if len(local_experts) != num_local_experts:
            raise RuntimeError(
                f"Found {len(local_experts)} local expert modules but expected "
                f"{num_local_experts}"
            )
        for expert_id, expert in zip(local_expert_indices, local_experts):
            _record_mlp(
                metrics,
                f"{prefix}/expert_{expert_id:03d}",
                expert,
                gated,
                power_iterations,
            )
        return

    if hasattr(experts, "weight1") and hasattr(experts, "weight2"):
        # Legacy GroupedMLP executes weight1 as [E, H, 2F] and weight2 as [E, F, H].
        weight1 = experts.weight1.reshape(num_local_experts, moe.config.hidden_size, -1)
        weight2 = experts.weight2.reshape(num_local_experts, -1, moe.config.hidden_size)
        for local_index, expert_id in enumerate(local_expert_indices):
            expert_prefix = f"{prefix}/expert_{expert_id:03d}"
            if gated:
                gate, up = split_gated_weight(weight1[local_index], dim=1)
                _record_matrix(metrics, f"{expert_prefix}/gate", gate, power_iterations)
                _record_matrix(metrics, f"{expert_prefix}/up", up, power_iterations)
            else:
                _record_matrix(
                    metrics,
                    f"{expert_prefix}/fc1",
                    weight1[local_index],
                    power_iterations,
                )
            _record_matrix(
                metrics, f"{expert_prefix}/down", weight2[local_index], power_iterations
            )
        return

    if hasattr(experts, "linear_fc1") and hasattr(experts, "linear_fc2"):
        fc1_weights = _grouped_linear_weights(experts.linear_fc1)
        fc2_weights = _grouped_linear_weights(experts.linear_fc2)
        if (
            len(fc1_weights) != num_local_experts
            or len(fc2_weights) != num_local_experts
        ):
            raise RuntimeError(
                "Unable to identify one GroupedLinear FC1/FC2 weight per local expert: "
                f"found {len(fc1_weights)} FC1 and {len(fc2_weights)} FC2 weights, expected "
                f"{num_local_experts}"
            )
        for expert_id, fc1, fc2 in zip(local_expert_indices, fc1_weights, fc2_weights):
            expert_prefix = f"{prefix}/expert_{expert_id:03d}"
            if gated:
                gate, up = split_gated_weight(fc1, dim=0)
                _record_matrix(metrics, f"{expert_prefix}/gate", gate, power_iterations)
                _record_matrix(metrics, f"{expert_prefix}/up", up, power_iterations)
            else:
                _record_matrix(metrics, f"{expert_prefix}/fc1", fc1, power_iterations)
            _record_matrix(metrics, f"{expert_prefix}/down", fc2, power_iterations)
        return

    raise RuntimeError(
        f"Unsupported MoE experts implementation: {type(experts).__name__}"
    )


def _record_transformer_layer(
    metrics: Dict[str, float],
    layer: TransformerLayer,
    power_iterations: int,
    record_dense: bool,
    record_experts: bool,
) -> None:
    if layer.layer_number is None:
        return
    prefix = f"layer_{layer.layer_number:02d}"
    config = layer.config

    if record_dense:
        attention = getattr(layer, "self_attention", None)
        qkv_weight = getattr(getattr(attention, "linear_qkv", None), "weight", None)
        if qkv_weight is not None:
            components, heads = split_qkv_weight(
                qkv_weight,
                config.num_attention_heads,
                config.num_query_groups,
                config.kv_channels,
            )
            for component_name in ("q", "k", "v"):
                component_prefix = f"{prefix}/attention/{component_name}"
                _record_matrix(
                    metrics,
                    component_prefix,
                    components[component_name],
                    power_iterations,
                )
                head_names = [
                    f"{component_prefix}/head_{head_index:03d}"
                    for head_index in range(heads[component_name].shape[0])
                ]
                _record_matrices(
                    metrics, head_names, heads[component_name], power_iterations
                )

        projection = getattr(getattr(attention, "linear_proj", None), "weight", None)
        if projection is not None:
            _record_matrix(
                metrics, f"{prefix}/attention/o_proj", projection, power_iterations
            )

    mlp = getattr(layer, "mlp", None)
    is_moe = mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "router")
    if not is_moe:
        if record_dense and mlp is not None:
            _record_mlp(
                metrics,
                f"{prefix}/mlp",
                mlp,
                bool(config.gated_linear_unit),
                power_iterations,
            )
        return

    if record_dense:
        router_weight = getattr(getattr(mlp, "router", None), "weight", None)
        if router_weight is not None:
            _record_matrix(metrics, f"{prefix}/router", router_weight, power_iterations)
        shared_experts = getattr(mlp, "shared_experts", None)
        if shared_experts is not None:
            _record_mlp(
                metrics,
                f"{prefix}/shared_expert",
                shared_experts,
                bool(config.gated_linear_unit),
                power_iterations,
            )
            shared_gate = getattr(shared_experts, "gate_weight", None)
            if shared_gate is not None:
                _record_matrix(
                    metrics,
                    f"{prefix}/shared_expert/output_gate",
                    shared_gate,
                    power_iterations,
                )

    if record_experts:
        _record_experts(metrics, prefix, mlp, power_iterations)


def _is_dense_replica() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0


def _is_expert_replica() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    expert_dp_group = parallel_state.get_expert_data_parallel_group(
        check_initialized=False
    )
    return expert_dp_group is None or expert_dp_group.rank() == 0


def compute_weight_matrix_norms(
    model,
    power_iterations: int,
) -> Dict[str, float]:
    """Compute local logical weight-matrix metrics for one replica."""
    model_chunks = unwrap_model(model)
    if not isinstance(model_chunks, list):
        model_chunks = [model_chunks]

    metrics: Dict[str, float] = {}
    seen_layers = set()
    record_dense = _is_dense_replica()
    record_experts = _is_expert_replica()
    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            if isinstance(module, TransformerLayer) and id(module) not in seen_layers:
                seen_layers.add(id(module))
                _record_transformer_layer(
                    metrics,
                    module,
                    power_iterations,
                    record_dense=record_dense,
                    record_experts=record_experts,
                )
    return metrics


def _gather_metrics(local_metrics: Dict[str, float]) -> Dict[str, float]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_metrics

    gathered_metrics = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered_metrics, local_metrics)
    merged_metrics = {}
    for rank_metrics in gathered_metrics:
        if rank_metrics:
            merged_metrics.update(rank_metrics)
    return merged_metrics


def log_weight_matrix_norms(
    model,
    iteration: int,
    writer,
    wandb_writer,
    power_iterations: int,
) -> None:
    """Compute and log logical weight-matrix norms once for an evaluation iteration."""
    global _LAST_LOGGED_ITERATION
    if _LAST_LOGGED_ITERATION == iteration:
        return

    local_metrics = compute_weight_matrix_norms(model, power_iterations)
    metrics = _gather_metrics(local_metrics)

    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(name, value, iteration)
    if wandb_writer is not None and parallel_state.is_pipeline_last_stage(
        ignore_virtual=True
    ):
        wandb_writer.log(metrics, iteration)

    _LAST_LOGGED_ITERATION = iteration
    if writer is not None or wandb_writer is not None:
        logger.info(
            "Logged %d logical weight-matrix norm values at iteration %d",
            len(metrics),
            iteration,
        )
