# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint helpers for the layer-wise distributed optimizer."""

from typing import Any

import torch


GlobalParamGroups = list[list[list[torch.nn.Parameter]]]


def snapshot_global_param_groups(optimizers: list[Any]) -> GlobalParamGroups:
    """Keep the unsharded parameter-group structure for checkpoint loading."""
    return [
        [list(group["params"]) for group in optimizer.param_groups] for optimizer in optimizers
    ]


def install_global_float16_groups(
    optimizers: list[Any], global_param_groups: GlobalParamGroups
) -> None:
    """Give mixed-precision optimizers a name-addressable load template.

    Layer-wise sharding changes parameter ownership when the DP size changes.  The
    distributed checkpoint loader can reshard optimizer tensors by their model
    parameter keys, but it first needs the original, unsharded group structure.
    """
    if len(optimizers) != len(global_param_groups):
        raise RuntimeError(
            "Layer-wise optimizer structure changed while preparing checkpoint load: "
            f"{len(optimizers)} optimizers, {len(global_param_groups)} snapshots"
        )

    float16_dtypes = (torch.float16, torch.bfloat16)
    for optimizer, optimizer_groups in zip(optimizers, global_param_groups):
        if not hasattr(optimizer, "float16_groups"):
            continue

        groups = [
            [param for param in group if param.requires_grad and param.dtype in float16_dtypes]
            for group in optimizer_groups
        ]
        if len(groups) != len(optimizer.float16_groups):
            raise RuntimeError(
                "Layer-wise optimizer parameter groups changed while preparing checkpoint load: "
                f"{len(optimizer.float16_groups)} current groups, {len(groups)} global groups"
            )
        optimizer.global_float16_groups = groups


def restore_empty_state_dict_entries(optimizers: list[Any], state_dict: Any) -> None:
    """Restore empty containers omitted by distributed checkpoint loading."""
    if len(optimizers) == 1:
        optimizer_state_dicts = [state_dict]
    elif isinstance(state_dict, dict):
        optimizer_state_dicts = [value for _, value in sorted(state_dict.items())]
    else:
        optimizer_state_dicts = state_dict

    for optimizer, optimizer_state_dict in zip(optimizers, optimizer_state_dicts):
        if not hasattr(optimizer, "float16_groups"):
            continue

        optimizer_key = (
            "optimizer" if "optimizer" in optimizer_state_dict else "optimizer_state_dict"
        )
        optimizer_state_dict[optimizer_key].setdefault("state", {})
        if not any(
            key in optimizer_state_dict
            for key in ("fp32_from_fp16_params", "fp32_from_fp16")
        ):
            optimizer_state_dict["fp32_from_fp16_params"] = []
