# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AdamW fallback with ORCA orthogonal weight decay."""

import math
from typing import Any, Optional

import torch

try:
    from emerging_optimizers import utils as emerging_utils
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        orthogonal_weight_decay_direction,
    )

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    emerging_utils = None
    orthogonal_weight_decay_direction = None


class ORCAAdamW:
    """AdamW-compatible wrapper with scheduled ORCA OWD for special matrices.

    Ordinary AdamW decay is applied explicitly before the wrapped optimizer's
    update, then temporarily disabled on the wrapped optimizer. This preserves
    ORCA's ordering: ordinary decoupled decay, OWD, then Adam moments/update.
    """

    metadata_key = "orca_adamw"
    _legacy_metadata_key = "period6adamw"

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        alpha1: float = 0.25,
        owdwd: float = 1.0,
        c: float = 0.1,
        wd_schedule: str = "stable_linear",
        wd_stable_ratio: float = 1.0,
        wd_end_scale: float = 0.0,
        router_owd: bool = True,
        shared_expert_gate_owd: bool = True,
        layerwise: bool = False,
        layerwise_alpha_start: float = 0.15,
        layerwise_alpha_end: float = 0.40,
        num_layers: Optional[int] = None,
        fp32_matmul_prec: str = "medium",
        use_triton_optimize: bool = False,
        initial_step: int = 0,
    ) -> None:
        if not HAVE_EMERGING_OPTIMIZERS:
            raise ImportError(
                "ORCAAdamW requires the emerging_optimizers package to be installed."
            )
        if total_steps is None:
            raise ValueError("ORCAAdamW requires total_steps to be set.")
        self.base_optimizer = base_optimizer
        self.total_steps = int(total_steps)
        self.alpha1 = float(alpha1)
        self.owdwd = float(owdwd)
        self.c = float(c)
        self.wd_schedule = str(wd_schedule)
        self.wd_stable_ratio = float(wd_stable_ratio)
        self.wd_end_scale = float(wd_end_scale)
        self.router_owd = bool(router_owd)
        self.shared_expert_gate_owd = bool(shared_expert_gate_owd)
        self.layerwise = bool(layerwise)
        self.layerwise_alpha_start = float(layerwise_alpha_start)
        self.layerwise_alpha_end = float(layerwise_alpha_end)
        self.num_layers = int(num_layers) if num_layers is not None else None
        self.fp32_matmul_prec = str(fp32_matmul_prec)
        self.use_triton_optimize = bool(use_triton_optimize)
        self.period_step = int(initial_step)
        self.special_steps = int(self.total_steps * self.alpha1)
        self.last_phase: Optional[str] = None
        self._validate_orca_hparams()

    @property
    def c_square(self) -> float:
        """Return the OWD target coefficient derived from ``c``."""
        return self.c * self.c

    def _validate_orca_hparams(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("ORCAAdamW requires total_steps to be positive.")
        if not math.isfinite(self.alpha1) or not 0.0 <= self.alpha1 <= 1.0:
            raise ValueError(f"ORCAAdamW alpha1 must be in [0, 1], got {self.alpha1}.")
        if not math.isfinite(self.owdwd) or self.owdwd < 0.0:
            raise ValueError(f"ORCAAdamW owdwd must be non-negative, got {self.owdwd}.")
        if not math.isfinite(self.c) or self.c < 0.0:
            raise ValueError(
                "ORCAAdamW c must be finite and non-negative, "
                f"got {self.c}."
            )
        try:
            c_square = self.c_square
        except OverflowError as exc:
            raise ValueError(f"ORCAAdamW c^2 must be finite, got {self.c}.") from exc
        if not math.isfinite(c_square):
            raise ValueError(f"ORCAAdamW c^2 must be finite, got {self.c_square}.")
        if self.wd_schedule != "stable_linear":
            raise ValueError(
                f"ORCAAdamW wd_schedule must be 'stable_linear', got {self.wd_schedule!r}."
            )
        if not math.isfinite(self.wd_stable_ratio) or not 0.0 <= self.wd_stable_ratio <= 1.0:
            raise ValueError(
                "ORCAAdamW wd_stable_ratio must be in [0, 1], "
                f"got {self.wd_stable_ratio}."
            )
        if not math.isfinite(self.wd_end_scale) or self.wd_end_scale < 0.0:
            raise ValueError(
                "ORCAAdamW wd_end_scale must be finite and non-negative, "
                f"got {self.wd_end_scale}."
            )
        if self.period_step < 0:
            raise ValueError(
                f"ORCAAdamW initial_step must be non-negative, got {self.period_step}."
            )
        if self.layerwise:
            if (
                not math.isfinite(self.layerwise_alpha_start)
                or not 0.0 <= self.layerwise_alpha_start <= 1.0
            ):
                raise ValueError(
                    "ORCAAdamW layerwise_alpha_start must be in [0, 1], "
                    f"got {self.layerwise_alpha_start}."
                )
            if (
                not math.isfinite(self.layerwise_alpha_end)
                or not 0.0 <= self.layerwise_alpha_end <= 1.0
            ):
                raise ValueError(
                    "ORCAAdamW layerwise_alpha_end must be in [0, 1], "
                    f"got {self.layerwise_alpha_end}."
                )
            if self.layerwise_alpha_start > self.layerwise_alpha_end:
                raise ValueError(
                    "ORCAAdamW layerwise_alpha_start must not exceed "
                    "layerwise_alpha_end."
                )
            if self.num_layers is None or self.num_layers <= 0:
                raise ValueError(
                    "ORCAAdamW layerwise schedule requires positive num_layers."
                )

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value

    @property
    def state(self):
        return self.base_optimizer.state

    @state.setter
    def state(self, value):
        self.base_optimizer.state = value

    def __getattr__(self, name: str) -> Any:
        # Delegate optimizer-specific methods/attributes (e.g. hooks) without
        # duplicating backend-specific AdamW interfaces.
        if name in {"base_optimizer", "metadata_key"}:
            raise AttributeError(name)
        return getattr(self.base_optimizer, name)

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    def _use_owd_phase(self) -> bool:
        return self.period_step < self.special_steps

    def current_phase(self) -> str:
        if self.layerwise:
            max_special_steps = max(
                int(self.total_steps * self.layerwise_alpha_end), self.special_steps
            )
            return (
                "orca_layerwise_owd_adamw"
                if self.period_step < max_special_steps
                else "orca_adamw"
            )
        return "orca_owd_adamw" if self._use_owd_phase() else "orca_adamw"

    def _owd_schedule_scale(self) -> float:
        if self.special_steps <= 0:
            return 0.0
        progress = min(
            max(float(self.period_step) / float(max(self.special_steps - 1, 1)), 0.0),
            1.0,
        )
        if progress <= self.wd_stable_ratio or self.wd_stable_ratio >= 1.0:
            return 1.0
        decay_progress = (progress - self.wd_stable_ratio) / (1.0 - self.wd_stable_ratio)
        return 1.0 + decay_progress * (self.wd_end_scale - 1.0)

    def _parameter_special_steps(self, p: torch.Tensor) -> int:
        layer_number = getattr(p, "orca_layer_number", None)
        if not self.layerwise or layer_number is None:
            return self.special_steps
        if self.num_layers <= 1:
            alpha = self.layerwise_alpha_start
        else:
            layer_index = min(max(int(layer_number) - 1, 0), self.num_layers - 1)
            alpha = self.layerwise_alpha_start + (
                self.layerwise_alpha_end - self.layerwise_alpha_start
            ) * (float(layer_index) / float(self.num_layers - 1))
        return int(self.total_steps * alpha)

    def _owd_schedule_scale_for_parameter(self, p: torch.Tensor) -> float:
        special_steps = self._parameter_special_steps(p)
        if special_steps <= 0:
            return 0.0
        progress = min(
            max(float(self.period_step) / float(max(special_steps - 1, 1)), 0.0),
            1.0,
        )
        if progress <= self.wd_stable_ratio or self.wd_stable_ratio >= 1.0:
            return 1.0
        decay_progress = (progress - self.wd_stable_ratio) / (
            1.0 - self.wd_stable_ratio
        )
        return 1.0 + decay_progress * (self.wd_end_scale - 1.0)

    def _owd_phase_for_parameter(self, p: torch.Tensor) -> bool:
        return self.period_step < self._parameter_special_steps(p)

    def _owd_enabled_for_parameter(self, p: torch.Tensor) -> bool:
        if self.owdwd == 0.0 or p.dim() != 2:
            return False
        if getattr(p, "is_router", False):
            return self.router_owd
        if getattr(p, "is_shared_expert_gate", False):
            return self.shared_expert_gate_owd
        return False

    def _owd_coefficient(self, lr: float, p: Optional[torch.Tensor] = None) -> float:
        scale = (
            self._owd_schedule_scale()
            if p is None
            else self._owd_schedule_scale_for_parameter(p)
        )
        return lr * self.owdwd * scale

    def _apply_orthogonal_weight_decay(self, p: torch.Tensor, lr: float) -> None:
        if not self._owd_enabled_for_parameter(p):
            return
        coefficient = self._owd_coefficient(lr, p)
        if coefficient == 0.0:
            return
        compute_matrix = (
            p.to(torch.bfloat16)
            if p.is_cuda and self.fp32_matmul_prec == "medium"
            else p
        )
        with emerging_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            direction = orthogonal_weight_decay_direction(
                compute_matrix,
                use_syrk=self.use_triton_optimize,
                alpha=self.c_square,
            ).to(p.dtype)
        p.add_(direction, alpha=-coefficient)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Match torch.optim's closure contract: evaluate the closure before
            # mutating parameters and let the wrapper own the single update call.
            with torch.enable_grad():
                loss = closure()

        self.last_phase = self.current_phase()
        original_weight_decays = [group.get("weight_decay", 0.0) for group in self.param_groups]
        try:
            for group in self.param_groups:
                lr = group["lr"]
                effective_weight_decay = group.get("wd_mult", 1.0) * group.get(
                    "weight_decay", 0.0
                )
                for p in group["params"]:
                    grad = getattr(p, "decoupled_grad", None)
                    if grad is None:
                        grad = p.grad
                    if grad is None:
                        continue
                    if effective_weight_decay != 0.0:
                        p.add_(p, alpha=-lr * effective_weight_decay)
                    if self._owd_phase_for_parameter(p):
                        self._apply_orthogonal_weight_decay(p, lr)
                # The ordinary decay has already been applied in the ORCA
                # order, so prevent the wrapped AdamW from applying it again.
                group["weight_decay"] = 0.0
            self.base_optimizer.step()
        finally:
            for group, weight_decay in zip(self.param_groups, original_weight_decays):
                group["weight_decay"] = weight_decay
        self.period_step += 1
        return loss

    def zero_grad(self, set_to_none: bool = True):
        return self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        state_dict = dict(self.base_optimizer.state_dict())
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
            "layerwise": self.layerwise,
            "layerwise_alpha_start": self.layerwise_alpha_start,
            "layerwise_alpha_end": self.layerwise_alpha_end,
            "num_layers": self.num_layers,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        metadata = state_dict.pop(self.metadata_key, None)
        loaded_legacy_metadata = False
        if metadata is None:
            metadata = state_dict.pop(self._legacy_metadata_key, None)
            loaded_legacy_metadata = metadata is not None
        inferred_period_step = self._infer_period_step(state_dict) if metadata is None else None
        self.base_optimizer.load_state_dict(state_dict)
        if metadata is None:
            # Checkpoints created before ORCAAdamW was introduced do not have
            # wrapper metadata. Reuse Adam's common update count so resuming an
            # old checkpoint does not restart the alpha1 OWD phase.
            if inferred_period_step is not None:
                self.period_step = inferred_period_step
                self.last_phase = None
            return
        self.total_steps = int(metadata.get("total_steps", self.total_steps))
        self.alpha1 = float(metadata.get("alpha1", self.alpha1))
        self.special_steps = int(metadata.get("special_steps", int(self.total_steps * self.alpha1)))
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
        self.wd_stable_ratio = float(metadata.get("wd_stable_ratio", self.wd_stable_ratio))
        self.wd_end_scale = float(metadata.get("wd_end_scale", self.wd_end_scale))
        self.router_owd = bool(metadata.get("router_owd", self.router_owd))
        self.shared_expert_gate_owd = bool(
            metadata.get("shared_expert_gate_owd", self.shared_expert_gate_owd)
        )
        self.layerwise = bool(metadata.get("layerwise", self.layerwise))
        self.layerwise_alpha_start = float(
            metadata.get("layerwise_alpha_start", self.layerwise_alpha_start)
        )
        self.layerwise_alpha_end = float(
            metadata.get("layerwise_alpha_end", self.layerwise_alpha_end)
        )
        stored_num_layers = metadata.get("num_layers", self.num_layers)
        self.num_layers = int(stored_num_layers) if stored_num_layers is not None else None
        self._validate_orca_hparams()

    @staticmethod
    def _infer_period_step(state_dict) -> Optional[int]:
        """Infer the completed Adam update count from a legacy state dict."""
        steps = []
        for param_state in state_dict.get("state", {}).values():
            if not isinstance(param_state, dict):
                continue
            if "step" not in param_state:
                continue
            step = param_state["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1:
                    continue
                step = step.item()
            try:
                step = int(step)
            except (TypeError, ValueError):
                continue
            if step >= 0:
                steps.append(step)

        if not steps:
            for group in state_dict.get("param_groups", []):
                step = group.get("step")
                if step is None:
                    continue
                if isinstance(step, torch.Tensor):
                    if step.numel() != 1:
                        continue
                    step = step.item()
                try:
                    step = int(step)
                except (TypeError, ValueError):
                    continue
                if step >= 0:
                    steps.append(step)

        if not steps:
            return None
        if len(set(steps)) != 1:
            # Adam may keep per-parameter steps when sparse/MoE parameters do
            # not receive a gradient on every update. The largest step is the
            # closest representation of the global training iteration and is
            # preferable to restarting the OWD phase.
            return max(steps)
        return steps[0]

    def __repr__(self) -> str:
        return f"ORCAAdamW({self.base_optimizer!r})"
