# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ..utils import is_te_min_version


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    ##############
    # General
    ##############
    optimizer: str = 'adam'
    """Optimizer to use (including the ORCA Muon variants)."""

    lr: Optional[float] = None
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    min_lr: Optional[float] = None
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    decoupled_lr: Optional[float] = None
    """Separate learning rate for the input and output layer."""

    decoupled_min_lr: Optional[float] = None
    """Minimum value for learning rate for the input and output layer. The scheduler clip values
       below this threshold.
    """

    weight_decay: float = 0.01
    """Weight decay coefficient for L2 regularization."""

    ##############
    # Precision
    ##############
    fp8_recipe: Optional[str] = None
    """The type of fp8 recipe will affect the processing logic inside distributed optimizer."""

    fp16: bool = False
    """If true, train with fp16 mixed precision training. Defaults to False."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training. Defaults to False."""

    reuse_grad_buf_for_mxfp8_param_ag: bool = False
    """If true, reuse the grad buffer for param AG when using mxfp8 recipe. Should be 
       set to True only when fp8_recipe is mxfp8 and fp8_param_gather is True."""

    params_dtype: torch.dtype = torch.float32
    """dtype used when intializing the weights. Defaults to torch.float32."""

    use_precision_aware_optimizer: bool = False
    """If true, allows optimizer-related tensors (master_param, gradients and optimizer states)
    to be set to lower precision. Defaults to False.
    """

    store_param_remainders: bool = True
    """If true, store the 16-bit FP32 parameter remainders in the optimizer state, excluding the
        16 bits shared with the BF16 parameters. This lowers GPU memory usage. Defaults to True.
    """

    main_grads_dtype: torch.dtype = torch.float32
    """dtype of main grads when enabling precision-aware-optimizer"""

    main_params_dtype: torch.dtype = torch.float32
    """dtype of main params when enabling precision-aware-optimizer"""

    exp_avg_dtype: torch.dtype = torch.float32
    """dtype of exp_avg when enabling precision-aware-optimizer"""

    exp_avg_sq_dtype: torch.dtype = torch.float32
    """dtype of exp_avg_sq when enabling precision-aware-optimizer"""

    ###############
    # Loss scaling
    ###############
    loss_scale: Optional[float] = None
    """Static loss scaling, positive power of 2 values can improve fp16 convergence. If None,
       dynamic loss scaling is used.
    """

    initial_loss_scale: float = 2**32
    """Initial loss-scale for dynamic loss scaling."""

    min_loss_scale: float = 1.0
    """Minimum loss scale for dynamic loss scaling."""

    loss_scale_window: float = 1000
    """Window over which to raise/lower dynamic scale."""

    hysteresis: int = 2
    """Hysteresis for dynamic loss scaling."""

    ##############
    # Optimizer
    ##############
    # Adam
    adam_beta1: float = 0.9
    """First coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_beta2: float = 0.999
    """Second coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_eps: float = 1e-08
    """Term added to the denominator to improve numerical stability in Adam optimizer."""

    decoupled_weight_decay: bool = True
    """If true, decouples weight decay from the gradient update, equivalent to AdamW. If false,
    original Adam update rule will be used. Defaults to True.
    """

    # SGD.
    sgd_momentum: float = 0.9
    """Momentum factor for SGD optimizer."""

    # Muon
    muon_momentum: float = 0.95
    """The momentum used by the internal SGD."""

    muon_split_qkv: bool = True
    """Whether to split QKV parameters for Muon optimizer."""

    muon_qkv_split_mode: str = "component"
    """QKV split mode for Muon optimizer. Options:
    - 'component': merge all groups' Q together, all K together, all V together (original behavior)
    - 'group': process each query group independently with Q/K/V split within each group
              (aligns with split_qkv_init initialization)
    - 'head': process each attention head independently for Q/K/V
    - 'orca': process Q/K headwise and merge all V query groups into one matrix
    """

    muon_split_fc1: bool = False
    """Whether to split FC1 (gate and up) for gated linear units (SwiGLU) in Muon optimizer.
    When enabled, gate and up projections are treated as independent linear transformations.
    """

    muon_use_nesterov: bool = False
    """Whether to use Nesterov-style momentum in the internal SGD."""

    muon_scale_mode: str = "align_adamw_rms"
    """The mode to use for the scale factor. Defaults to AdamW-RMS alignment."""

    muon_fp32_matmul_prec: str = "medium"
    """The precision to use for the fp32 matmul. Defaults to "medium"."""

    muon_coefficient_type: str = "polar_express"
    """Newton-Schulz coefficient set used by Muon. Defaults to "polar_express"."""

    muon_num_ns_steps: int = 5
    """The number of iteration steps to use in the Newton-Schulz iteration."""

    muon_optimizer_max_batch_size: int = 32
    """Maximum logical matrices stacked by one batched Muon/ORCA call.

    A value of one preserves logical splitting but disables batching across
    gate/up, K/V, TE experts, and layers.
    """

    use_triton_optimize: bool = False
    """Whether Muon should use the custom Triton SYRK kernels.

    When disabled, Newton-Schulz and ORCA orthogonal weight decay use
    PyTorch/cuBLAS matrix operations instead.
    """

    muon_tp_mode: str = "blockwise"
    """How to perform NS calculation for tensor parallel weights. Defaults to "blockwise"."""

    muon_extra_scale_factor: float = 1.0
    """Additional scale factor for the muon update."""

    muon_split_moe_experts: bool = True
    """Whether to split MoE experts for Muon optimizer.
    When enabled, each expert's parameters in GroupedMLP are orthogonalized independently,
    preserving expert independence and avoiding gradient interference across experts.
    """

    muon_router_adamw: bool = False
    """Whether MoE router matrices use the AdamW fallback instead of Muon."""

    muon_shared_expert_gate_adamw: bool = False
    """Whether shared-expert gate matrices use the AdamW fallback instead of Muon."""

    # ORCA
    orca_total_steps: Optional[int] = None
    """Total optimizer steps used to determine the ORCA phase boundary."""

    orca_num_layers: Optional[int] = None
    """Total Transformer layers used by the layerwise ORCA variant."""

    orca_layerwise: bool = False
    """Whether ORCA OWD uses the layerwise alpha schedule."""

    orca_alpha1: float = 0.25
    """Fraction of training steps during which orthogonal weight decay is active."""

    orca_layerwise_alpha_start: float = 0.15
    """OWD phase fraction for the first Transformer layer in the layerwise variant."""

    orca_layerwise_alpha_end: float = 0.40
    """OWD phase fraction for the last Transformer layer in the layerwise variant."""

    orca_owdwd: float = 1.0
    """Orthogonal weight decay coefficient used during the initial ORCA phase."""

    orca_c: float = 0.1
    """Target singular-value scale in ORCA OWD; the direction uses ``c**2``."""

    orca_wd_schedule: str = "stable_linear"
    """Schedule for ORCA orthogonal weight decay within the initial phase."""

    orca_wd_stable_ratio: float = 1.0
    """Fraction of the initial phase for which the ORCA OWD scale remains at one."""

    orca_wd_end_scale: float = 0.0
    """Final ORCA OWD scale at the end of the initial phase."""

    orca_router_muon: bool = True
    """Whether router matrices are updated by Muon. If disabled, AdamW updates them."""

    orca_router_owd: bool = True
    """Whether router matrices receive ORCA orthogonal weight decay (ORCA)."""

    orca_shared_expert_gate_muon: bool = True
    """Whether shared-expert gate matrices are updated by Muon. If disabled, AdamW updates them."""

    orca_shared_expert_gate_owd: bool = True
    """Whether shared-expert gate matrices receive ORCA orthogonal weight decay (ORCA)."""

    orca_adam_owd: bool = False
    """Internal flag enabling ORCA OWD on AdamW fallback special parameters."""

    #######################
    # Distributed optimizer
    #######################
    use_distributed_optimizer: bool = False
    """Distribute optimizer state over data-parallel replicas."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute. 
        This argument is intended to have the same value as the "overlap_param_gather" argument 
        in the "distributed_data_parallel_config.py" file. In the optimizer, this argument is 
        only used when "reuse_grad_buf_for_mxfp8_param_ag=True & fp8_param_gather=True".
    """

    overlap_param_gather_with_optimizer_step: bool = False
    """If true, overlap param all-gather of first bucket with optimizer step."""

    #######################
    # Optimizer Offload
    #######################

    optimizer_cpu_offload: bool = False
    """If True, offload optimizer states tensor and compute to CPU."""

    optimizer_offload_fraction: float = 0.0
    """Specifies the fraction of optimizer states to offload from GPU memory to CPU."""

    use_torch_optimizer_for_cpu_offload: bool = False
    """If True, use torch.optim.Optimizer for CPU offload."""

    overlap_cpu_optimizer_d2h_h2d: bool = False
    """
    When set to `True`, this flag enables overlapping of the CPU optimizer
    update process with the data transfer operations. This can help improve
    overall training efficiency by reducing idle time during data movement,
    allowing the optimizer to perform updates while gradients and parameters
    are being transferred between devices.
    """

    pin_cpu_grads: bool = True
    """If True, pin the optimizer gradients to CPU memory."""

    pin_cpu_params: bool = True
    """If True, pin the optimizer parameters to CPU memory."""

    ################
    # Miscellaneous
    ################
    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    log_num_zeros_in_grad: bool = False
    """If true, calculate and log the number of zeros in gradient."""

    log_per_module_update_rms: bool = False
    """If true, calculate and log per-module update RMS for optimizers (Muon and AdamW)."""

    log_per_module_grad_rms: bool = False
    """If true, calculate and log per-module grad RMS for optimizers."""

    barrier_with_L1_time: bool = False
    """If true, use barrier with level 1 time measurements."""

    timers: Optional[Callable] = None
    """Function to get timers."""

    config_logger_dir: str = ""
    """When non-empty, dumps entry-point configs to config_logger_dir"""

    def __post_init__(self):
        """Check the validity of the config."""

        if self.muon_optimizer_max_batch_size < 1:
            raise ValueError(
                "muon_optimizer_max_batch_size must be at least 1, got "
                f"{self.muon_optimizer_max_batch_size}"
            )

        # The following condition is used to avoid repetition in distrib_optimizer.py.
        # This is because in distrib_optimizer.py, the process to handle parameters are
        # different for different training precision settings. FP8 cases require different
        # handling while FP8 delayed scaling is an exception because the Adam optimizer in
        # TransformerEngine supports it in the kernel computation.
        # This is also the flag to determine the usage of param.grad or param.decoupled_grad
        self.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = (
            self.use_precision_aware_optimizer
            and (
                self.main_params_dtype != torch.float32
                or (self.fp8_recipe is None or self.fp8_recipe == "delayed")
                or self.optimizer_cpu_offload
            )
        )

        if self.fp8_recipe == "mxfp8":
            if not self.reuse_grad_buf_for_mxfp8_param_ag:
                import warnings

                warnings.warn(
                    "mxfp8 without using reuse_grad_buf_for_mxfp8_param_ag and fp8_param_gather"
                    "will use significant amount additional GPU memory."
                    "Setting --reuse-grad-buf-for-mxfp8-param-ag and --fp8-param-gather is "
                    "recommended for mxfp8 training."
                )

        if self.use_precision_aware_optimizer:
            assert (
                self.optimizer == 'adam'
            ), '--use-precision-aware-optimizer only supported with adam'
            assert (
                self.use_distributed_optimizer
            ), '--use-precision-aware-optimizer only supported with distributed optimizer'

            if not is_te_min_version("2.1.0"):
                self.store_param_remainders = False

            # Only the FusedAdam in TE and HybridDeviceOptimizer supports
            # --use-precision-aware-optimizer.
            # TODO: Remove this check when apex's FusedAdam is no longer used.
            if self.optimizer_cpu_offload:
                return
            try:
                import inspect

                from transformer_engine.pytorch.optimizers import FusedAdam as Adam

                adam_args = inspect.signature(Adam).parameters
                arg_names = [
                    'master_weight_dtype',
                    'exp_avg_dtype',
                    'exp_avg_sq_dtype',
                    'use_decoupled_grad',
                ]
                for name in arg_names:
                    assert name in adam_args, (
                        "Current FusedAdam of TE doesn't support --use-precision-aware-optimizer, "
                        "please update TE version."
                    )
            except ImportError:
                raise RuntimeError(
                    '--use-precision-aware-optimizer requires FusedAdam from TransformerEngine, '
                    'but not found.'
                )
        else:
            assert (
                self.main_grads_dtype == torch.float32
            ), "main_grads_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.main_params_dtype == torch.float32
            ), "main_params_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.exp_avg_dtype == torch.float32
            ), "exp_avg_dtype can only be fp32 when not using precision-aware optimizer"
            assert (
                self.exp_avg_sq_dtype == torch.float32
            ), "exp_avg_sq_dtype can only be fp32 when not using precision-aware optimizer"
