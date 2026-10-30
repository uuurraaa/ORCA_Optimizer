# ORCA

This repository provides an experimental implementation of ORCA (**Orthogonal Regularization, Cooled After**) based on Megatron-LM. ORCA temporarily shapes the spectrum of matrix weights early in training, then releases the constraint so the model can freely adapt to the task during later training. The repository exposes two methods:

- `orca`: all eligible matrices share the same ORCA release phase;
- `orca_layerwise`: different Transformer layers use different release phases.

The underlying training framework is [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). The matrix optimizer code is in [`megatron/core/optimizer/muon.py`](./megatron/core/optimizer/muon.py).

## Method

The ORCA orthogonal regularization objective is

$$
R_c(W)=\frac14\lVert WW^\top-c^2I\rVert_F^2,
\qquad
\nabla R_c(W)=(WW^\top-c^2I)W.
$$

During the shaping phase, the idealized full update can be written as

$$
\begin{aligned}
W_{t+1}
={}&W_t-\eta_tU_t-\eta_t\lambda W_t\\
&-\eta_t\gamma s_t(W_tW_t^\top-c^2I)W_t,
\end{aligned}
$$

where:

- $U_t$ is the base update from Muon or the corresponding fallback optimizer;
- $\lambda$ is ordinary weight decay;
- `--orca-c` specifies the target scale $c$;
- `--orca-owdwd` specifies the shaping strength $\gamma$;
- $s_t$ is the ORCA phase coefficient.

Thus, $c$ and $\gamma$ are separate hyperparameters. The current experiment scripts use $c=0.1$ and $\gamma=1$.

## Release Schedule

`orca` uses a global active-window fraction, with a code default of `0.25`. `orca_layerwise` linearly interpolates the release fraction across layers. The current paper reproduction experiments use `0.20` for the first layer and `0.60` for the last layer.

## Quick Start

### MoE 8B / 1B Active

```bash
# Global ORCA, 50k steps
bash scripts/megatron_scripts/MoE-8Ba1B/orca/launch_orca_moe_seqaux_olmo_stage1_ep4_50k_mbs16.sh

# Layerwise ORCA, 20% -> 60%, 50k steps
bash scripts/megatron_scripts/MoE-8Ba1B/orca_layerwise/launch_orca_layerwise20_60_c0p1_moe_seqaux_olmo_stage1_ep4_50k_mbs16.sh
```

### MoE 0.8B / 0.24B Active

```bash
bash scripts/megatron_scripts/MoE-0.8Ba0.24B/orca/launch_orca_moe_seqaux_olmo_stage1_ep8.sh
```

The training scripts use the existing OLMo IndexedDataset, checkpoint, and launcher configurations by default. Before submitting a long training run, check the data paths, checkpoint paths, GPU count, `EP_SIZE`, and distributed environment variables in the scripts.

## Practical Guidance from Current Experiments

The following observations come from the current implementation and internal experiments. Retune the settings when changing the data, model, or training budget.

- Larger models may benefit from a larger $\gamma$ (`--orca-owdwd`). Internal experiments suggest that values above `1` and up to `10` can still be effective. However, excessive shaping strength may require a lower learning rate, a shorter shaping window, or further tuning. The code does not automatically provide stability safeguards for very large $\gamma$.

- $c$ controls the target singular-value scale. Current experience suggests that larger and deeper models often require a smaller $c$.

- The method depends on the number of training steps. Internal searches found that runs with around 10k optimizer steps or fewer provide no benefit and can even perform worse. The method generally needs more optimizer steps to achieve sufficient loss reduction.

- In the current search over logical matrix granularity, the best-performing configuration splits Q/K by head or query group while treating V/O as whole matrices. This is an internal search result, not an ablation result from the paper.

- Applying ORCA/OWD to the router provides some benefit in the current MoE configurations. Layerwise release is also used to mitigate router perturbations that may arise from a fixed global release point.

## Parameters and Implementation Details

### Parameter Groups

- Embeddings, the LM head, and one-dimensional parameters use the AdamW fallback.
- Other eligible two-dimensional matrices use Muon/ORCA.
- The router and shared-expert gate use Muon + OWD by default.

ORCA OWD is independent of ordinary `weight_decay`. Setting ordinary weight decay to zero does not automatically disable OWD. To disable it, set `--orca-owdwd` to `0` or use the corresponding no-OWD flag.

### Logical Matrix Splitting

The implementation uses the following logical matrix boundaries:

- Q: one matrix per attention head;
- K: one matrix per query group, which should not simply be called an attention head in GQA;
- V: one whole matrix combining all query groups;
- O projection: one whole matrix;
- SwiGLU FC1: separate gate and up matrices;
- Routed experts: each expert is processed independently.

Matrices with the same shape can be batched for Muon/OWD, with a default limit of 32 logical matrices per batch. Batching affects execution efficiency only and does not change these logical boundaries.

## Checkpoints and Reproducibility

The ORCA checkpoint state records the training phase, release fractions, $\gamma$, $c$ and $c^2$, the OWD schedule, and the router/shared-gate policies. When fully restoring the optimizer, these values may override the current command-line arguments. To retain momentum while changing the schedule, use `--orca-override-checkpoint-hparams`.

## Relevant Paths

- [`megatron/core/optimizer/muon.py`](./megatron/core/optimizer/muon.py): Muon, ORCA, and logical matrix splitting.
- [`emerging_optimizers/orthogonalized_optimizers/`](./emerging_optimizers/orthogonalized_optimizers/): Orthogonalization and Gram/SYRK utilities.
- [`scripts/megatron_scripts/MoE-8Ba1B/`](./scripts/megatron_scripts/MoE-8Ba1B/): Experiment scripts for the 8B / 1B-active model.
- [`scripts/megatron_scripts/MoE-0.8Ba0.24B/`](./scripts/megatron_scripts/MoE-0.8Ba0.24B/): Experiment scripts for the 0.8B / 0.24B-active model.
- [`scripts/megatron_scripts/MoE-8Ba1B/orca_layerwise/README.md`](./scripts/megatron_scripts/MoE-8Ba1B/orca_layerwise/README.md): Layerwise experiment documentation.
