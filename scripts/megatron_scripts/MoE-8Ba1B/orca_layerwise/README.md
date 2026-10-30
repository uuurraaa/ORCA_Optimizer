# Layerwise ORCA for MoE-8Ba1B

This directory retains only the layerwise configuration used for the
MoE-8Ba1B experiment in `orca.pdf`. It inherits the standard
`MoE-8Ba1B/muonbaseline` recipe (64 routed experts, top-4 routing,
micro-batch 16, global batch 1024, sequence length 2048, and 50k iterations)
and applies the paper's layerwise ORCA schedule:

- first layer: OWD active through 20% of training;
- last layer: OWD active through 60% of training;
- intermediate layers: uniformly interpolated release times;
- OWD target: `(XX^T - c^2 I)X`, with `c=0.1` and `c^2=0.01`;
- parameters without a layer marker: the standard 25% fallback.

Run the experiment with:

```bash
bash scripts/megatron_scripts/MoE-8Ba1B/orca_layerwise/launch_orca_layerwise20_60_c0p1_moe_seqaux_olmo_stage1_ep4_50k_mbs16.sh
```

Set `JOB_NAME`, `CHECKPOINT_ROOT`, or other baseline environment variables to
change the output locations. Additional arguments are forwarded to the
inherited training script.
