#!/usr/bin/env bash

# MoE-8Ba1B layerwise ORCA, 20% to 60%, OWD c=0.1, 50k steps, micro-batch 16.
# LAUNCHER=single_node bash launch_orca_layerwise20_60_c0p1_moe_seqaux_olmo_stage1_ep4_50k_mbs16.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export TRAINING_SCRIPT=${TRAINING_SCRIPT:-"${SCRIPT_DIR}/orca_layerwise20_60_c0p1_moe_seqaux_olmo_stage1_ep4_50k_mbs16.sh"}

exec bash "${REPO_ROOT}/scripts/launchers/launch_experiment.sh" "$@"
