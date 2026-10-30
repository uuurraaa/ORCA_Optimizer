#!/usr/bin/env bash

# ORCA run for 50k steps with micro-batch 16.
# LAUNCHER=single_node bash launch_orca_moe_seqaux_olmo_stage1_ep4_50k_mbs16.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export TRAINING_SCRIPT=${TRAINING_SCRIPT:-"${SCRIPT_DIR}/orca_moe_seqaux_olmo_stage1_ep4_50k_mbs16.sh"}

exec bash "${REPO_ROOT}/scripts/launchers/launch_experiment.sh" "$@"
