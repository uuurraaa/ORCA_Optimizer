#!/usr/bin/env bash

# OLMo stage-1 Lite ORCA experiment.
# LAUNCHER=single_node bash launch_orca_moe_seqaux_olmo_stage1_ep8.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export TRAINING_SCRIPT=${TRAINING_SCRIPT:-"${SCRIPT_DIR}/orca_moe_seqaux_olmo_stage1_ep8.sh"}
export NGPU=${NGPU:-${GPUS_PER_NODE:-4}}
export EP_SIZE=${EP_SIZE:-4}

exec bash "${REPO_ROOT}/scripts/launchers/launch_experiment.sh" "$@"
