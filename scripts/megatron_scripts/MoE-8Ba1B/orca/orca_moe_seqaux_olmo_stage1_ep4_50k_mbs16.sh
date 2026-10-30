#!/usr/bin/env bash

# ORCA trained from scratch for 50k iterations with micro-batch 16.
#
# Keep the full training recipe aligned with the Muon baseline and
# override only the optimizer plus the ORCA-specific hyperparameters.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUON_BASELINE="${SCRIPT_DIR}/../muonbaseline/muonw_moe_seqaux_olmo_stage1_ep4_50k_mbs16.sh"

export JOB_NAME=${JOB_NAME:-orca_moe_seqaux_olmo_stage1_ep4_50k_mbs16_v2}
export TRAIN_ITER=${TRAIN_ITER:-50000}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
export TIMING_LOG_LEVEL=${TIMING_LOG_LEVEL:-1}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}

exec bash "${MUON_BASELINE}" \
    --optimizer orca \
    --orca-alpha1 0.25 \
    --orca-c 0.1 \
    --orca-owdwd 1 \
    --orca-wd-schedule stable_linear \
    --orca-wd-stable-ratio 1 \
    --orca-wd-end-scale 0 \
    "$@"
