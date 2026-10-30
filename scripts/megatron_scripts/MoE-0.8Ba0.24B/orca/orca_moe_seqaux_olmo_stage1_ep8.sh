#!/usr/bin/env bash

# ORCA training for the 0.8B-total / 0.24B-active Lite MoE model.
#
# Keep the full training recipe aligned with the Muon baseline and
# override only the optimizer plus the ORCA-specific LLaMA-1B
# hyperparameters from lowcopts-min.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUON_BASELINE="${SCRIPT_DIR}/../muonbaseline/muonw_moe_seqaux_olmo_stage1_ep8.sh"

export JOB_NAME=${JOB_NAME:-orca_moe0.8Ba0.24B_seqaux_olmo_stage1_ep4_mbs128_seq1024_gbs512}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-128}
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
