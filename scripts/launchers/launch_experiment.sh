#!/usr/bin/env bash

# Run one node-local training script on this machine, or once per node through MPI.
# Set TRAINING_SCRIPT to that script. An existing hostfile selects MPI unless
# LAUNCHER=single_node or LAUNCHER=multi_node_mpi is set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

[[ -n "${TRAINING_SCRIPT:-}" ]] || {
    echo "ERROR: set TRAINING_SCRIPT to the node-local training script" >&2
    exit 1
}

MPI_LAUNCHER="${REPO_ROOT}/scripts/launchers/launch_multi_node_mpi.sh"
HOSTFILE=${HOSTFILE:-/etc/mpi/hostfile}

if [[ -z "${LAUNCHER:-}" ]]; then
    if [[ -f "${HOSTFILE}" ]]; then
        LAUNCHER=multi_node_mpi
    else
        LAUNCHER=single_node
    fi
fi

case "${LAUNCHER}" in
    single_node)
        exec env \
            "NNODES=1" \
            "NODE_RANK=0" \
            "MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}" \
            bash "${TRAINING_SCRIPT}" "$@"
        ;;
    multi_node_mpi)
        exec env \
            "EXPERIMENT_SCRIPT=${TRAINING_SCRIPT}" \
            "HOSTFILE=${HOSTFILE}" \
            bash "${MPI_LAUNCHER}" "$@"
        ;;
    *)
        echo "ERROR: unsupported LAUNCHER '${LAUNCHER}'; expected single_node or multi_node_mpi" >&2
        exit 1
        ;;
esac
