#!/usr/bin/env bash

# Launch exactly one complete experiment-script instance on each node.
# The experiment script is responsible for node-local setup and torchrun.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

HOSTFILE=${HOSTFILE:-/etc/mpi/hostfile}
NGPU=${NGPU:-${GPUS_PER_NODE:-8}}
MASTER_PORT=${MASTER_PORT:-6000}
MY_NODE_IP=${MY_NODE_IP:-${MASTER_ADDR:-}}
EXPERIMENT_SCRIPT=${EXPERIMENT_SCRIPT:-}
MPIRUN=${MPIRUN:-mpirun}

# Keep the same NCCL/IB defaults as the known-good lowcopts-min launcher.
# These are overridable for clusters with a different HCA or RoCE setup.
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-20}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"
TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-"${CUDA_HOME}/bin/ptxas"}
PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

[[ -f "${HOSTFILE}" ]] || {
    echo "ERROR: MPI hostfile not found: ${HOSTFILE}" >&2
    exit 1
}
[[ -n "${EXPERIMENT_SCRIPT}" ]] || {
    echo "ERROR: set EXPERIMENT_SCRIPT to the complete node-local training script" >&2
    exit 1
}
[[ -f "${EXPERIMENT_SCRIPT}" ]] || {
    echo "ERROR: experiment script not found: ${EXPERIMENT_SCRIPT}" >&2
    exit 1
}
[[ -n "${MY_NODE_IP}" ]] || {
    echo "ERROR: set MY_NODE_IP to the rank-0 node IP" >&2
    exit 1
}

if [[ -z "${NNODES:-}" ]]; then
    NNODES="$(awk '!/^[[:space:]]*(#|$)/ { count++ } END { print count + 0 }' "${HOSTFILE}")"
fi
[[ "${NNODES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: NNODES must be a positive integer, got '${NNODES}'" >&2
    exit 1
}

DUPLICATE_HOSTS="$(awk '!/^[[:space:]]*(#|$)/ { print $1 }' "${HOSTFILE}" | sort | uniq -d)"
[[ -z "${DUPLICATE_HOSTS}" ]] || {
    echo "ERROR: MPI hostfile contains duplicate node entries:" >&2
    while IFS= read -r duplicate_host; do
        echo "  ${duplicate_host}" >&2
    done <<< "${DUPLICATE_HOSTS}"
    exit 1
}

if [[ -z "${TCP_NIC:-}" ]]; then
    host_ip="$(hostname -i | awk '{ print $1 }')"
    if command -v ip >/dev/null 2>&1; then
        TCP_NIC="$(ip -o -4 addr show | awk -v target="${host_ip}" '
            { split($4, address, "/") }
            address[1] == target { print $2; exit }
        ')"
    elif command -v ifconfig >/dev/null 2>&1; then
        TCP_NIC="$(ifconfig | awk -v target="${host_ip}" '
            /^[^[:space:]]/ { interface=$1; sub(":$", "", interface) }
            $0 ~ "inet " target "([[:space:]]|$)" { print interface; exit }
        ')"
    fi
fi
[[ -n "${TCP_NIC:-}" ]] || {
    echo "ERROR: could not detect the TCP interface; set TCP_NIC explicitly" >&2
    exit 1
}

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    command -v "${MPIRUN}" >/dev/null 2>&1 || {
        echo "ERROR: mpirun not found: ${MPIRUN}" >&2
        exit 1
    }
fi

MPI_EXPORT_ARGS=(
    -x "PATH=${PATH}"
    -x "CUDA_HOME=${CUDA_HOME}"
    -x "CPATH=${CPATH}"
    -x "TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH}"
    -x "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}"
    -x "OMPI_MCA_btl_tcp_if_include=${TCP_NIC}"
    -x "OMPI_MCA_oob_tcp_if_include=${TCP_NIC}"
    -x "NNODES=${NNODES}"
    -x "MASTER_ADDR=${MY_NODE_IP}"
    -x "MASTER_PORT=${MASTER_PORT}"
    -x "MPI_MASTER_ADDR=${MY_NODE_IP}"
    -x "MPI_MASTER_PORT=${MASTER_PORT}"
    -x "GPUS_PER_NODE=${NGPU}"
    -x "NGPU=${NGPU}"
    -x "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
    -x "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
    -x "NCCL_IB_HCA=${NCCL_IB_HCA}"
    -x "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION}"
    -x "NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT}"
    -x "NCCL_DEBUG=${NCCL_DEBUG}"
    -x "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
)

for optional_name in \
    JOB_NAME \
    CHECKPOINT_ROOT \
    CHECKPOINT_PATH \
    SOURCE_JOB_NAME \
    SOURCE_CHECKPOINT_PATH \
    RESUME_STEP \
    MICRO_BATCH_SIZE \
    GLOBAL_BATCH \
    SEQ_LENGTH \
    TP_SIZE \
    PP_SIZE \
    EP_SIZE \
    CUDA_VISIBLE_DEVICES \
    TIMING_LOG_LEVEL \
    SAVE_INTERVAL \
    SAVE_CHECKPOINT \
    AUTO_RESUME \
    TRAIN_ITER \
    WSD_DECAY_ITERS \
    RDZV_ID \
    LD_LIBRARY_PATH \
    PYTHONPATH \
    NCCL_DEBUG_SUBSYS \
    NCCL_SOCKET_IFNAME \
    GLOO_SOCKET_IFNAME \
    TORCH_CPP_LOG_LEVEL \
    TORCH_DISTRIBUTED_DEBUG; do
    if [[ -n "${!optional_name:-}" ]]; then
        MPI_EXPORT_ARGS+=(-x "${optional_name}=${!optional_name}")
    fi
done

COMMAND=(
    "${MPIRUN}"
    --allow-run-as-root
    --hostfile "${HOSTFILE}"
    --np "${NNODES}"
    --bind-to none
    --map-by slot
    --mca btl self,tcp
    --mca pml ob1
    --mca plm_rsh_num_concurrent 600
    --mca routed_radix 600
    --mca btl_tcp_if_include "${TCP_NIC}"
    --mca oob_tcp_if_include "${TCP_NIC}"
    --mca btl_openib_allow_ib false
    --mca opal_set_max_sys_limits 1
    "${MPI_EXPORT_ARGS[@]}"
    -pernode
    -wdir "${REPO_ROOT}"
    bash "${EXPERIMENT_SCRIPT}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
else
    exec "${COMMAND[@]}"
fi
