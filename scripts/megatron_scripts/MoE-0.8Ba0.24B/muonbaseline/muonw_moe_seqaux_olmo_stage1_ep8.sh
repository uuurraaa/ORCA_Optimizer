#!/usr/bin/env bash

# Muon baseline training for the 0.8B-total / 0.24B-active Lite MoE model.
#
# This is the node-local experiment script. The outer launcher decides whether
# to run it once locally or once per node through MPI.

set -euo pipefail

LAUNCH_NNODES=${NNODES:-}
LAUNCH_NODE_RANK=${NODE_RANK:-}
LAUNCH_MASTER_ADDR=${MASTER_ADDR:-}
LAUNCH_MASTER_PORT=${MASTER_PORT:-}
LAUNCH_GPUS_PER_NODE=${GPUS_PER_NODE:-${NGPU:-}}
LAUNCH_EP_SIZE=${EP_SIZE:-}
LAUNCH_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

CLUSTER_ENV=${CLUSTER_ENV:-/share/liuyuanshi/liuyuanshi/env/env_cluster.sh}
source "${CLUSTER_ENV}"

if [[ -n "${LAUNCH_CUDA_VISIBLE_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES=${LAUNCH_CUDA_VISIBLE_DEVICES}
fi

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

GPUS_PER_NODE=${LAUNCH_GPUS_PER_NODE:-4}
NNODES=${OMPI_COMM_WORLD_SIZE:-${LAUNCH_NNODES:-${NNODES:-1}}}
NODE_RANK=${OMPI_COMM_WORLD_RANK:-${LAUNCH_NODE_RANK:-${NODE_RANK:-0}}}
MASTER_ADDR=${MPI_MASTER_ADDR:-${LAUNCH_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}}
MASTER_PORT=${MPI_MASTER_PORT:-${LAUNCH_MASTER_PORT:-${MASTER_PORT:-6000}}}
LOG_RANK=${LOG_RANK:-}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${LAUNCH_EP_SIZE:-4}

TRAIN_ITER=${TRAIN_ITER:-50000}
WSD_DECAY_ITERS=${WSD_DECAY_ITERS:-$((TRAIN_ITER * 15 / 100))}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-128}
GLOBAL_BATCH=${GLOBAL_BATCH:-512}
SEQ_LENGTH=${SEQ_LENGTH:-1024}

DATA_ROOT=${DATA_ROOT:-/share/liuyuanshi/public_datasets/olmo_stage1_400b/megatron_indexed_a800}

REPO_PATH=${REPO_PATH:-results/optimizer_arena_v2/MoE_0.8Ba0.24B}
JOB_NAME_BASE=${JOB_NAME:-muonbaseline_moe0.8Ba0.24B_seqaux_olmo_stage1_ep4_mbs128_seq1024_gbs512}
JOB_NAME="${JOB_NAME_BASE}_nodes${NNODES}"
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/share/liuyuanshi/liuyuanshi/checkpoints/orth_megatron}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"${CHECKPOINT_ROOT}/${JOB_NAME}"}
DATA_PATH_CACHE=${DATA_PATH_CACHE:-"${REPO_PATH}/data_cache/${JOB_NAME}"}
TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"
LOG_DIR="${REPO_PATH}/logs"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.node-${NODE_RANK}.log"

NSYS_OUTPUT_DIR=/share/liuyuanshi/liuyuanshi/temp/nsystest
NSYS_PROFILE_NODE_RANK=0
NSYS_PROFILE_LOCAL_RANK=1
NSYS_PROFILE_GLOBAL_RANK=$((NSYS_PROFILE_NODE_RANK * GPUS_PER_NODE + NSYS_PROFILE_LOCAL_RANK))
NSYS_ARGS=(
    --sample=none
    --trace=cuda,nvtx
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --force-overwrite=true
    --output="${NSYS_OUTPUT_DIR}/${JOB_NAME}.node-${NSYS_PROFILE_NODE_RANK}.rank-${NSYS_PROFILE_GLOBAL_RANK}"
)

SAVE_INTERVAL=${SAVE_INTERVAL:-500}
SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-5000}
NUM_DATASET_BUILDER_THREADS=${NUM_DATASET_BUILDER_THREADS:-128}
# This is per data-loading rank. With the current TP=PP=1 eight-GPU layout,
# that means 8 * NUM_WORKERS processes. Four workers is a conservative starting
# point for multi-node runs; tune it from batch-generator timing.
NUM_WORKERS=${NUM_WORKERS:-4}

# Performance controls.
MMAP_BIN_FILES=${MMAP_BIN_FILES:-1}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-1}
DETAILED_LOGGING=${DETAILED_LOGGING:-1}
TIMING_LOG_LEVEL=${TIMING_LOG_LEVEL:-1}

for boolean_name in MMAP_BIN_FILES OVERLAP_GRAD_REDUCE DETAILED_LOGGING; do
    [[ "${!boolean_name}" == 0 || "${!boolean_name}" == 1 ]] || {
        echo "ERROR: ${boolean_name} must be 0 or 1, got '${!boolean_name}'" >&2
        exit 1
    }
done

[[ "${TIMING_LOG_LEVEL}" =~ ^[0-2]$ ]] || {
    echo "ERROR: TIMING_LOG_LEVEL must be 0, 1, or 2, got '${TIMING_LOG_LEVEL}'" >&2
    exit 1
}

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
MODEL_PARALLEL_SIZE=$((TP_SIZE * PP_SIZE))
if (( WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
    echo "ERROR: world size ${WORLD_SIZE} is not divisible by TP*PP=${MODEL_PARALLEL_SIZE}" >&2
    exit 1
fi
DATA_PARALLEL_SIZE=$((WORLD_SIZE / MODEL_PARALLEL_SIZE))
SAMPLES_PER_MICROBATCH_WAVE=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))
if ((
    GLOBAL_BATCH < SAMPLES_PER_MICROBATCH_WAVE
    || GLOBAL_BATCH % SAMPLES_PER_MICROBATCH_WAVE != 0
)); then
    echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} must be a positive multiple of " \
         "MICRO_BATCH_SIZE*DP=${MICRO_BATCH_SIZE}*${DATA_PARALLEL_SIZE}=" \
         "${SAMPLES_PER_MICROBATCH_WAVE}" >&2
    exit 1
fi
NUM_MICROBATCHES=$((GLOBAL_BATCH / SAMPLES_PER_MICROBATCH_WAVE))
TOKENS_PER_STEP=$((GLOBAL_BATCH * SEQ_LENGTH))

PRETRAIN_SCRIPT="${REPO_ROOT}/pretrain_gpt.py"
TOKENIZER_MODEL="${REPO_ROOT}/models/OLMo-2-1124-7B"

mkdir -p \
    "${TENSORBOARD_PATH}" \
    "${WANDB_PATH}" \
    "${DATA_PATH_CACHE}" \
    "${CHECKPOINT_PATH}" \
    "${LOG_DIR}"

# Keep completed IndexedDataset parts as equal-weight prefixes, matching the
# existing Muon experiment scripts. A published .done marker means the
# matching bin/idx pair is ready.
DATA_PATH=()
while IFS= read -r done_file; do
    DATA_PATH+=(1 "${done_file%.done}")
done < <(find "${DATA_ROOT}" -type f -name '*_text_document.done' | sort)

DATA_ARGS=(
    --tokenizer-model "${TOKENIZER_MODEL}"
    --tokenizer-type HuggingFaceTokenizer
    --data-path "${DATA_PATH[@]}"
    --data-cache-path "${DATA_PATH_CACHE}"
    --train-iters "${TRAIN_ITER}"
    --split 99,1,0
    --num-dataset-builder-threads "${NUM_DATASET_BUILDER_THREADS}"
    --num-workers "${NUM_WORKERS}"
    --distributed-timeout-minutes 60
)

if [[ "${MMAP_BIN_FILES}" == 0 ]]; then
    DATA_ARGS+=(--no-mmap-bin-files)
fi

TRAINING_ARGS=(
    --lr 1e-3
    --lr-warmup-iters 2000
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-wsd-decay-iters "${WSD_DECAY_ITERS}"
    --min-lr 1e-4
    --lr-decay-iters "${TRAIN_ITER}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --optimizer dist_muon
    --muon-momentum 0.95
    --muon-use-nesterov
    --muon-scale-mode align_adamw_rms
    --muon-num-ns-steps 8
    --muon-qkv-split-mode component
    --weight-decay 0.1
    --no-weight-decay-cond-type all_params
    --clip-grad 1.0
    --profile
    --profile-step-start 2040
    --profile-step-end 2050
    --profile-ranks "${NSYS_PROFILE_GLOBAL_RANK}"
)

if [[ "${OVERLAP_GRAD_REDUCE}" == 1 ]]; then
    TRAINING_ARGS+=(--overlap-grad-reduce)
fi

MODEL_ARGS=(
    --num-layers 16
    --hidden-size 768
    --ffn-hidden-size 1536
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-topk 4
    --moe-router-dtype fp32
    --num-experts 64
    --moe-ffn-hidden-size 288
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-router-topk-scaling-factor 2
    --moe-router-enable-expert-bias
    --moe-router-score-function sigmoid
    --moe-router-bias-update-rate 0.001
    --moe-layer-freq '([0]*1+[1]*15)'
    --moe-shared-expert-intermediate-size 288
    --moe-shared-expert-overlap
    --moe-shared-expert-gate
    --moe-grouped-gemm
    --group-query-attention
    --num-attention-heads 6
    --num-query-groups 3
    --norm-epsilon 1e-6
    --kv-channels 128
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings 40960
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --disable-bias-linear
    --transformer-impl transformer_engine
    --attention-backend fused
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --init-method-std 0.02
    --split-qkv-init-mode component
    --use-cpu-initialization
)

# Megatron writes latest_checkpointed_iteration.txt after each completed save.
# Passing --load makes a later tidal job pick up that latest checkpoint.
LOAD_ARGS=()
if [[ -f "${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" ]]; then
    LOAD_ARGS=(
        --load "${CHECKPOINT_PATH}"
        --exit-on-missing-checkpoint
    )
fi

CKPT_ARGS=(
    --ckpt-format torch_dist
    --save-interval "${SAVE_INTERVAL}"
    --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
    --async-save
    --exit-signal-handler
    --save "${CHECKPOINT_PATH}"
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH}"
)

LOGGER_ARGS=(
    --log-throughput
    --log-interval 10
    --log-weight-matrix-norms
    --log-validation-ppl-to-tensorboard
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --timing-log-level "${TIMING_LOG_LEVEL}"
    --tensorboard-dir "${TENSORBOARD_PATH}"
)

if [[ "${DETAILED_LOGGING}" == 1 ]]; then
    LOGGER_ARGS+=(
        --log-params-norm
        --log-hidden-states embeddings input_layernorm attention::linear_qkv attention::linear_q attention::linear_k attention::linear_v attention::core_attention attention::o_proj pre_mlp_layernorm mlp
        --log-params attention::linear_qkv attention::o_proj mlp::linear_fc1 mlp::linear_fc2 input_layernorm pre_mlp_layernorm embedding lm_head
    )
fi

WANDB_ARGS=(
    --wandb-project optimizer_arena_v2
    --wandb-exp-name "${JOB_NAME}"
    --wandb-save-dir "${WANDB_PATH}"
)

if [[ "${NNODES}" -gt 1 ]]; then
    # Let c10d assign the node/global ranks. This avoids coupling torchrun's
    # rank map to Open MPI's hostfile ordering.
    RDZV_ID=${RDZV_ID:-megatron_${JOB_NAME}_${MASTER_PORT}}
    DISTRIBUTED_LAUNCH_MODE=c10d
    DISTRIBUTED_ARGS=(
        --nproc_per_node "${GPUS_PER_NODE}"
        --nnodes "${NNODES}"
        --rdzv_id "${RDZV_ID}"
        --rdzv_backend c10d
        --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"
    )
else
    DISTRIBUTED_LAUNCH_MODE=static
    DISTRIBUTED_ARGS=(
        --nproc_per_node "${GPUS_PER_NODE}"
        --nnodes "${NNODES}"
        --node_rank "${NODE_RANK}"
        --master_addr "${MASTER_ADDR}"
        --master_port "${MASTER_PORT}"
    )
fi

# Avoid merging every local worker's stdout into the benchmark log when the
# caller requests rank filtering.  This is intentionally opt-in because some
# debugging runs need all ranks.
TORCHRUN_OUTPUT_ARGS=()
if [[ -n "${LOG_RANK}" ]]; then
    TORCHRUN_OUTPUT_ARGS=(--local-ranks-filter="${LOG_RANK}" --tee=3)
fi

NSYS_CMD=()
if [[ "${NODE_RANK}" == "${NSYS_PROFILE_NODE_RANK}" ]]; then
    command -v nsys >/dev/null 2>&1 || {
        echo "ERROR: nsys not found on node ${NODE_RANK}" >&2
        exit 1
    }
    mkdir -p "${NSYS_OUTPUT_DIR}"
    NSYS_CMD=(nsys profile "${NSYS_ARGS[@]}")
fi

echo "Launch: mode=${DISTRIBUTED_LAUNCH_MODE} nnodes=${NNODES} gpus_per_node=${GPUS_PER_NODE} mpi_rank=${NODE_RANK} tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE}"
echo "Batch: micro=${MICRO_BATCH_SIZE} global=${GLOBAL_BATCH} seq=${SEQ_LENGTH} dp=${DATA_PARALLEL_SIZE} accumulation=${NUM_MICROBATCHES} tokens_per_step=${TOKENS_PER_STEP}"
echo "Performance: mmap=${MMAP_BIN_FILES} grad_accum_fusion=1 overlap_grad_reduce=${OVERLAP_GRAD_REDUCE} num_workers=${NUM_WORKERS} detailed_logging=${DETAILED_LOGGING} cuda_device_max_connections=${CUDA_DEVICE_MAX_CONNECTIONS}"
if [[ "${NODE_RANK}" == "${NSYS_PROFILE_NODE_RANK}" ]]; then
    echo "Nsys: global_rank=${NSYS_PROFILE_GLOBAL_RANK} steps=2040-2050 output=${NSYS_OUTPUT_DIR}"
fi
if [[ -n "${LOG_RANK}" ]]; then
    echo "Torchrun output ranks: ${LOG_RANK}"
fi

if [[ "${TORCH_DISTRIBUTED_DEBUG:-OFF}" == DETAIL ]]; then
    echo "WARNING: TORCH_DISTRIBUTED_DEBUG=DETAIL adds collective checks and can lower MFU." >&2
fi
if [[ "${TORCHDYNAMO_DISABLE:-0}" == 1 || "${TORCH_COMPILE_DISABLE:-0}" == 1 ]]; then
    echo "WARNING: torch.compile appears disabled; this can lower MFU even when GPU utilization is 100%." >&2
fi

"${NSYS_CMD[@]}" torchrun \
    "${DISTRIBUTED_ARGS[@]}" \
    "${TORCHRUN_OUTPUT_ARGS[@]}" \
    "${PRETRAIN_SCRIPT}" \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LOAD_ARGS[@]}" \
    "${LOGGER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    --eval-interval 1000 \
    --eval-iters 60 \
    "$@" \
    2>&1 | tee -a "${LOG_FILE}"
