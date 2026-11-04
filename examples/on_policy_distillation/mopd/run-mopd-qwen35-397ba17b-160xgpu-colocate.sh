#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 397B MOPD baseline: 20 nodes x 8 GPUs, train and inference share the GPU pool.
# Rollout: five mixed Student engines (5 x 16 GPUs) + five Teachers (5 x 16 GPUs).
# Train: TP4 / PP5 / CP8 / EP32 / ETP1; layers 6/14/14/14/12, full recompute.
# Defaults: GBS160, N=1, 10 rollouts; source weights 1:2:3:4:5.
# Set MODEL_DIR, DATA_DIR and MEGATRON; see README.md for the tested runtime.
# Connect to the existing Ray cluster; do not start or clean up the cluster.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${RELAX_ROOT}"

: "${MODEL_DIR:?Set MODEL_DIR to the shared model root}"
: "${DATA_DIR:?Set DATA_DIR to the shared dataset root}"
: "${MEGATRON:?Set MEGATRON to the same Megatron source path on every node}"
export MODEL_DIR DATA_DIR MEGATRON
export RELAX="${RELAX_ROOT}"
export MODEL_CONFIG_DIR="${RELAX_ROOT}/scripts/models"
export RELAX_ENTRYPOINT_MODE=external
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export PYTHONPATH="${RELAX_ROOT}:${MEGATRON}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
MASTER_ADDR="${MASTER_ADDR:-$(python3 -c 'import ray; print(ray.util.get_node_ip_address())')}"
export MASTER_ADDR

# Torch/NVSHMEM 动态库路径也要传给节点上的训练进程。
TORCH_LIB_PATH="$(python3 -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
NVSHMEM_LIB_PATH="${NVSHMEM_LIB_PATH:-$(dirname "${TORCH_LIB_PATH}")/../nvidia/nvshmem/lib}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${NVSHMEM_LIB_PATH}:${TORCH_LIB_PATH}"
export NCCL_NVLS_ENABLE=0

# Native image OPD 保留 Student 采样的 token IDs；Teacher 对相同图片和回答评分。
export SGLANG_MM_AVOID_RETOKENIZE=true
export SGLANG_ENABLE_LOGPROB_CHUNK=1
# 限制 Teacher 临时 logits 工作区，避免一次生成长序列 × 全词表的大张量。
export SGLANG_LOGPROB_CHUNK_SIZE="${SGLANG_LOGPROB_CHUNK_SIZE:-1024}"
# 视觉编码器也参与训练，不能复用旧权重产生的图片 embedding。
export SGLANG_VLM_CACHE_SIZE_MB=0
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
# 003 runtime: refresh colocate memory descriptors and avoid SGLang RPC ports.
export MC_DISABLE_METACACHE=1
export MC_MIN_RPC_PORT="${MC_MIN_RPC_PORT:-20000}"
export MC_MAX_RPC_PORT="${MC_MAX_RPC_PORT:-22000}"
# 多图的数据传输可能超过默认等待时间；超时增加不代表 I/O 变快。
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT="${TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT:-900}"
export NO_PROXY="${no_proxy:-${NO_PROXY:-}},localhost,127.0.0.1,10.0.0.0/8"
export no_proxy="${NO_PROXY}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}MASTER_ADDR,PYTHONUNBUFFERED,CUDA_DEVICE_MAX_CONNECTIONS,OMP_NUM_THREADS,MKL_NUM_THREADS,OPENBLAS_NUM_THREADS,LD_LIBRARY_PATH,NCCL_NVLS_ENABLE,SGLANG_MM_AVOID_RETOKENIZE,SGLANG_ENABLE_LOGPROB_CHUNK,SGLANG_LOGPROB_CHUNK_SIZE,SGLANG_VLM_CACHE_SIZE_MB,TORCH_DISTRIBUTED_DEFAULT_TIMEOUT,TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT,NO_PROXY,no_proxy,MC_DISABLE_METACACHE,MC_MIN_RPC_PORT,MC_MAX_RPC_PORT,TENSORBOARD_DIR"

# 仅加载仓库自带的模型架构参数，不执行入口脚本的集群清理。
source "${MODEL_CONFIG_DIR}/qwen35-397B-A17B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/recipes/mopd}"
EXP_NAME="${EXP_NAME:-mopd-qwen35-397ba17b-160xgpu-mixed5-cp8-source-mixture}"
now="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log/${EXP_NAME}-${now}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
BATCH_SIZE="${BATCH_SIZE:-160}"
MM_PROCESSOR_POOL_SIZE="${MM_PROCESSOR_POOL_SIZE:-8}"
CONTEXT_LEN=131072
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-$((CONTEXT_LEN - MAX_RESPONSE_LEN))}"
if (( BATCH_SIZE < 5 || NUM_ROLLOUT < 1 || MAX_PROMPT_LEN <= 0 || MAX_RESPONSE_LEN <= 0 ||
      MAX_PROMPT_LEN + MAX_RESPONSE_LEN > CONTEXT_LEN )); then
    echo "Require BS >= 5, NUM_ROLLOUT >= 1 and prompt + response <= 131072." >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"

STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_DIR}/Qwen3.5-397B-A17B}"
TEACHER_1_MODEL="${TEACHER_1_MODEL:-${STUDENT_MODEL}}"
TEACHER_2_MODEL="${TEACHER_2_MODEL:-${STUDENT_MODEL}}"
TEACHER_3_MODEL="${TEACHER_3_MODEL:-${STUDENT_MODEL}}"
TEACHER_4_MODEL="${TEACHER_4_MODEL:-${STUDENT_MODEL}}"
TEACHER_5_MODEL="${TEACHER_5_MODEL:-${STUDENT_MODEL}}"
# Smoke 可共用一个权重目录，但仍启动五个独立 Teacher 引擎；以后逐个替换路径。
TEACHER_ROUTES="$(python3 - "${TEACHER_1_MODEL}" "${TEACHER_2_MODEL}" \
    "${TEACHER_3_MODEL}" "${TEACHER_4_MODEL}" "${TEACHER_5_MODEL}" <<'PY'
import json
import sys

print(json.dumps({f"teacher{i}": path for i, path in enumerate(sys.argv[1:], 1)}))
PY
)"
export SOURCE_1_DATA="${SOURCE_1_DATA:-${DATA_DIR}/MOPD-397B-5teacher-128k/teacher1.jsonl}"
export SOURCE_2_DATA="${SOURCE_2_DATA:-${DATA_DIR}/MOPD-397B-5teacher-128k/teacher2.jsonl}"
export SOURCE_3_DATA="${SOURCE_3_DATA:-${DATA_DIR}/MOPD-397B-5teacher-128k/teacher3.jsonl}"
export SOURCE_4_DATA="${SOURCE_4_DATA:-${DATA_DIR}/MOPD-397B-5teacher-128k/teacher4.jsonl}"
export SOURCE_5_DATA="${SOURCE_5_DATA:-${DATA_DIR}/MOPD-397B-5teacher-128k/teacher5.jsonl}"
# 固定配置放在 recipe 中，可直接 review；启动不生成或覆盖 YAML。
SOURCE_MIXTURE_CONFIG="${SOURCE_MIXTURE_CONFIG:-${SCRIPT_DIR}/source-mixture-397b-5teacher.yaml}"
SGLANG_CONFIG="${SGLANG_CONFIG:-${SCRIPT_DIR}/sglang-397b-mixed5.yaml}"


CKPT_ARGS=(
    --hf-checkpoint "${STUDENT_MODEL}"
    --megatron-to-hf-mode bridge
    # 397B 共享存储上的 mmap 碎片读取曾很慢：逐节点顺序预读 HF shards 到 OS
    # page cache，同节点同路径加锁去重。预热不减少权重大小，也不保证缓存永久驻留。
    --warm-hf-checkpoint-page-cache
)
# 沿用性能 smoke 不保存 checkpoint 的约定。需要保存时显式设 SAVE_INTERVAL>0，
# SAVE_DIR/EXP_NAME 保持稳定用于续训；时间戳仅用于每次日志和 ClearML task。
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
if (( SAVE_INTERVAL > 0 )); then
    : "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
    CKPT_ARGS+=(
        --load "${SAVE_DIR}/${EXP_NAME}"
        --save "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL}"
        --max-actor-ckpt-to-keep 2
    )
fi

ROLLOUT_ARGS=(
    # 与 --prompt-data 互斥；来源名称必须和 Teacher routes 完全一致。
    --rollout-source-mixture "${SOURCE_MIXTURE_CONFIG}"
    --input-key prompt
    --label-key label
    --tool-key tools
    --metadata-key extra_info
    --apply-chat-template
    --multimodal-keys '{"image":"images"}'
    # 每张图的视觉 token 预算，不是每请求总图 token 数；多图仍会累加。
    --image-max-token-num 1024
    --mm-processor-pool-size "${MM_PROCESSOR_POOL_SIZE}"
    --rm-type dummy
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${BATCH_SIZE}"
    --over-sampling-batch-size "${BATCH_SIZE}"
    --global-batch-size "${BATCH_SIZE}"
    --n-samples-per-prompt 1
    --num-iters-per-train-update 1
    # 只做 token 长度打包，不改变固定 Global Batch 的条数和来源配额。
    --balance-data
    --rollout-max-prompt-len "${MAX_PROMPT_LEN}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-max-context-len "${CONTEXT_LEN}"
    --rollout-temperature 1
    --seed 1234
    --rollout-seed 42
    --use-streaming-dataset
    # source-mixture 管理每个来源自己的 shuffle 游标、跨 epoch 循环和同源补齐。
    --rollout-shuffle
    --streaming-buffer-size 64
    --prefetch-chunk-size 8
    --prefetch-max-cached 16
    --prefetch-num-workers 4
    --rollout-result-dir "${LOG_DIR}/rollout_result"
)

OPD_ARGS=(
    --use-opd
    --opd-only-reward
    --opd-type sglang
    --opd-kl-coef 0.3
    --opd-teacher-key data_source
    --opd-teacher-routes "${TEACHER_ROUTES}"
    --teacher-num-gpus-per-engine 16
    --teacher-sglang-ep-size 16
    --teacher-sglang-dp-size 8
    --teacher-sglang-enable-dp-attention
    --teacher-sglang-enable-dp-lm-head
    --teacher-sglang-moe-a2a-backend none
    # 静态权重/KV 池的显存比例；运行时激活、Graph 和工作区仍会额外占用。
    --teacher-sglang-mem-fraction-static 0.8
    # Teacher 需要给完整 prompt+response 评分，额外留 SGLang 输入安全余量。
    --teacher-sglang-context-length "$((CONTEXT_LEN + 256))"
    --teacher-sglang-chunked-prefill-size 8192
    --teacher-sglang-max-prefill-tokens 1024
    --teacher-sglang-mm-process-config '{"image":{"disable_grouping":true}}'
    --teacher-sglang-max-running-requests 8
    --teacher-sglang-max-total-tokens 147456
    --teacher-sglang-max-mamba-cache-size 48
    # Teacher 纯 prefill 评分，保持 eager；使用新版 Graph 参数避免弃用警告。
    --teacher-sglang-cuda-graph-backend-decode disabled
    --teacher-sglang-cuda-graph-backend-prefill disabled
    --teacher-sglang-dist-timeout 3600
    --teacher-sglang-watchdog-timeout 3600
    --teacher-sglang-enable-request-time-stats-logging
    --teacher-sglang-enable-metrics-for-all-schedulers
    --opd-teacher-timeout-s 3600
    --opd-teacher-image-key images
    --opd-log-prob-min-clamp -10.0
    --use-rollout-logprobs
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    # 将 optimizer 状态放到 CPU；代价是主机内存和 CPU/GPU 传输开销。
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --use-distributed-optimizer
    --accumulate-allreduce-grads-in-fp32
    --moe-router-load-balancing-type none
    # 256 MiB buckets for training-to-Student weight synchronization (010 recipe).
    # If temporary GPU memory is tight, reduce this value; no isolated speedup is claimed.
    --update-weight-buffer-size 268435456
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 5
    --context-parallel-size 8
    # Apply gdn-thd-route-reuse.patch to the matching old Megatron runtime on all nodes.
    # This recipe does not install or upgrade dependencies; see README.md.
    --linear-cp-mode chunkwise
    --seq-length "${CONTEXT_LEN}"
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --decoder-first-pipeline-num-layers 6
    --decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS:-12}"
    --vision-dp-when-tp
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-loss-function
    --log-probs-chunk-size 2048
    --calculate-per-token-loss
    --use-dynamic-batch-size
    # CP8 下每卡 16K，合起来容纳 128K；GBS 不改变单个 microbatch 的 token 预算。
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-16384}"
    --moe-flex-dispatcher-backend deepep
    --moe-token-dispatcher-type flex
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 16
    # Five mixed Student engines: Decode Graph enabled, HiCache disabled.
    --sglang-config "${SGLANG_CONFIG}"
    --sglang-ep-size 16
    --sglang-dp-size 8
    --sglang-enable-dp-attention
    --sglang-enable-dp-lm-head
    --sglang-moe-a2a-backend none
    --sglang-load-format dummy
    --sglang-enable-weights-cpu-backup
    # 沿用基线，0.8 不等于单卡峰值硬上限；78 GiB 目标仍需实测验收。
    --sglang-mem-fraction-static 0.8
    --sglang-context-length "${CONTEXT_LEN}"
    # DP8 后有效 prefill chunk 为 1024；max-running-requests 按 DP 分摊。
    --sglang-chunked-prefill-size 8192
    --sglang-max-prefill-tokens 1024
    --sglang-mm-process-config '{"image":{"disable_grouping":true}}'
    --sglang-max-running-requests 8
    --sglang-server-concurrency 8
    # 每个 Attention DP worker 的 KV token 容量，TP2两卡共同承载；
    # 不是八个 DP worker 加起来147456，也不是图片token预算。
    --sglang-max-total-tokens 147456
    # DP8 分摊后每 worker 六个状态槽；prefill overlap 单请求需五个。
    --sglang-max-mamba-cache-size 48
    --sglang-cuda-graph-backend-prefill disabled
    --sglang-router-policy cache_aware
    # Let the existing relative-load threshold trigger Student rebalancing
    # without first requiring an absolute queue gap of four requests.
    # This does not guarantee equal queues or eliminate all rollout tails.
    --sglang-router-balance-abs-threshold 0
    --rollout-engine-init-timeout 3600
    --sglang-dist-timeout 3600
    --sglang-watchdog-timeout 3600
    --sglang-enable-request-time-stats-logging
    --sglang-enable-metrics-for-all-schedulers
)

TRACKING_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --skip-eval-before-train
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --no-rope-fusion
)

# actor 与 rollout/teacher 共用一个160卡池，不能把160+80+80算作320张物理卡。
# 八个 TQ storage units 分摊多图 batch 的接收内存，但不消除图片张量本身。
# PP0 pixel routing and host sharing were validated at GBS160 in experiment 008.
python3 -m relax.entrypoints.train \
    --resource '{"actor":[1,160],"rollout":[1,80],"teacher":[1,80]}' \
    --rollout-num-gpus 80 \
    --max-staleness 0 \
    --num-data-storage-units 8 \
    --colocate \
    --offload \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${MISC_ARGS[@]}" "$@" 2>&1 | tee "${LOG_DIR}/${EXP_NAME}-${now}.log"
