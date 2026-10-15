#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Invoke through scripts/entrypoint/ray-job.sh on an available Ray cluster.
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to a root containing the prepared native base directory}"
: "${DATA_DIR:?Set DATA_DIR to a root containing the prepared nli directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root outside the code working directory}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the verified Ray Jobs HTTP address}"
: "${RELAX_ENTRYPOINT_MODE:?Launch through scripts/entrypoint/ray-job.sh}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/lamra}"
EXP_NAME="${EXP_NAME:-qwen2-vl-2b-nli-pt-8xgpu-r64-seed42}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1078}"
now="$(date +%Y%m%d-%H%M%S)"
mkdir -p "${SAVE_DIR}/${EXP_NAME}/logs"

cp "${DATA_DIR}/nli/data_audit.json" "${SAVE_DIR}/${EXP_NAME}/data_audit.json"
python3 - "${MODEL_DIR}" "${SAVE_DIR}/${EXP_NAME}" <<'PY'
import json
import pathlib
import sys

model, save = map(pathlib.Path, sys.argv[1:])
config = {
    "lamra_processor_path": str(model / "base"),
    "lamra_audit_path": str(save / "data_audit.json"),
    "lamra_expected_queries": 275601,
    "lamra_forward_subbatch_size": 32,
    "lamra_temperature": 0.05,
}
(save / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
PY

MODEL_ARGS=(
    --train-backend fsdp --hf-checkpoint "${MODEL_DIR}/base"
    --skip-hf-validate
    --model-adapter-path relax.models.lamra.model.load_retrieval_model
    --fsdp-trainable-mode lora --fsdp-master-dtype fp32 --fsdp-param-dtype bf16 --fsdp-reduce-dtype fp32
    --lora-rank 64 --lora-alpha 128 --lora-dropout 0.05
    --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --fsdp-activation-checkpointing
)
CKPT_ARGS=(
    --save "${SAVE_DIR}/${EXP_NAME}" --load "${SAVE_DIR}/${EXP_NAME}"
    --megatron-to-hf-mode bridge --save-interval 200 --max-actor-ckpt-to-keep 2
    --num-rollout "${NUM_ROLLOUT}" --num-epoch 2
    --custom-config-path "${SAVE_DIR}/${EXP_NAME}/run_config.json"
)
SFT_ARGS=(
    --loss-type sft --custom-dataset-class relax.models.lamra.dataset.NLIDataset
    --prompt-data "${DATA_DIR}/nli/nli_cn.jsonl"
    --global-batch-size 512 --micro-batch-size 64 --n-samples-per-prompt 1
    --use-dynamic-batch-size --max-tokens-per-gpu 32768 --seq-length 1024
    --multimodal-keys '{"image":"image"}' --image-min-token-num 4 --image-max-token-num 300
    --sft-oversize-strategy keep --sft-invalid-multimodal-strategy error
    --sft-max-in-flight-steps 2 --num-data-storage-units 8
)
OPTIMIZER_ARGS=(
    --optimizer adam --lr 2e-4 --lr-decay-style cosine --lr-warmup-iters 33
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --weight-decay 0.0 --clip-grad 1.0
)
PERF_ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --use-gloo-process-groups --seed 42 --bf16 --use-health-check --max-global-restart 0
)
TRACKING_ARGS=(
    --use-clearml --use-metrics-service --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

cd "${RELAX_ROOT}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train --resource '{"sft":[1,0],"actor":[1,8]}' \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" "${TRACKING_ARGS[@]}" "$@" \
    2>&1 | tee "${SAVE_DIR}/${EXP_NAME}/logs/launch-${now}.log"
