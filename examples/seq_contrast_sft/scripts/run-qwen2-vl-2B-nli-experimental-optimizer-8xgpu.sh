#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Experimental optimizer opt-in; preserves SDK clipping, scheduler and DCP.
set -e
set -o pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export EXP_NAME="${EXP_NAME:-qwen2-vl-2b-nli-experimental-optimizer-v1-dp8}"
exec bash "${SCRIPT_DIR}/run-qwen2-vl-2B-nli-pt-8xgpu.sh" \
    --no-gradient-accumulation-fusion \
    --custom-megatron-before-train-step-hook-path examples.seq_contrast_sft.native_optimizer.before_train \
    "$@"
