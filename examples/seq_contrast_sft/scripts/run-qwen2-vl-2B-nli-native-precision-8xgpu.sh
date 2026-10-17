#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Opt-in measured forward/backward profile; original optimizer math is unchanged.
set -e
set -o pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export EXP_NAME="${EXP_NAME:-qwen2-vl-2b-nli-native-precision-v1-dp8}"
exec bash "${SCRIPT_DIR}/run-qwen2-vl-2B-nli-pt-8xgpu.sh" \
    --no-gradient-accumulation-fusion \
    --custom-megatron-before-train-step-hook-path examples.seq_contrast_sft.native_precision.before_train \
    "$@"
