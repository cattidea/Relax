#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full-weight baseline: WEIGHT_SOURCE=disk (default) or memory; never ADP.
# Five logical Teachers share one TP1 engine. NUM_GPUS=2 (default) or 8.
# This file also defines the common recipe used by the ADP entry, so training
# options stay identical. Run from any directory; --dry-run never touches Ray.
# Inputs default to Relax/local/mopd/qwen35-9b; outputs to Relax/outputs/mopd.
# Required for execution: RAY_ADDRESS and RAY_DASHBOARD_ADDRESS.
# Optional: MOPD_INPUT_DIR, MODEL_DIR, STUDENT_MODEL, PROMPT_SET, SAVE_DIR,
# SHARED_TEACHER_MANIFEST, EXP_NAME, NUM_ROLLOUT, NUM_GPUS. See the MOPD README.
# Dataset/model downloads are never performed. Disk reads may hit OS page cache.

run_mopd() {
WEIGHT_SOURCE="$1"
shift
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h) sed -n '3,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown argument: $1 (use environment overrides or --dry-run)" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"
MOPD_INPUT_DIR="${MOPD_INPUT_DIR:-${RELAX_ROOT}/local/mopd/qwen35-9b}"
MODEL_DIR="${MODEL_DIR:-${MOPD_INPUT_DIR}/models}"
DATA_DIR="${DATA_DIR:-${MOPD_INPUT_DIR}}"
SAVE_DIR="${SAVE_DIR:-${RELAX_ROOT}/outputs/mopd}"
SHARED_TEACHER_MANIFEST="${SHARED_TEACHER_MANIFEST:-${MOPD_INPUT_DIR}/teacher_manifest.${WEIGHT_SOURCE}.json}"
case "${WEIGHT_SOURCE}" in disk|memory|adp) ;; *) echo "WEIGHT_SOURCE must be disk, memory or adp" >&2; exit 2 ;; esac
if [[ "${WEIGHT_SOURCE}" == adp && "${TEACHER_MAX_RUNNING_REQUESTS:-1}" != 1 ]]; then
    echo "The current ADP acceptance contract requires TEACHER_MAX_RUNNING_REQUESTS=1" >&2
    exit 2
fi
case "${DRY_RUN}" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
# Keep rollout, training TP/DP and checkpoint defaults in one supported profile.
NUM_GPUS="${NUM_GPUS:-2}"
case "${NUM_GPUS}" in
    2) ACTOR_TP=2 ;;
    8) ACTOR_TP=4 ;;
    *) echo "NUM_GPUS must be 2 or 8" >&2; exit 2 ;;
esac
ACTOR_DP=$((NUM_GPUS / ACTOR_TP))
ROLLOUT_GPUS=$((NUM_GPUS - 1))
export NUM_GPUS ACTOR_DP
PROJECT_NAME="${PROJECT_NAME:-Relax/recipes/mopd}"
EXP_NAME="${EXP_NAME:-qwen35-9b-5teacher-${WEIGHT_SOURCE}-${NUM_GPUS}xgpu-colocate}"
now="$(date '+%Y%m%d-%H%M%S')-$$"
STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_DIR}/Qwen3.5-9B}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/train.jsonl}"
SOURCE_TEACHER_MANIFEST="${SHARED_TEACHER_MANIFEST}"
# Resolve launch artifacts outside any Ray-uploaded working directory. Their
# timestamp never enters the checkpoint path used by --load and --save.
SAVE_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${SAVE_DIR}")"
LAUNCH_DIR="${SAVE_DIR}/${EXP_NAME}/runs/${now}"
SHARED_TEACHER_MANIFEST="${LAUNCH_DIR}/teacher_manifest.json"
CUSTOM_CONFIG_PATH="${LAUNCH_DIR}/shared_teacher.yaml"

export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
# This recipe never automatically stops other jobs, workers or placement groups.
# When using the external entrypoint, set this before invoking that entrypoint.
export RELAX_SKIP_GLOBAL_CLEANUP=1
export RELAX="${RELAX_ROOT}"
export PYTHONPATH="${RELAX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export SGLANG_MM_AVOID_RETOKENIZE="${SGLANG_MM_AVOID_RETOKENIZE:-true}"
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE="${SGLANG_LOGITS_PROCESSER_CHUNK_SIZE:-1024}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,NCCL_NVLS_ENABLE,SGLANG_MM_AVOID_RETOKENIZE,SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK,SGLANG_LOGITS_PROCESSER_CHUNK_SIZE,SHARED_TEACHER_MANIFEST"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS},RELAX_BASELINE_SOURCE_SHA256,PYTHONSAFEPATH,PYTHONDONTWRITEBYTECODE"
RELAX_BASELINE_SOURCE_SHA256=$(python3 - "${RELAX_ROOT}" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / "examples/on_policy_distillation/mopd/verified_baseline_entrypoint.py"
spec = importlib.util.spec_from_file_location("verified_baseline_entrypoint", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in module.SOURCES}))
PY
)
export RELAX_BASELINE_SOURCE_SHA256
export DRY_RUN WEIGHT_SOURCE MODEL_DIR DATA_DIR SAVE_DIR STUDENT_MODEL PROMPT_SET
export SOURCE_TEACHER_MANIFEST SHARED_TEACHER_MANIFEST CUSTOM_CONFIG_PATH LAUNCH_DIR EXP_NAME PROJECT_NAME

if [[ "${DRY_RUN}" != 1 ]]; then
    : "${RAY_ADDRESS:?Set RAY_ADDRESS to the existing Ray GCS address}"
    : "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the existing Ray Jobs HTTP origin}"
fi
source "${RELAX_ROOT}/scripts/models/qwen35-9B.sh"
# The shared template contains a legacy flag absent from the installed Megatron
# parser. Keep its supported --attention-output-gate; Bridge reads the Qwen3.5
# hybrid attention architecture from the HF config.
BASELINE_MODEL_ARGS=()
for model_arg in "${MODEL_ARGS[@]}"; do
    if [[ "${model_arg}" != --use-gated-attention ]]; then
        BASELINE_MODEL_ARGS+=("${model_arg}")
    fi
done
MODEL_ARGS=("${BASELINE_MODEL_ARGS[@]}")

export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-20}" N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-40}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}" MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-4096}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}" MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export USE_CLEARML="${USE_CLEARML:-0}" USE_METRICS_SERVICE="${USE_METRICS_SERVICE:-0}"
export SHARED_TEACHER_MAX_BATCH_SIZE="${SHARED_TEACHER_MAX_BATCH_SIZE:-8}"
export SHARED_TEACHER_MAX_QUEUE_SIZE="${SHARED_TEACHER_MAX_QUEUE_SIZE:-1024}"
export SHARED_TEACHER_BATCH_WAIT_MS="${SHARED_TEACHER_BATCH_WAIT_MS:-5}"

LAUNCH_METADATA_JSON=$(python3 - <<'PY'
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

e = os.environ

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def positive(name):
    require(e[name].isdigit() and int(e[name]) > 0, name + " must be a positive integer")
    return int(e[name])

require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", e["EXP_NAME"]), "EXP_NAME must be a stable directory name")
for key in ("NUM_ROLLOUT", "ROLLOUT_BATCH_SIZE", "N_SAMPLES_PER_PROMPT", "GLOBAL_BATCH_SIZE", "MAX_RESPONSE_LEN", "MAX_CONTEXT_LEN", "SAVE_INTERVAL", "MAX_ACTOR_CKPT_TO_KEEP", "SHARED_TEACHER_MAX_BATCH_SIZE", "SHARED_TEACHER_MAX_QUEUE_SIZE", "SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"):
    positive(key)
require(positive("ROLLOUT_BATCH_SIZE") % 5 == 0, "ROLLOUT_BATCH_SIZE must be divisible by five")
require(positive("GLOBAL_BATCH_SIZE") == positive("ROLLOUT_BATCH_SIZE") * positive("N_SAMPLES_PER_PROMPT"), "GLOBAL_BATCH_SIZE must equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT")
require(positive("GLOBAL_BATCH_SIZE") % int(e["ACTOR_DP"]) == 0, "GLOBAL_BATCH_SIZE must be divisible by actor DP" + e["ACTOR_DP"])
require(positive("N_SAMPLES_PER_PROMPT") >= 2, "GRPO requires at least two samples per prompt")
require(positive("MAX_RESPONSE_LEN") < positive("MAX_CONTEXT_LEN"), "MAX_RESPONSE_LEN must be smaller than MAX_CONTEXT_LEN")
require(e["SHARED_TEACHER_BATCH_WAIT_MS"].isdigit(), "SHARED_TEACHER_BATCH_WAIT_MS must be a nonnegative integer")
require(e["USE_CLEARML"] in {"0", "1"} and e["USE_METRICS_SERVICE"] in {"0", "1"}, "Monitoring overrides must be 0 or 1")
source = Path(e["SOURCE_TEACHER_MANIFEST"]).resolve()
manifest = json.loads(source.read_text())
require(isinstance(manifest, dict), "Manifest must be a JSON object")
require(manifest.get("version") == 1, "Manifest requires version=1")
require(manifest.get("backend") == "sglang-" + e["WEIGHT_SOURCE"] + "-v1", "Manifest backend does not match WEIGHT_SOURCE")
require(type(manifest.get("tp_size")) is int and manifest["tp_size"] == 1, "Shared baseline requires tp_size=1")
teachers = manifest.get("teachers")
require(isinstance(teachers, dict) and set(teachers) == {"T1", "T2", "T3", "T4", "T5"}, "Manifest must contain exactly Teachers T1..T5")
require(manifest.get("initial_teacher") in teachers, "Invalid initial_teacher")
resolved = {}
for teacher, path in teachers.items():
    require(isinstance(path, str) and path and "\n" not in path and "\r" not in path, "Teacher paths must be local path strings")
    resolved[teacher] = str((source.parent / path).resolve())
require(len(set(resolved.values())) == 5, "Five Teachers must use distinct checkpoint directories")
manifest["teachers"] = resolved
if e["WEIGHT_SOURCE"] == "adp":
    package_dir = manifest.get("package_dir")
    require(isinstance(package_dir, str) and bool(package_dir), "ADP manifest requires package_dir")
    manifest["package_dir"] = str((source.parent / package_dir).resolve())
    dependencies = manifest.get("dependency_pythonpath", [])
    require(isinstance(dependencies, list) and all(isinstance(p, str) and p for p in dependencies), "ADP dependency_pythonpath must be local paths")
    manifest["dependency_pythonpath"] = [str((source.parent / p).resolve()) for p in dependencies]
    if e["DRY_RUN"] != "1":
        require((Path(manifest["package_dir"]) / "manifest.json").is_file(), "Prepared ADP manifest is missing")
manifest["event_log"] = str(Path(e["LAUNCH_DIR"]) / "full_reload_events.jsonl")
student = Path(e["STUDENT_MODEL"]).resolve()
prompt = Path(e["PROMPT_SET"]).resolve()
require(prompt.suffix in {".jsonl", ".parquet"}, "PROMPT_SET must be JSONL or Parquet")
if e["DRY_RUN"] != "1":
    require(prompt.is_file(), "PROMPT_SET does not exist")
    from examples.on_policy_distillation.mopd.prepare_qwen35_9b_baseline import read_rows, validate_models, validate_rows
    # Also validate hand-written manifests: all six local checkpoints must have
    # matching config/tokenizer/header layouts and data must exercise T1..T5.
    validate_models(student, resolved)
    validate_rows(read_rows(prompt))
    url = urlparse(e["RAY_DASHBOARD_ADDRESS"])
    require(url.scheme in {"http", "https"} and bool(url.netloc) and url.path in {"", "/"} and not url.query and not url.fragment, "RAY_DASHBOARD_ADDRESS must be a Ray Jobs HTTP(S) origin")
custom = {
    "opd_shared_teacher_manifest": e["SHARED_TEACHER_MANIFEST"],
    "opd_require_all_teacher_scores": True,
    "opd_shared_teacher_max_batch_size": positive("SHARED_TEACHER_MAX_BATCH_SIZE"),
    "opd_shared_teacher_max_queue_size": positive("SHARED_TEACHER_MAX_QUEUE_SIZE"),
    "opd_shared_teacher_batch_wait_ms": int(e["SHARED_TEACHER_BATCH_WAIT_MS"]),
}
print(json.dumps({"manifest": manifest, "custom_config": custom, "initial_teacher_path": resolved[manifest["initial_teacher"]], "student_path": str(student), "prompt_path": str(prompt)}))
PY
)
export LAUNCH_METADATA_JSON
INITIAL_TEACHER=$(python3 -c 'import json,os; print(json.loads(os.environ["LAUNCH_METADATA_JSON"])["initial_teacher_path"])')
STUDENT_MODEL=$(python3 -c 'import json,os; print(json.loads(os.environ["LAUNCH_METADATA_JSON"])["student_path"])')
PROMPT_SET=$(python3 -c 'import json,os; print(json.loads(os.environ["LAUNCH_METADATA_JSON"])["prompt_path"])')

# All local manifest/model/data checks precede any Ray environment queries.
# Dry-run must never source any entrypoint: even environment setup queries Ray.
if [[ "${DRY_RUN}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    # The existing entrypoint expects optional variables to be unset-safe.
    set +u
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
    set +x
    set -u
fi

# Merge AFTER entrypoint setup, preserving its hooks, env_vars and packaging.
# Propagate the list itself so nested Ray actors receive these same values.
if [[ -z "${RUNTIME_ENV_JSON:-}" ]]; then RUNTIME_ENV_JSON='{}'; fi
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
value = json.loads(os.environ["RUNTIME_ENV_JSON"])
if not isinstance(value, dict) or not isinstance(value.get("env_vars", {}), dict):
    raise SystemExit("RUNTIME_ENV_JSON must contain an env_vars object")
worker = value.setdefault("env_vars", {})
guard_hook = "examples.on_policy_distillation.mopd.verified_baseline_entrypoint.worker_setup"
previous_hook = value.get("worker_process_setup_hook", "")
if previous_hook and not isinstance(previous_hook, str):
    raise SystemExit("Baseline worker setup hook must be an importable function name")
if previous_hook != guard_hook:
    worker["RELAX_BASELINE_PRIOR_WORKER_SETUP_HOOK"] = previous_hook
value["worker_process_setup_hook"] = guard_hook
names = set(os.environ["RELAX_PROPAGATE_ENV_VARS"].split(","))
names.update({"RELAX_PROPAGATE_ENV_VARS", "RELAX_SKIP_GLOBAL_CLEANUP", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"})
for name in sorted(names):
    if name and name in os.environ:
        worker[name] = os.environ[name]
print(json.dumps(value))
PY
)
export RUNTIME_ENV_JSON

CKPT_ARGS=(
    --hf-checkpoint "${STUDENT_MODEL}" --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}" --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL}" --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP}"
)
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}" --input-key prompt --label-key label --metadata-key extra_info
    --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking": false}'
    --multimodal-keys '{"image":"images"}' --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-8192}"
    --rm-type mopd --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}" --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-max-context-len "${MAX_CONTEXT_LEN}" --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
    # The prepared text dataset is small. Sequential Dataset preserves its
    # balanced T1..T5 blocks; StreamingDataset shuffles indices independently.
    --seed "${SEED:-20261015}"
)
OPD_ARGS=(
    --use-opd --opd-type sglang --opd-only-reward --opd-kl-coef 1.0
    --opd-teacher-key data_source --teacher-hf-checkpoint "${INITIAL_TEACHER}"
    --teacher-num-gpus-per-engine 1 --opd-token-selection student_sampled
    --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-1200}"
    --opd-log-prob-min-clamp -10.0 --opd-teacher-image-key images --use-rollout-logprobs
    --custom-config-path "${CUSTOM_CONFIG_PATH}"
    --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.3}"
    --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-1}"
    --teacher-sglang-max-total-tokens "${TEACHER_MAX_TOTAL_TOKENS:-8192}"
    --teacher-sglang-context-length "${MAX_CONTEXT_LEN}"
    --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
    --teacher-sglang-max-prefill-tokens "${TEACHER_MAX_PREFILL_TOKENS:-4096}"
    --teacher-sglang-disable-cuda-graph
    --teacher-sglang-disable-overlap-schedule
)
GRPO_ARGS=(
    --advantage-estimator grpo --kl-loss-coef 0.0 --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28
)
OPTIMIZER_ARGS=(
    --optimizer adam --lr "${LEARNING_RATE:-1e-6}" --lr-decay-style constant --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.98 --use-distributed-optimizer
    # Resume the saved scheduler together with optimizer state. The constant
    # schedule permits extending NUM_ROLLOUT after a completed acceptance run.
    --use-checkpoint-opt-param-scheduler
)
PERF_ARGS=(
    --tensor-model-parallel-size "${ACTOR_TP}" --sequence-parallel --pipeline-model-parallel-size 1
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-10240}"
    --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-40960}"
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1 --sglang-load-format dummy --sglang-enable-weights-cpu-backup
    --sglang-mem-fraction-static "${ROLLOUT_MEM_FRACTION:-0.3}"
    --sglang-max-total-tokens "${ROLLOUT_MAX_TOTAL_TOKENS:-8192}"
    --sglang-context-length "${MAX_CONTEXT_LEN}"
    --sglang-chunked-prefill-size "${ROLLOUT_CHUNKED_PREFILL_SIZE:-4096}"
    --sglang-max-running-requests "${ROLLOUT_MAX_RUNNING_REQUESTS:-8}" --sglang-disable-cuda-graph
)
MONITOR_ARGS=(--tb-project-name "${PROJECT_NAME}" --tb-experiment-name "${EXP_NAME}-${now}")
if [[ "${USE_CLEARML}" == 1 ]]; then MONITOR_ARGS+=(--use-clearml); fi
if [[ "${USE_METRICS_SERVICE}" == 1 ]]; then MONITOR_ARGS+=(--use-metrics-service); fi
MISC_ARGS=(
    --bf16 --attention-dropout 0.0 --hidden-dropout 0.0 --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32 --attention-backend flash --no-rope-fusion --skip-eval-before-train
)
TRAIN_ARGS=(
    python3 -m relax.entrypoints.train
    --resource "{\"actor\":[1,${NUM_GPUS}],\"rollout\":[1,${ROLLOUT_GPUS}],\"teacher\":[1,1]}" --rollout-num-gpus "${ROLLOUT_GPUS}"
    --max-staleness 0 --num-data-storage-units 1 --colocate --offload --use-health-check
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPD_ARGS[@]}"
    "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}"
    "${MONITOR_ARGS[@]}" "${MISC_ARGS[@]}"
)
RAY_ARGS=(ray job submit)
if [[ -n "${RAY_NO_WAIT}" ]]; then RAY_ARGS+=(--no-wait); fi
RAY_ARGS+=(--address="${RAY_DASHBOARD_ADDRESS:-http://<existing-ray-jobs-host>:8265}")
if [[ -n "${RAY_SUBMISSION_ID:-}" ]]; then RAY_ARGS+=(--submission-id "${RAY_SUBMISSION_ID}"); fi
if [[ -n "${WORKING_DIR:-}" ]]; then RAY_ARGS+=(--working-dir "${WORKING_DIR}"); fi
RAY_ARGS+=(--runtime-env-json="${RUNTIME_ENV_JSON}")
# The absolute-file wrapper enters this checkout and checks source hashes before
# importing Relax. A dashboard's old cwd must not override PYTHONPATH for -m.
ENTRYPOINT_ARGS=(python3 "${SCRIPT_DIR}/../verified_baseline_entrypoint.py" "${TRAIN_ARGS[@]:3}")

if [[ "${DRY_RUN}" == 1 ]]; then
    python3 - "${RAY_ARGS[@]}" -- "${ENTRYPOINT_ARGS[@]}" <<'PY'
import json
import os
import shlex
import sys
argv = sys.argv[1:]
separator = argv.index("--")
entrypoint = argv[separator + 1:]
metadata = json.loads(os.environ["LAUNCH_METADATA_JSON"])
print(json.dumps({"dry_run": True, "weight_source": os.environ["WEIGHT_SOURCE"], "train_argv": ["python3", "-m", "relax.entrypoints.train", *entrypoint[2:]], "entrypoint_argv": entrypoint, "ray_argv": argv, "shell_command": shlex.join(argv), "runtime_env": json.loads(os.environ["RUNTIME_ENV_JSON"]), "runtime_manifest_path": os.environ["SHARED_TEACHER_MANIFEST"], "custom_config_path": os.environ["CUSTOM_CONFIG_PATH"], "runtime_manifest": metadata["manifest"], "custom_config": metadata["custom_config"], "source_manifest_unchanged": os.environ["SOURCE_TEACHER_MANIFEST"]}, indent=2))
PY
    exit 0
fi

mkdir -p "${LAUNCH_DIR}"
python3 - <<'PY'
import json
import os
from pathlib import Path
value = json.loads(os.environ["LAUNCH_METADATA_JSON"])
Path(os.environ["SHARED_TEACHER_MANIFEST"]).write_text(json.dumps(value["manifest"], indent=2) + "\n")
# JSON is valid YAML; do not import optional training dependencies in the launcher.
Path(os.environ["CUSTOM_CONFIG_PATH"]).write_text(json.dumps(value["custom_config"], indent=2) + "\n")
PY
"${RAY_ARGS[@]}" -- "${ENTRYPOINT_ARGS[@]}" 2>&1 | tee "${LAUNCH_DIR}/submission.log"

}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${WEIGHT_SOURCE:-disk}" in
        disk|memory) run_mopd "${WEIGHT_SOURCE:-disk}" "$@" ;;
        *) echo "Baseline WEIGHT_SOURCE must be disk or memory; use the ADP script for delta switching" >&2; exit 2 ;;
    esac
fi
