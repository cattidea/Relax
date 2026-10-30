#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Compare repository sources on one immutable image; this does not restore the old image.
set -euo pipefail

fail() {
    echo "$*" >&2
    exit 1
}

for variable in CI_SOURCE_OLD CI_SOURCE_NEW CI_SOURCE_OLD_DIR CI_SOURCE_NEW_DIR \
    CI_HARNESS_PATH CI_CONTAINER_PREFIX CI_ARTIFACT_ROOT HOST_UV_CACHE_DIR CI_IMAGE \
    CI_HARNESS_REF CI_RUN_ID CI_RUN_ATTEMPT GITHUB_WORKSPACE; do
    : "${!variable:?Missing required environment variable: $variable}"
done
[[ "$CI_SOURCE_OLD" == 13816f8692f087104b7222087af565b6dee5fbab ]] || fail "Unexpected old source"
[[ "$CI_SOURCE_NEW" == ea3f0cee94b9400cfe1591beb4f2dbfe712d362c ]] || fail "Unexpected new source"
[[ "$CI_IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "Invalid immutable image ID"
[[ "$CI_HARNESS_REF" =~ ^[0-9a-f]{40}$ ]] || fail "Invalid harness SHA"
[[ "$CI_RUN_ID" =~ ^[1-9][0-9]*$ && "$CI_RUN_ATTEMPT" =~ ^[1-9][0-9]*$ ]] || fail "Invalid run identity"
[[ "$CI_CONTAINER_PREFIX" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail "Invalid container prefix"
for directory in "$CI_SOURCE_OLD_DIR" "$CI_SOURCE_NEW_DIR" "$CI_HARNESS_PATH" \
    "$CI_ARTIFACT_ROOT" "$HOST_UV_CACHE_DIR"; do
    [[ "$directory" == /* && "$directory" != / ]] || fail "Expected an absolute non-root directory"
done
workspace=$(cd "$GITHUB_WORKSPACE" && pwd -P)
source_old_dir=$(cd "$CI_SOURCE_OLD_DIR" && pwd -P)
source_new_dir=$(cd "$CI_SOURCE_NEW_DIR" && pwd -P)
[[ "$source_old_dir" == "$workspace/"* && "$source_new_dir" == "$workspace/"* ]] || fail "Source outside workspace"
[[ "$source_old_dir" != "$source_new_dir" && -d "$CI_HARNESS_PATH" ]] || fail "Invalid source or harness directory"
actual_image=$(docker image inspect --format '{{.Id}}' "$CI_IMAGE")
[[ "$actual_image" == "$CI_IMAGE" ]] || fail "Immutable image ID mismatch"
mkdir -p "$CI_ARTIFACT_ROOT" "$HOST_UV_CACHE_DIR"
export UV_CACHE_DIR=/uv-cache UV_LINK_MODE=copy

# Only validated SHA/numeric fields enter shell-generated JSON.
printf '{"mode":"same_image_source_ABBA","host":"M01","source_A":"%s","source_B":"%s","image":"%s","harness_ref":"%s","run_id":%s,"run_attempt":%s,"order":["01-A","02-B","03-B","04-A"],"physical_gpus":[4,5,6,7],"pytest_scope":"original_plus_warm","historical_image_restored":false}\n' \
    "$CI_SOURCE_OLD" "$CI_SOURCE_NEW" "$CI_IMAGE" "$CI_HARNESS_REF" \
    "$CI_RUN_ID" "$CI_RUN_ATTEMPT" > "$CI_ARTIFACT_ROOT/comparison-plan.json"
docker image inspect --format '{{json .RootFS.Layers}}' "$CI_IMAGE" \
    > "$CI_ARTIFACT_ROOT/image-rootfs.json"

current_container=
sampler_pid=
cleanup_trial() {
    if [[ -n "$sampler_pid" ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=
    fi
    if [[ -n "$current_container" ]]; then
        if docker rm --force "$current_container" >/dev/null 2>&1; then
            current_container=
        else
            echo "Failed to remove current trial container: $current_container" >&2
            return 1
        fi
    fi
}
finish() {
    local status=$?
    trap - EXIT
    if ! cleanup_trial; then
        if (( status == 0 )); then status=1; fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sample_host() {
    trap - EXIT
    sleep_pid=
    trap 'if [[ -n "$sleep_pid" ]]; then kill "$sleep_pid" 2>/dev/null || true; wait "$sleep_pid" 2>/dev/null || true; fi; exit 0' TERM INT
    while true; do
        date -u +%FT%TZ
        awk '/^cpu |^procs_running/ {print}' /proc/stat
        awk '/^processor|^cpu MHz/ {print}' /proc/cpuinfo
        cat /proc/loadavg /proc/diskstats
        nvidia-smi --query-gpu=index,utilization.gpu,clocks.sm,power.draw --format=csv,noheader,nounits
        ps -eo comm | sort | uniq -c | awk '$2 ~ /^(nvcc|cc1plus|buildkitd)$/ {print}'
        sleep 5 &
        sleep_pid=$!
        wait "$sleep_pid"
        sleep_pid=
    done
}

tree_a=
tree_b=
environment_reference=

run_trial() {
    local trial=$1 arm=$2 source_ref source_dir container artifact_dir tree_sha environment_record environment_sha
    if [[ "$arm" == A ]]; then
        source_ref=$CI_SOURCE_OLD
        source_dir=$source_old_dir
    else
        source_ref=$CI_SOURCE_NEW
        source_dir=$source_new_dir
    fi
    container="$CI_CONTAINER_PREFIX-$trial"
    artifact_dir="$CI_ARTIFACT_ROOT/$trial"
    if docker inspect "$container" >/dev/null 2>&1 || [[ -e "$artifact_dir" ]]; then
        echo "Refusing to overwrite an existing trial: $trial" >&2
        exit 1
    fi
    mkdir -p "$artifact_dir"
    echo "Starting trial $trial: source=$source_ref image=$CI_IMAGE"
    docker create --pull never --rm --init \
        --name "$container" --gpus '"device=4,5,6,7"' --shm-size=32g \
        -v "$source_dir:/source:ro" -v "$artifact_dir:/artifacts" \
        -v "$CI_HARNESS_PATH:/duration-harness:ro" \
        -v "$HOST_UV_CACHE_DIR:$UV_CACHE_DIR" --workdir /workspace/Relax \
        --env UV_CACHE_DIR --env UV_LINK_MODE \
        --env http_proxy --env https_proxy --env no_proxy \
        --entrypoint /bin/bash "$CI_IMAGE" -c 'sleep infinity'
    current_container=$container
    docker inspect --format 'Image={{.Image}} CpuQuota={{.HostConfig.CpuQuota}} CpuPeriod={{.HostConfig.CpuPeriod}} CpusetCpus={{.HostConfig.CpusetCpus}} Memory={{.HostConfig.Memory}}' \
        "$container" | tee "$artifact_dir/container-conditions.txt"
    docker start "$container"
    docker exec "$container" bash -euc '
        cp -a --no-preserve=ownership /source/. /workspace/Relax/
        if [[ -e .git || -L .git ]]; then
            echo "Test copy Git metadata before normalization: present"
        else
            echo "Test copy Git metadata before normalization: absent"
        fi
        rm -rf -- .git
        [[ ! -e .git && ! -L .git ]]
        echo "Test copy Git metadata after normalization: absent"
        nvidia-smi
    '
    tree_sha=$(docker exec --env CI_TRIAL="$trial" --env CI_ARM="$arm" \
        --env CI_SOURCE_REF="$source_ref" --env CI_IMAGE --env CI_HARNESS_REF \
        --env CI_RUN_ID --env CI_RUN_ATTEMPT "$container" python -c '
import hashlib, json, os
from pathlib import Path
def tree_digest(root):
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Expected a regular Relax source directory")
    digest, count = hashlib.sha256(), 0
    for entry in sorted(root.rglob("*")):
        if entry.is_symlink():
            raise RuntimeError("Symlink in measured Relax tree: " + str(entry))
        if entry.is_file():
            relative = entry.relative_to(root).as_posix().encode()
            data = entry.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
            count += 1
        elif not entry.is_dir():
            raise RuntimeError("Special file in measured Relax tree: " + str(entry))
    return digest.hexdigest(), count
source_tree, source_count = tree_digest(Path("/source/relax"))
private_tree, private_count = tree_digest(Path("/workspace/Relax/relax"))
if (source_tree, source_count) != (private_tree, private_count) or source_count == 0:
    raise RuntimeError("Private Relax copy does not match the selected source")
info = {key: os.environ[key] for key in ("CI_TRIAL", "CI_ARM", "CI_SOURCE_REF", "CI_IMAGE", "CI_HARNESS_REF", "CI_RUN_ID", "CI_RUN_ATTEMPT")}
info.update(mode="same_image_source_control", pytest_scope="original_plus_warm", historical_image_restored=False)
info.update(relax_tree_sha256=source_tree, relax_tree_file_count=source_count, private_copy_matches=True, relax_tree_hash_mode="sorted_relative_path_and_contents_with_8byte_length_prefixes")
Path("/artifacts/trial.json").write_text(json.dumps(info, indent=2) + "\n")
print(source_tree)
')
    [[ "$tree_sha" =~ ^[0-9a-f]{64}$ ]] || fail "Invalid Relax tree hash"
    if [[ "$arm" == A ]]; then
        [[ -z "$tree_a" || "$tree_a" == "$tree_sha" ]] || fail "A source changed between trials"
        tree_a=$tree_sha
    else
        [[ -z "$tree_b" || "$tree_b" == "$tree_sha" ]] || fail "B source changed between trials"
        tree_b=$tree_sha
    fi
    [[ -z "$tree_a" || -z "$tree_b" || "$tree_a" != "$tree_b" ]] || fail "A and B Relax source trees are identical"
    docker exec "$container" uv pip install \
        --system --break-system-packages -r requirements.txt \
        --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
        cryptography==50.0.2 oauthlib==4.0.0 pyparsing==3.3.3 \
        2>&1 | tee "$artifact_dir/dependencies.log"
    docker exec --env CI_DURATION_TRACE_DIR=/artifacts "$container" \
        python /duration-harness/instrument_duration.py /workspace/Relax "$source_ref"
    docker exec "$container" python -c '
import importlib.metadata as m, json, os
from pathlib import Path
info = {"packages": sorted((d.metadata["Name"], d.version) for d in m.distributions()), "affinity": sorted(os.sched_getaffinity(0)), "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
Path("/artifacts/environment.json").write_text(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k != "packages"}))
'
    environment_record=$(docker exec "$container" sha256sum /artifacts/environment.json)
    [[ "$environment_record" =~ ^([0-9a-f]{64})[[:space:]]+/artifacts/environment\.json$ ]] || fail "Invalid environment hash"
    environment_sha=${BASH_REMATCH[1]}
    printf '%s\n' "$environment_sha" > "$artifact_dir/environment.sha256"
    [[ -z "$environment_reference" || "$environment_reference" == "$environment_sha" ]] || fail "Environment changed between trials"
    environment_reference=$environment_sha
    sample_host > "$artifact_dir/host-samples.log" 2>&1 &
    sampler_pid=$!
    docker exec "$container" python -I -S \
        /duration-harness/duration_cpu_control.py /artifacts/cpu-control.json
    docker exec --env CI_DURATION_TRACE_DIR=/artifacts "$container" bash -euo pipefail -c '
        export PYTHONPATH="/duration-harness:$PWD${PYTHONPATH:+:$PYTHONPATH}"
        timeout --signal=TERM --kill-after=60s 40m \
        python -m pytest -p duration_trace -v -ra --tb=short --durations=20 --junitxml=/artifacts/junit.xml \
            "tests/backends/fsdp/test_weight_transaction.py::test_non_src_serialization_failure_exits_all_ranks_without_hang" \
            "tests/backends/megatron/test_gdn_cp_gpu.py::test_gdn_cp_layout_and_gradients[chunkwise]" \
            "tests/backends/megatron/test_qsa_mixed.py::test_qsa_mixed_output_and_gradients[lengths1-8]" \
            "tests/backends/megatron/test_qsa_mixed.py::test_qsa_mixed_output_and_gradients[lengths2-8]" \
            2>&1 | tee /artifacts/pytest.log
    '
    cleanup_trial || exit 1
    echo "Completed trial $trial"
}

run_trial 01-A A
run_trial 02-B B
run_trial 03-B B
run_trial 04-A A
