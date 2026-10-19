#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Same MOPD recipe as the full-weight baseline, using pinned-CPU ADP packages.
# See ../README.md for input preparation and validation. No downloads.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    sed -n '3,4p' "${BASH_SOURCE[0]}"
    exit 0
fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run-mopd-qwen35-9b-5teacher-baseline.sh"
run_mopd adp "$@"
