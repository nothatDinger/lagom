#!/usr/bin/env bash
# gpuq job entry point. Export variables from env.example before submitting it.
set -Eeuo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/run.sh" "$@"
