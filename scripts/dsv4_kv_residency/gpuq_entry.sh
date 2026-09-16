#!/usr/bin/env bash
# Environment-only gpuq entry point; gpuq supplies every setting with --env.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/scripts/dsv4_kv_residency/run_experiment.sh"
