#!/usr/bin/env bash
# gpuq entry point. Example: gpuq submit -- ./scripts/dsv4_kv_residency/entrypoint.sh config.env
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-$ROOT/scripts/dsv4_kv_residency/config.env}"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG (copy config.env.example)" >&2; exit 2; }
set -a; source "$CONFIG"; set +a
exec "$ROOT/scripts/dsv4_kv_residency/run_experiment.sh"
