#!/usr/bin/env bash
set -Eeuo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$DIR/config.env}"
[[ ! -f "$CONFIG" ]] || source "$CONFIG"
exec "$DIR/run_experiment.sh"
