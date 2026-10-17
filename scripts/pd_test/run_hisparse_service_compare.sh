#!/usr/bin/env bash
# Foreground launcher. Stop with Ctrl-C; only its own process groups are stopped.
set -euo pipefail
[[ $# == 3 && "$1" =~ ^(split|pr_fused)$ && "$2" =~ ^(off|on)$ && "$3" == /* && ! -e "$3" ]] || {
  echo "Usage: LAGOM_PY=/absolute/python MODEL_PATH=/model bash $0 split|pr_fused off|on /new/absolute/out" >&2; exit 2;
}
: "${LAGOM_PY:?Set existing Python}" "${MODEL_PATH:?Set local DeepSeek-V4-Flash path}"
[[ -x "$LAGOM_PY" && "$LAGOM_PY" == /* && -d "$MODEL_PATH" ]] || exit 2
command -v setsid >/dev/null
command -v numactl >/dev/null
H2D_SVC_IMPL=$1 H2D_SVC_PROBE=$2 H2D_SVC_OUT=$3
H2D_SVC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$H2D_SVC_ROOT"
export PYTHONPATH="$H2D_SVC_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SGLANG_PROFILE_V2=0 SGLANG_DEBUG_HISPARSE_SKIP_IO=0 SGLANG_DISABLE_HISPARSE_PREFETCH=1
export SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY=legacy
unset SGLANG_DEBUG_HISPARSE_COUNT_BYTES SGLANG_DEBUG_HISPARSE_PROBE_VOLUME
"$LAGOM_PY" - <<'PY'
import socket
for port in (30000,30001,30002):
    with socket.socket() as s:
        s.bind(('127.0.0.1',port))
PY
mkdir -p "$H2D_SVC_OUT"
printf 'backend=%s probe=%s model=%s\n' "$H2D_SVC_IMPL" "$H2D_SVC_PROBE" "$MODEL_PATH" >"$H2D_SVC_OUT/config.txt"
git rev-parse HEAD >"$H2D_SVC_OUT/commit.txt" 2>"$H2D_SVC_OUT/git-error.txt" || true
H2D_SVC_PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for H2D_SVC_PID in "${H2D_SVC_PIDS[@]}"; do
    kill -TERM -- "-$H2D_SVC_PID" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT INT TERM
H2D_SVC_COMMON=(--model-path "$MODEL_PATH" --tp 4 --moe-runner-backend flashinfer_mxfp4
 --cuda-graph-max-bs-decode 64 --skip-server-warmup
 --hisparse-config '{"force_h2d_each_step":true,"top_k":512,"device_buffer_size":3072,"host_to_device_ratio":5}'
 --speculative-algorithm DSPARK --speculative-dspark-block-size 5)
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 SGLANG_HISPARSE_VERIFY_IMPL=split SGLANG_HISPARSE_H2D_PROBE_CAPACITY=0 \
 numactl --cpunodebind=0 --membind=0 "$LAGOM_PY" -m sglang.launch_server "${H2D_SVC_COMMON[@]}" \
 --mem-fraction-static 0.80 --radix-eviction-policy lru --port 30001 --disaggregation-mode prefill \
 --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
 --disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_3","2":"mlx5_4","3":"mlx5_5"}' \
 >"$H2D_SVC_OUT/prefill.log" 2>&1 &
H2D_SVC_PIDS+=("$!")
H2D_SVC_CAPACITY=0
[[ "$H2D_SVC_PROBE" == off ]] || H2D_SVC_CAPACITY=16384
setsid env CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_HISPARSE_VERIFY_IMPL="$H2D_SVC_IMPL" \
 SGLANG_HISPARSE_H2D_PROBE_CAPACITY="$H2D_SVC_CAPACITY" SGLANG_HISPARSE_H2D_PROBE_MAX_BATCH=1024 \
 numactl --cpunodebind=1 --membind=1 "$LAGOM_PY" -m sglang.launch_server "${H2D_SVC_COMMON[@]}" \
 --mem-fraction-static 0.85 --enable-hisparse --disable-radix-cache \
 --port 30000 --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
 --disaggregation-bootstrap-port 8998 \
 --disaggregation-ib-device '{"0":"mlx5_6","1":"mlx5_7","2":"mlx5_8","3":"mlx5_9"}' \
 >"$H2D_SVC_OUT/decode.log" 2>&1 &
H2D_SVC_PIDS+=("$!")
setsid "$LAGOM_PY" -m sglang_router.launch_router --host 127.0.0.1 --port 30002 \
 --pd-disaggregation --mini-lb --prefill http://127.0.0.1:30001 --decode http://127.0.0.1:30000 \
 >"$H2D_SVC_OUT/router.log" 2>&1 &
H2D_SVC_PIDS+=("$!")
printf '%s\n' "${H2D_SVC_PIDS[@]}" >"$H2D_SVC_OUT/owned-pids.txt"
echo "Services started. Wait for readiness in logs; use bench_hisparse_service_compare.sh from a second terminal/job session."
echo "Launcher owns only these process groups: ${H2D_SVC_PIDS[*]}. Ctrl-C stops them."
wait -n "${H2D_SVC_PIDS[@]}"
