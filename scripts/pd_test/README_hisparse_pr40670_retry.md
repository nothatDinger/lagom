# PR #40670: original vs retry-accounting candidate

Correctness only. Keep the pinned reference unmodified; do not change service
kernels, run bandwidth tests, or infer TPOT improvements from this check.

## Question

The original W6/K16/H128/scratch13 cold-96 case left token 196 absent from the
final union table (95 entries), with three unresolved valid occurrences. Cold-8
passed. Post-commit observation alone does not prove which instruction lost it.

There is a concrete candidate in gather insertion: a failed claiming CAS retries
at the same hash slot, but the original `for` loop still consumes one of its N
probe attempts. Under contention, an insertion can exhaust that budget before
visiting all N slots. A CPU adversarial model demonstrates this failure, not its
occurrence on the GPU.

The candidate adds only `--attempt` before that retry's `continue`. The loop
increment cancels it, so the failed CAS does not consume a slot-probe allowance.
Each slot can be claimed only once in a gather epoch; this does not introduce an
arbitrarily unbounded retry in that monotonic insertion model. This is not a
proof of all PR concurrency behavior.

`validate_candidate()` checks that the candidate header equals the pinned header
plus exactly this statement/comment and an include-path redirect. Transfer code,
layouts, scratch capacity, launch configuration, and output oracle are unchanged.
The original manifest remains enforced. Candidate and original use distinct JIT
module names. No production source is patched.

## Run

Keep the three previously modified reference harness files; do not reset them.
Their SHA256s are recorded. Fast-forward the branch after ensuring no conflicting
local edits. In one gpuq allocation for physical GPU4, from the repository root:

```bash
LAGOM_PY=/absolute/path/to/existing/venv/bin/python \
  bash scripts/pd_test/run_hisparse_pr40670_retry.sh \
  "$PWD/work/hisparse-pr40670/retry-v1"
```

Use a new output directory. The wrapper pins CUDA_VISIBLE_DEVICES=4 and CPU/memory
NUMA1. No cuobjdump, NCU, root access, or new software is needed.

The job checks original cold-8, original cold-96, candidate cold-8, candidate
cold-96 with the same geometry, allocations and input values. Each passing eager
case then receives six cold-reset graph replays and a profiled replay. Exact
gather/commit grid and block, all KV bytes, guards, active-invalid/padded sentinel
rules and final union state remain checked. An eager failure is recorded without
claiming its graph/profile passed. There is no independent copy kernel.

## Read the result

Send `result/retry-compact.txt`. On failure also send the last 30 lines of
`run.log`; retain `result/retry.json` and traces locally.

`candidate_validated=True` requires original cold-8 passing, original cold-96
reproducing unresolved valid KV without other errors, and both candidate cases
passing eager, all six graph replays, profile and final-state checks (three
requests, 96 occurrences/request; 8 or 96 unique entries/request).

Exit zero means that comparison passed, NOT that the original reference passed.
The output separately retains `original_reference_all_pass=False`. If original
cold-96 happens to pass, the comparison is inconclusive and exits nonzero; don't
silently count it as proof of the fix. If the candidate still fails, retain its
missing-token/final-state report and stop. No automatic performance run follows.
A positive result supports this software mechanism on this reproducer; it does
not establish consumer safety for all cases or authorize service integration.
