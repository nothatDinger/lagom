"""CPU utilities for the equal-warp block-shape throughput screen."""

import itertools
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256
from sglang.benchmark.hisparse_h2d_iterator_codegen import (
    instruction_order,
    resource_summary,
)

CONFIGS = {"c4_t1024": (4, 1024), "c8_t512": (8, 512), "c8_t1024": (8, 1024)}
ORDERS = tuple(itertools.permutations(CONFIGS))


def schedule(blocks, threads, batch=32, misses=100):
    if (blocks, threads) not in CONFIGS.values():
        raise ValueError("Unexpected configuration")
    warps = blocks * (threads // 32)
    kv = [len(range(w, batch * misses, warps)) for w in range(warps)]
    return dict(
        total_warps=warps,
        total_kv=sum(kv),
        visits=batch * warps,
        kv_per_warp=sorted(set(kv)),
        kv_per_cta=[
            sum(kv[i : i + threads // 32]) for i in range(0, warps, threads // 32)
        ],
    )


def loaded_artifacts():
    paths = {
        f[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(f := line.split(None, 5)) == 6
    }
    result = []
    for threads in (1024, 512):
        name = f"sgl_kernel_jit_sparse_copy_planned_{threads}_True_True_False_0.so"
        found = [p for p in paths if Path(p).name == name]
        if len(found) != 1:
            raise ValueError(f"Expected one {name}: {found}")
        result.append(dict(threads=threads, path=found[0], sha256=sha256(found[0])))
    return result


def extract_codegen(artifacts, output, tool):
    output.mkdir()
    result = []
    for a in artifacts:
        if sha256(a["path"]) != a["sha256"]:
            raise ValueError("Artifact changed")
        texts = {}
        for kind, option in (
            ("sass", "--dump-sass"),
            ("resources", "--dump-resource-usage"),
        ):
            r = subprocess.run(
                [tool, option, a["path"]], capture_output=True, text=True, timeout=60
            )
            (output / f"t{a['threads']}.{kind}.txt").write_text(r.stdout)
            (output / f"t{a['threads']}.{kind}.stderr.txt").write_text(r.stderr)
            if r.returncode:
                raise RuntimeError(r.stderr)
            texts[kind] = r.stdout
        kernels = instruction_order(texts["sass"], "copy_cache_planned_kernel")
        resources = resource_summary(texts["resources"])
        expected = {"LDG.E.CONSTANT": 4, "LDG.E.64.CONSTANT": 3, "STG.E.64": 3}
        shapes = [
            dict(
                Counter(
                    i["opcode"]
                    for i in k["instructions"]
                    if i["opcode"].startswith(("LDG", "STG"))
                )
            )
            for k in kernels
        ]
        valid = bool(kernels) and all(k["controls_complete"] for k in kernels)
        valid &= all(shape == expected for shape in shapes)
        valid &= resources["complete"] and resources["no_local_or_stack"]
        result.append(
            dict(
                a, resources=resources["rows"], memory_opcodes=shapes, valid=bool(valid)
            )
        )
    return dict(
        valid=all(a["valid"] for a in result),
        artifacts=result,
        note="Static shape/no-spill only; identical scheduling and placement not established",
    )


def summarize(waves):
    expected = {(t, k) for t in range(6) for k in CONFIGS}
    keys = [(w["trial"], w["label"]) for w in waves]
    if len(keys) != 18 or set(keys) != expected:
        raise ValueError("Require 18 unique waves")
    if any(
        len(w["copy_us"]) != 64
        or any(not math.isfinite(x) or x <= 0 for x in w["copy_us"])
        for w in waves
    ):
        raise ValueError("Require 64 positive finite events per wave")
    lookup = {k: statistics.fmean(w["copy_us"]) for k, w in zip(keys, waves)}
    groups = {
        k: dict(
            wave_us=[lookup[t, k] for t in range(6)],
            mean_us=statistics.fmean(lookup[t, k] for t in range(6)),
        )
        for k in CONFIGS
    }
    ratios = {}
    for a, b in [
        ("c8_t512", "c4_t1024"),
        ("c8_t1024", "c4_t1024"),
        ("c8_t1024", "c8_t512"),
    ]:
        values = [lookup[t, a] / lookup[t, b] for t in range(6)]
        ratios[f"{a}/{b}"] = dict(
            waves=values,
            median=statistics.median(values),
            faster=sum(x < 1 for x in values),
        )
    return dict(valid=True, groups=groups, ratios=ratios)
