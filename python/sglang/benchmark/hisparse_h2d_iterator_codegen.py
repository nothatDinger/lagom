"""Read-only artifact identity and instruction-order extraction for D1/D2."""

import re
from collections import Counter
from pathlib import Path

from sglang.benchmark.hisparse_h2d_codegen import sha256

LIBRARIES = {
    "legacy": "sgl_kernel_jit_sparse_copy_planned_1024_True_True_False_0.so",
    "iterator_d1": "sgl_kernel_jit_sparse_copy_iterator_1024_1.so",
    "iterator_d2": "sgl_kernel_jit_sparse_copy_iterator_1024_2.so",
}


def decode_sm90_control(raw, arch):
    """Decode the upper 64-bit word, not the LDG opcode suffix.

    SM90 uses upper-word bits 46:48 / 49:51 / 52:57 for write dependency,
    read dependency and wait mask. Barrier IDs are 0..5; 7 means unassigned.
    This labels static scheduling fields, not elapsed waits or live requests.
    Keep unsupported architectures or missing/truncated control words unknown.
    """
    if arch not in ("sm_90", "sm_90a"):
        return None
    words = [
        match[1]
        for line in raw[1:]
        if (match := re.fullmatch(r"\s*/\*\s*0x([0-9a-fA-F]{16})\s*\*/\s*", line))
    ]
    if len(words) != 1:
        return None
    value = int(words[0], 16)
    mask = (value >> 52) & 0x3F
    return dict(
        upper_word=f"0x{value:016x}",
        write_barrier=(value >> 46) & 7,
        read_barrier=(value >> 49) & 7,
        wait_mask=mask,
        wait_barriers=[i for i in range(6) if mask & (1 << i)],
    )


def loaded_iterator_artifacts(maps_text=None):
    maps_text = Path("/proc/self/maps").read_text() if maps_text is None else maps_text
    paths = {
        fields[5]
        for line in maps_text.splitlines()
        if len(fields := line.split(None, 5)) == 6
    }
    artifacts = []
    for variant, name in LIBRARIES.items():
        found = sorted(
            p for p in paths if Path(p.removesuffix(" (deleted)")).name == name
        )
        if len(found) != 1:
            raise ValueError(f"Expected one loaded {variant} artifact, found {found}")
        path = Path(found[0])
        artifacts.append(dict(variant=variant, path=str(path), sha256=sha256(path)))
    return artifacts


def instruction_order(sass, kernel_name):
    """Preserve operands and control words; static order is not a dynamic trace."""
    kernels, current, arch = [], None, "unknown"
    for line in sass.splitlines():
        if match := re.search(r"arch\s*=\s*(\S+)", line):
            arch = match[1]
        if "Function :" in line:
            name = line.split("Function :", 1)[1].strip()
            current = None
            if kernel_name in name:
                current = dict(arch=arch, function=name, instructions=[])
                kernels.append(current)
            continue
        if current is None:
            continue
        match = re.match(r"\s*/\*([0-9a-fA-F]+)\*/\s*(.+?;)", line)
        if match:
            instruction = match[2].strip()
            opcode = re.match(r"(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)\b", instruction)
            if opcode:
                current["instructions"].append(
                    dict(
                        offset=match[1],
                        opcode=opcode[1],
                        text=instruction,
                        raw=[line.rstrip()],
                    )
                )
        elif current["instructions"] and re.match(
            r"\s*/\*\s*0x[0-9a-fA-F]+\s*\*/", line
        ):
            current["instructions"][-1]["raw"].append(line.rstrip())
    if not kernels or any(not row["instructions"] for row in kernels):
        raise ValueError(f"Missing populated SASS for {kernel_name}")
    for kernel in kernels:
        instructions = kernel["instructions"]
        for row in instructions:
            row["control"] = decode_sm90_control(row["raw"], kernel["arch"])
        kernel["controls_decoded"] = sum(
            row["control"] is not None for row in instructions
        )
        kernel["controls_complete"] = kernel["controls_decoded"] == len(instructions)
        first_store = next(
            (
                i
                for i, row in enumerate(instructions)
                if row["opcode"].startswith("STG")
            ),
            None,
        )
        kernel["first_store_offset"] = (
            instructions[first_store]["offset"] if first_store is not None else None
        )
        kernel["read64_before_first_store_static_hint"] = (
            sum(
                row["opcode"].startswith("LDG.") and ".64." in row["opcode"]
                for row in instructions[:first_store]
            )
            if first_store is not None
            else None
        )
        kernel["memory_opcodes"] = dict(
            Counter(
                row["opcode"]
                for row in instructions
                if row["opcode"].startswith(("LDG", "STG", "LDL", "STL"))
            )
        )
        kernel["order"] = [
            row
            for row in instructions
            if row["opcode"].startswith(
                ("LDG", "STG", "LDL", "STL", "BRA", "BSYNC", "DEPBAR")
            )
        ]
        # A MOV or arithmetic consumer can wait on A before B is issued even
        # when all six LDGs precede the first STG. Keep a continuous window,
        # including control words, for manual def-use/control-flow review.
        first_read = next(
            (
                i
                for i, row in enumerate(instructions)
                if row["opcode"].startswith("LDG.") and ".64." in row["opcode"]
            ),
            None,
        )
        stores = [
            i for i, row in enumerate(instructions) if row["opcode"].startswith("STG")
        ]
        kernel["read_window"] = (
            instructions[max(0, first_read - 8) : stores[-1] + 9]
            if first_read is not None and stores
            else []
        )
    return kernels


def resource_summary(text):
    rows = []
    for line in text.splitlines():
        if "REG:" in line:
            fields = {
                key: int(value)
                for key, value in re.findall(
                    r"\b(REG|STACK|SHARED|LOCAL):\s*(\d+)", line
                )
            }
            rows.append(dict(text=line.strip(), fields=fields))
    complete = bool(rows) and all(
        {"REG", "STACK", "SHARED", "LOCAL"} <= row["fields"].keys() for row in rows
    )
    return dict(
        rows=rows,
        complete=complete,
        no_local_or_stack=(
            all(row["fields"]["LOCAL"] == row["fields"]["STACK"] == 0 for row in rows)
            if complete
            else None
        ),
    )
