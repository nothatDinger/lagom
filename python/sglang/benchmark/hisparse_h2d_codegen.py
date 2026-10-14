"""Read-only provenance and SASS inspection for the vector-copy experiment."""

import hashlib
import re
from collections import Counter
from pathlib import Path

POLICIES = {0: "legacy", 1: "ca", 3: "ca_pair", 4: "ca_v2"}
_LIBRARY = re.compile(
    r"sgl_kernel_jit_sparse_copy_planned_1024_True_True_False_([0134])\.so$"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loaded_copy_artifacts(maps_text=None):
    """Identify the actual loaded libraries, never guess the newest cache file."""
    if maps_text is None:
        maps_text = Path("/proc/self/maps").read_text()
    paths = set()
    for line in maps_text.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and _LIBRARY.search(fields[5].removesuffix(" (deleted)")):
            paths.add(fields[5])
    result = []
    for code, policy in POLICIES.items():
        found = sorted(
            p for p in paths if p.removesuffix(" (deleted)").endswith(f"_{code}.so")
        )
        if len(found) != 1:
            raise ValueError(f"Expected one loaded {policy} library, found {found}")
        path = Path(found[0])
        result.append(
            dict(policy=policy, code=code, path=str(path), sha256=sha256(path))
        )
    return result


def kernel_memory_ops(sass):
    """Static opcode counts, not source attribution or dynamic transactions."""
    kernels = []
    arch, current = "unknown", None
    for line in sass.splitlines():
        match = re.search(r"arch\s*=\s*(\S+)", line)
        if match:
            arch = match[1]
        if "Function :" in line:
            name = line.split("Function :", 1)[1].strip()
            current = None
            if "copy_cache_planned_kernel" in name:
                current = dict(
                    arch=arch, function=name, opcodes=Counter(), memory_ops=[]
                )
                kernels.append(current)
        elif current is not None:
            instruction = re.sub(r"/\*.*?\*/", "", line).strip()
            match = re.match(r"(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)\s", instruction)
            if match and match[1].startswith(("LDG", "STG", "LDL", "STL")):
                current["opcodes"][match[1]] += 1
                current["memory_ops"].append(line.strip())
    if not kernels or any(not kernel["memory_ops"] for kernel in kernels):
        raise ValueError("No populated copy_cache_planned_kernel SASS section")
    return kernels
