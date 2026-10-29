"""The HiSparse startup gate must not undo Lagom's graph configuration."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.speculative.ragged_verify import RaggedVerifyMode


def graph_disable_reason(*, lagom, prefetcher, mode, monkeypatch):
    # Load the production decision block; constructing a ModelRunner would
    # require CUDA and a checkpoint even though this gate is pure host logic.
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "maybe_init_hisparse_coordinator"
    )
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "disable_decode_graph_reason"
            for t in n.targets
        )
    )
    end = next(
        i
        for i, n in enumerate(method.body[start + 1 :], start + 1)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "disable_decode_graph_reason"
    )
    monkeypatch.setattr(
        "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode", lambda: mode
    )
    namespace = dict(
        logger=Mock(),
        self=SimpleNamespace(
            server_args=SimpleNamespace(is_lagom=lagom),
            spec_algorithm=SimpleNamespace(
                is_eagle=lambda: False, is_dflash_family=lambda: True
            ),
            hisparse_coordinator=SimpleNamespace(
                prefetcher=object() if prefetcher else None,
                prefetcher_name=prefetcher,
                is_dsv4_hisparse=True,
            ),
        ),
    )
    module = ast.Module(body=method.body[start:end], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["disable_decode_graph_reason"]


@pytest.mark.parametrize("mode", [RaggedVerifyMode.STATIC, RaggedVerifyMode.COMPACT])
def test_lagom_retains_graphs_regardless_of_ordinary_dspark_layout(mode, monkeypatch):
    assert (
        graph_disable_reason(
            lagom=True, prefetcher="lagom", mode=mode, monkeypatch=monkeypatch
        )
        is None
    )


def test_ordinary_compact_hisparse_verify_retains_existing_eager_guard(monkeypatch):
    reason = graph_disable_reason(
        lagom=False,
        prefetcher=None,
        mode=RaggedVerifyMode.COMPACT,
        monkeypatch=monkeypatch,
    )
    assert "eager decode" in reason
