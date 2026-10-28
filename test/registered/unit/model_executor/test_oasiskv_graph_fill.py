"""Cached replay copy plans preserve padding and always read fresh sources."""

from types import SimpleNamespace as NS

import torch
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    GraphSlot,
    PaddingPolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def registry():
    r = CudaGraphBufferRegistry(device=torch.device("cpu"), max_bs=8, max_num_tokens=16)
    for name, dtype, policy, value in (
        ("input_ids", torch.int64, PaddingPolicy.FOREACH_COPY, None),
        ("positions", torch.int64, PaddingPolicy.ZERO, None),
        ("seq_lens", torch.int32, PaddingPolicy.FILL_SENTINEL, 1),
    ):
        r.register_slot(
            GraphSlot(
                name,
                lambda bs, mt: (mt,),
                dtype,
                padding_policy=policy,
                pad_value=value,
            )
        )
    r.register_slot(
        GraphSlot(
            "mrope",
            lambda bs, mt: (3, mt),
            torch.int64,
            padding_policy=PaddingPolicy.ZERO,
            slice_fn=lambda x, n: x[:, :n],
        )
    )
    r.register_slot(
        GraphSlot(
            "nested",
            lambda bs, mt: (mt,),
            torch.int64,
            source_fn=lambda fb, ctx: fb.nested.values,
        )
    )
    r.register_slot(
        GraphSlot(
            "computed",
            lambda bs, mt: (1,),
            torch.int64,
            axis="none",
            copy_from_fb=False,
            post_fill=lambda dst, fb, ctx: dst.copy_(
                r.get_slot("input_ids").buffer[: ctx.raw_num_tokens].sum()
            ),
        )
    )
    return r


def test_cached_fill_matches_general_path_with_rebound_sources_and_padding():
    reference, cached = registry(), registry()
    cached.enable_cached_fill()
    for step, (raw, padded) in enumerate([(3, 4), (3, 4), (1, 2), (4, 4), (3, 4)]):
        count = raw * 2
        fb = NS(
            input_ids=torch.arange(count) + 100 * step,
            positions=torch.arange(count) + step if step != 1 else None,
            seq_lens=torch.arange(count, dtype=torch.int64) + 20,
            mrope=torch.arange(3 * count).reshape(3, count) + step,
            nested=NS(values=torch.arange(step + 1) if step != 2 else None),
        )
        for item in (reference, cached):
            item.fill_from(
                fb,
                raw_bs=raw,
                padded_bs=padded,
                raw_num_tokens=count,
                padded_num_tokens=padded * 2,
            )
        for name in reference.slot_names():
            torch.testing.assert_close(
                cached.get_slot(name).buffer, reference.get_slot(name).buffer
            )
    assert len(cached._fill_plans) == 3


def test_cached_fill_reuses_destination_views_and_invalidates_on_registration(
    monkeypatch,
):
    cached = registry()
    cached.enable_cached_fill()
    fb = NS(input_ids=torch.tensor([1, 2]), nested=NS(values=None))
    kwargs = {"raw_bs": 1, "padded_bs": 2, "raw_num_tokens": 2, "padded_num_tokens": 4}
    cached.fill_from(fb, **kwargs)
    plan = next(iter(cached._fill_plans.values()))
    with monkeypatch.context() as patch:
        for slot in cached._slots.values():
            patch.setattr(
                slot,
                "_raw_n",
                lambda *a: (_ for _ in ()).throw(AssertionError("rebuilt views")),
            )
        fb.input_ids = torch.tensor([7, 8], dtype=torch.int32)
        cached.fill_from(fb, **kwargs)
    assert next(iter(cached._fill_plans.values())) is plan
    assert cached.get_slot("input_ids").buffer[:2].tolist() == [7, 8]
    assert cached.get_slot("computed").buffer.item() == 15
    cached.register_slot(GraphSlot("extra", lambda bs, mt: (mt,), torch.int64))
    assert not cached._fill_plans
    fb.extra = torch.tensor([9, 10])
    cached.fill_from(fb, **kwargs)
    assert cached.get_slot("extra").buffer[:2].tolist() == [9, 10]


def test_cached_plan_count_is_bounded():
    cached = CudaGraphBufferRegistry(
        device=torch.device("cpu"), max_bs=64, max_num_tokens=128
    )
    cached.register_slot(GraphSlot("ids", lambda bs, mt: (mt,), torch.int64))
    cached.enable_cached_fill()
    for size in range(1, 50):
        cached.fill_from(
            NS(ids=torch.ones(size)),
            raw_bs=size,
            padded_bs=size,
            raw_num_tokens=size,
            padded_num_tokens=size,
        )
    assert len(cached._fill_plans) == 32
