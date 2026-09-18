from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.test.simple_eval_longbench_v2 import LongBenchV2Eval


class _LengthTokenizer:
    def encode(self, text):
        return list(range(200 if "oversized" in text else 10))


def _example(example_id, context):
    return {
        "_id": example_id,
        "context": context,
        "question": "question",
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
        "answer": "A",
    }


def test_filter_mode_discards_oversized_examples_before_sampling():
    examples = [
        _example("long", "oversized"),
        _example("short-1", "short"),
        _example("short-2", "short"),
    ]

    with (
        envs.SGLANG_REQUEST_INPUT_LENGTH_LIMIT.override(128),
        envs.SGLANG_REQUEST_INPUT_LENGTH_LIMIT_MODE.override("filter"),
        patch.object(LongBenchV2Eval, "_load_dataset", return_value=examples),
        patch(
            "sglang.test.simple_eval_longbench_v2.AutoTokenizer.from_pretrained",
            return_value=_LengthTokenizer(),
        ),
    ):
        evaluation = LongBenchV2Eval(model="model", num_examples=2)

    assert [example["_id"] for example in evaluation.examples] == [
        "short-1",
        "short-2",
    ]


def test_truncate_mode_leaves_client_side_sampling_unchanged():
    examples = [
        _example("long", "oversized"),
        _example("short", "short"),
    ]

    with (
        envs.SGLANG_REQUEST_INPUT_LENGTH_LIMIT.override(128),
        envs.SGLANG_REQUEST_INPUT_LENGTH_LIMIT_MODE.override("truncate"),
        patch.object(LongBenchV2Eval, "_load_dataset", return_value=examples),
        patch(
            "sglang.test.simple_eval_longbench_v2.AutoTokenizer.from_pretrained",
            return_value=_LengthTokenizer(),
        ),
    ):
        evaluation = LongBenchV2Eval(model="model", num_examples=1)

    assert evaluation.examples[0]["_id"] == "long"
