from types import SimpleNamespace

import pytest

from sglang.srt.managers.tokenizer_manager import TokenizerManager

REQUEST_INPUT_LIMIT = 128 * 1024


def make_manager(mode, limit=REQUEST_INPUT_LIMIT):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.context_len = REQUEST_INPUT_LIMIT * 2
    manager.num_reserved_tokens = 0
    manager.allow_auto_truncate = False
    manager.request_input_length_limit = limit
    manager.request_input_length_limit_mode = mode
    manager.validate_total_tokens = False
    manager.is_generation = False
    return manager


@pytest.mark.parametrize("mode", ["filter", "truncate"])
def test_request_at_128k_is_allowed(mode):
    input_ids = [1] * REQUEST_INPUT_LIMIT

    make_manager(mode)._validate_one_request(
        SimpleNamespace(sampling_params={}), input_ids
    )

    assert len(input_ids) == REQUEST_INPUT_LIMIT


def test_oversized_request_is_filtered():
    input_ids = [1] * (REQUEST_INPUT_LIMIT + 1)

    with pytest.raises(ValueError, match="exceeds the configured input limit"):
        make_manager("filter")._validate_one_request(
            SimpleNamespace(sampling_params={}), input_ids
        )


def test_oversized_request_is_truncated():
    input_ids = [1] * (REQUEST_INPUT_LIMIT + 1)

    make_manager("truncate")._validate_one_request(
        SimpleNamespace(sampling_params={}), input_ids
    )

    assert len(input_ids) == REQUEST_INPUT_LIMIT


def test_zero_limit_disables_guard():
    input_ids = [1] * (REQUEST_INPUT_LIMIT + 1)

    make_manager("filter", limit=0)._validate_one_request(
        SimpleNamespace(sampling_params={}), input_ids
    )

    assert len(input_ids) == REQUEST_INPUT_LIMIT + 1


def test_custom_limit_is_applied():
    input_ids = [1] * 17

    make_manager("truncate", limit=16)._validate_one_request(
        SimpleNamespace(sampling_params={}), input_ids
    )

    assert len(input_ids) == 16
