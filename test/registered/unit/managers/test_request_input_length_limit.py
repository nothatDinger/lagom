import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    REQUEST_INPUT_LENGTH_LIMIT,
    TokenizerManager,
)


class TestRequestInputLengthLimit(unittest.TestCase):
    def _make_manager(self, mode: str) -> TokenizerManager:
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = REQUEST_INPUT_LENGTH_LIMIT * 2
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.request_input_length_limit_mode = mode
        manager.validate_total_tokens = False
        manager.is_generation = True
        manager.server_args = SimpleNamespace(enable_custom_logit_processor=False)
        return manager

    @staticmethod
    def _make_request() -> GenerateReqInput:
        return GenerateReqInput(input_ids=[1], sampling_params={})

    def test_filter_rejects_input_over_128k(self):
        input_ids = [1] * (REQUEST_INPUT_LENGTH_LIMIT + 1)

        with self.assertRaisesRegex(ValueError, "request input limit"):
            self._make_manager("filter")._validate_one_request(
                self._make_request(), input_ids
            )

    def test_truncate_reduces_input_to_128k(self):
        input_ids = list(range(REQUEST_INPUT_LENGTH_LIMIT + 1))

        self._make_manager("truncate")._validate_one_request(
            self._make_request(), input_ids
        )

        self.assertEqual(len(input_ids), REQUEST_INPUT_LENGTH_LIMIT)
        self.assertEqual(input_ids[-1], REQUEST_INPUT_LENGTH_LIMIT - 1)

    def test_exactly_128k_is_accepted(self):
        input_ids = [1] * REQUEST_INPUT_LENGTH_LIMIT

        self._make_manager("filter")._validate_one_request(
            self._make_request(), input_ids
        )

        self.assertEqual(len(input_ids), REQUEST_INPUT_LENGTH_LIMIT)

    def test_limit_is_disabled_by_default(self):
        input_ids = [1] * (REQUEST_INPUT_LENGTH_LIMIT + 1)

        self._make_manager("none")._validate_one_request(
            self._make_request(), input_ids
        )

        self.assertEqual(len(input_ids), REQUEST_INPUT_LENGTH_LIMIT + 1)


if __name__ == "__main__":
    unittest.main()
