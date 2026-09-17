import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("collect.py")
SPEC = importlib.util.spec_from_file_location("dsv4_kv_residency_collect", MODULE_PATH)
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)


def test_sharegpt_json(tmp_path):
    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps([{"conversations": [{"from": "human", "value": "Hello"}]}])
    )

    assert list(collect.prompts(path, "sharegpt")) == ["Hello"]


def test_longbench_directory_of_jsonl(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "narrativeqa.jsonl").write_text(
        json.dumps({"context": "A long document", "input": "What happened?"}) + "\n"
    )

    assert list(collect.prompts(tmp_path, "longbench")) == [
        "A long document\n\nQuestion: What happened?\nAnswer:"
    ]


def test_longbench_v2_choices(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {
                    "context": "A long document",
                    "question": "Which option?",
                    "choice_A": "One",
                    "choice_B": "Two",
                    "choice_C": "Three",
                    "choice_D": "Four",
                }
            ]
        )
    )

    [prompt] = collect.prompts(path, "longbench-v2")
    assert "<text>\nA long document\n</text>" in prompt
    assert "Question: Which option?" in prompt
    assert "(A) One" in prompt
    assert "(D) Four" in prompt
