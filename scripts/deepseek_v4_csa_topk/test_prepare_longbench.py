import json

import pytest

from prepare_longbench import prepare


@pytest.mark.parametrize(
    ("variant", "row", "expected"),
    [
        (
            "longbench",
            {"context": "document", "input": "summarize", "answers": ["summary"]},
            "document\n\nsummarize",
        ),
        (
            "longbench_v2",
            {
                "context": "document",
                "question": "Which?",
                "choice_A": "one",
                "choice_B": "two",
                "choice_C": "three",
                "choice_D": "four",
                "answer": "B",
            },
            "Question: Which?",
        ),
    ],
)
def test_prepare_jsonl(tmp_path, variant, row, expected):
    source = tmp_path / "dataset" / "data.jsonl"
    source.parent.mkdir()
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "prepared.json"

    prepare(source.parent, output, variant, 1)

    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert expected in prepared[0]["conversations"][0]["value"]
    assert prepared[0]["conversations"][1]["value"]
