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


def test_prepare_json_array(tmp_path):
    source = tmp_path / "dataset" / "data.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            [
                {
                    "context": "document",
                    "question": "Which?",
                    "choice_A": "one",
                    "choice_B": "two",
                    "choice_C": "three",
                    "choice_D": "four",
                    "answer": "B",
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared.json"

    prepare(source.parent, output, "longbench_v2", 1)

    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert "Question: Which?" in prepared[0]["conversations"][0]["value"]
    assert prepared[0]["conversations"][1]["value"] == "B"


def test_prepare_zero_count_writes_all_rows(tmp_path):
    source = tmp_path / "data.jsonl"
    rows = [
        {"context": f"document {i}", "input": "summarize", "answers": ["ok"]}
        for i in range(3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "prepared.json"

    prepare(source, output, "longbench", 0)

    assert len(json.loads(output.read_text(encoding="utf-8"))) == 3
