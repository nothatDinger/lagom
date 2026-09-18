import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("verification_similarity", MODULE_PATH)
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


def test_similarity_rows_include_all_and_adjacent_pairs():
    data = {
        "records": [
            {
                "forward_ct": 7,
                "reqs": [
                    {
                        "rid": "request-1",
                        "verify_len": 3,
                        "draft_tokens": [10, 20, 30],
                        "csa_topk_intersections": [
                            [8, 4, 2],
                            [4, 8, 6],
                            [2, 6, 8],
                        ],
                        "csa_topk_unions": [
                            [8, 12, 14],
                            [12, 8, 10],
                            [14, 10, 8],
                        ],
                    }
                ],
            }
        ]
    }

    pairs, adjacent = analyze.similarity_rows(data)

    assert len(pairs) == 3
    assert len(adjacent) == 2
    assert pairs[0]["jaccard"] == 4 / 12
    assert pairs[-1]["distance"] == 1


def test_similarity_rows_ignore_padding_with_zero_union():
    data = {
        "records": [
            {
                "forward_ct": 1,
                "reqs": [
                    {
                        "verify_len": 2,
                        "draft_tokens": [1, 2],
                        "csa_topk_intersections": [[4, 0], [0, 0]],
                        "csa_topk_unions": [[4, 0], [0, 0]],
                    }
                ],
            }
        ]
    }

    assert analyze.similarity_rows(data) == ([], [])
