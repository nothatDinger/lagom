import importlib.util
from pathlib import Path


_PATH = Path(__file__).parents[4] / "scripts" / "analyze_dspark_filter.py"
_SPEC = importlib.util.spec_from_file_location("analyze_dspark_filter", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_summarize_filter_edge_acceptance_and_cache_misses():
    rows = [
        {
            "kept": True,
            "confidence": 0.9,
            "survival": 0.9,
            "threshold": 0.5,
            "accepted_by_target": True,
            "csa_cache_miss_rate": 0.0,
        },
        {
            "kept": False,
            "confidence": 0.55,
            "survival": 0.49,
            "threshold": 0.5,
            "accepted_by_target": True,
            "csa_cache_miss_rate": 1.0,
        },
        {
            "kept": False,
            "confidence": 0.2,
            "survival": 0.1,
            "threshold": 0.5,
            "accepted_by_target": False,
            "csa_cache_miss_rate": None,
        },
    ]
    result = _MODULE.summarize(rows, margin=0.02)
    assert result["filtered_tokens"] == 2
    assert result["edge_tokens"] == 1
    assert result["edge_target_accept_rate"] == 1.0
    assert result["edge_csa_cache_miss_rate"] == 1.0
    assert result["edge_csa_samples"] == 1


def test_summarize_reports_null_rates_without_samples():
    result = _MODULE.summarize([], margin=0.02)
    assert result["filtered_confidence_mean"] is None
    assert result["edge_target_accept_rate"] is None
    assert result["edge_csa_cache_miss_rate"] is None
