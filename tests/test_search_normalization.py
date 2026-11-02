import numpy as np

from core.search_engine.hybrid_search import HybridSearch


def test_rare_term_bm25_scores_are_not_zeroed() -> None:
    # 3 of 200 chunks match: p95 is 0, the matches must still rank.
    scores = np.zeros(200, dtype=np.float32)
    scores[[5, 17, 120]] = [4.0, 2.0, 1.0]
    normalized = HybridSearch._normalize_scores_robust_p95(None, scores)
    assert normalized[5] == 1.0
    assert 0 < normalized[120] < normalized[17] < 1.0
    assert np.count_nonzero(normalized) == 3


def test_common_term_scores_keep_p95_scaling() -> None:
    scores = np.linspace(0.0, 10.0, 200, dtype=np.float32)
    normalized = HybridSearch._normalize_scores_robust_p95(None, scores)
    assert normalized.max() == 1.0
    assert np.count_nonzero(normalized == 1.0) > 1  # top 5% clipped to 1


def test_no_match_gives_zeros() -> None:
    normalized = HybridSearch._normalize_scores_robust_p95(None, np.zeros(10, dtype=np.float32))
    assert not np.any(normalized)
