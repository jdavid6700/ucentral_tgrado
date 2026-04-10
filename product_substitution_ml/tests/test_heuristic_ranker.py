from src.substitution.models.heuristic_ranker import HeuristicRanker


def test_heuristic_ranker(sample_catalog):
    source = sample_catalog[sample_catalog["is_current_season"] == 0].copy()
    candidates = sample_catalog[sample_catalog["is_current_season"] == 1].copy()
    ranker = HeuristicRanker()
    recs = ranker.recommend("SKU_OLD_1", source, candidates, top_k=2)
    assert len(recs) >= 1
    assert recs[0].sku == "SKU_NEW_1"
