from src.substitution.models.vector_ranker import RankedNeighbor


def test_vector_ranker_returns_neighbors(fitted_model):
    recs = fitted_model.recommend_by_sku("SKU_OLD_1", top_k=2)
    assert len(recs) >= 1
    assert all(isinstance(r, RankedNeighbor) for r in recs)
    assert recs[0].sku != "SKU_OLD_1"


def test_vector_ranker_filters_current_season(fitted_model):
    recs = fitted_model.recommend_by_sku("SKU_OLD_1", top_k=2)
    assert all(r.line == "MORRAL" for r in recs)
