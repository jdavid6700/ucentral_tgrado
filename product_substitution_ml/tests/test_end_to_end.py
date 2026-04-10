from src.substitution.evaluation.metrics import evaluate_retrieval


def test_end_to_end_retrieval(fitted_model):
    validation_pairs = [{"source_sku": "SKU_OLD_1", "relevant_skus": ["SKU_NEW_1"]}]
    metrics = evaluate_retrieval(fitted_model, validation_pairs, top_k=2)
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
