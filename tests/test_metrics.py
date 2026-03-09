from recsys_ml25m.eval.metrics import hitrate_at_k, map_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def test_metrics_simple_case() -> None:
    gt = {1: {10, 20}, 2: {30}}
    pred = {1: [10, 99, 20], 2: [40, 30, 50]}

    r = recall_at_k(gt, pred, 2)
    h = hitrate_at_k(gt, pred, 2)
    n = ndcg_at_k(gt, pred, 3)
    m = map_at_k(gt, pred, 3)
    rr = mrr_at_k(gt, pred, 3)

    assert abs(r - 0.75) < 1e-9
    assert abs(h - 1.0) < 1e-9
    assert 0.0 < n <= 1.0
    assert 0.0 < m <= 1.0
    assert abs(rr - 0.75) < 1e-9
