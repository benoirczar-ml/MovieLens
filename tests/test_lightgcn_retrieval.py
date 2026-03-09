from __future__ import annotations

import pandas as pd

from recsys_ml25m.retrieval import build_retrieval_model, generate_candidates


def test_lightgcn_retrieval_candidates_smoke() -> None:
    train_df = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 3],
            "movieId": [10, 11, 10, 12, 11, 12, 13],
            "rating": [4.0, 4.5, 5.0, 4.0, 4.0, 4.5, 5.0],
            "timestamp": [1, 2, 1, 3, 1, 2, 3],
        }
    )
    cfg = {
        "algorithm": "lightgcn",
        "use_gpu": False,
        "require_gpu": False,
        "embedding_dim": 8,
        "num_layers": 1,
        "epochs": 1,
        "batch_size": 8,
        "num_negatives": 2,
        "query_batch_size": 8,
        "candidate_multiplier": 2,
        "seed": 42,
    }

    art = build_retrieval_model(train_df, cfg)
    assert art.algorithm == "lightgcn"

    cand = generate_candidates(art, user_ids=[1, 2, 3], k=2, filter_seen=True)
    assert set(["userId", "movieId", "retrieval_score"]).issubset(cand.columns)
    assert cand["userId"].nunique() == 3
    assert (cand.groupby("userId").size() <= 2).all()

    seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    for uid, rows in cand.groupby("userId"):
        assert all(int(mid) not in seen[int(uid)] for mid in rows["movieId"].tolist())
