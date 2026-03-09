from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys_ml25m.pipeline import run_pipeline


def test_smoke_pipeline(tmp_path: Path) -> None:
    data_dir = tmp_path / "ml-25m"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ts = 1_700_000_000
    for u in range(1, 9):
        for j in range(1, 9):
            rows.append({"userId": u, "movieId": j + u, "rating": 4.0, "timestamp": ts + j})

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "ratings.csv", index=False)

    cfg = {
        "data": {
            "dataset_dir": str(data_dir),
            "min_rating": 3.5,
            "max_rows": None,
            "val_k": 1,
            "test_k": 1,
            "min_user_interactions": 5,
        },
        "retrieval": {"algorithm": "popular"},
        "ranker": {"framework": "xgboost", "n_estimators": 20, "n_jobs": 1},
        "final_model": {"enabled": False},
        "evaluation": {"k_candidates": 20, "ks": [5, 10]},
        "outputs": {
            "tables_dir": str(tmp_path / "tables"),
            "figures_dir": str(tmp_path / "figures"),
        },
    }

    out = run_pipeline(cfg)
    assert not out["comparison"].empty
    assert (tmp_path / "tables" / "metrics_comparison.csv").exists()
