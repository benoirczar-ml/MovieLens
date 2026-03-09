from __future__ import annotations

import pandas as pd

from .metrics import hitrate_at_k, map_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def _ground_truth_dict(df: pd.DataFrame) -> dict[int, set[int]]:
    return df.groupby("userId")["movieId"].apply(lambda x: set(x.tolist())).to_dict()


def _prediction_dict(df: pd.DataFrame, score_col: str) -> dict[int, list[int]]:
    ranked = df.sort_values(["userId", score_col], ascending=[True, False], kind="mergesort")
    return ranked.groupby("userId")["movieId"].apply(list).to_dict()


def evaluate_predictions(
    ground_truth_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    score_col: str,
    ks: list[int],
    model_name: str,
) -> pd.DataFrame:
    gt = _ground_truth_dict(ground_truth_df)
    pred = _prediction_dict(prediction_df, score_col=score_col)

    rows = []
    for k in ks:
        rows.append(
            {
                "model": model_name,
                "k": int(k),
                "recall": recall_at_k(gt, pred, k),
                "hitrate": hitrate_at_k(gt, pred, k),
                "ndcg": ndcg_at_k(gt, pred, k),
                "map": map_at_k(gt, pred, k),
                "mrr": mrr_at_k(gt, pred, k),
                "users": len(gt),
            }
        )

    return pd.DataFrame(rows)
