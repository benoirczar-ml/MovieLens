from __future__ import annotations

import pandas as pd


def topn_for_user(pred_df: pd.DataFrame, user_id: int, score_col: str, n: int = 10) -> pd.DataFrame:
    out = pred_df[pred_df["userId"] == int(user_id)].sort_values(score_col, ascending=False)
    return out.head(n).reset_index(drop=True)
