#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.data.io import load_ratings, temporal_leave_last_split
from recsys_ml25m.ranking.features import build_candidate_features, prepare_feature_context
from recsys_ml25m.ranking.ranker import score_ranker, train_ranker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ranker on exported baseline candidates")
    p.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.get("outputs", {}).get("tables_dir", ROOT / "outputs" / "tables"))
    ranker_cfg = cfg.get("ranker", {})
    features_use_gpu = bool(ranker_cfg.get("features_use_gpu", True))

    cval = pd.read_csv(out_dir / "candidates_val_baseline.csv")
    ctest = pd.read_csv(out_dir / "candidates_test_baseline.csv")

    dcfg = cfg["data"]
    ratings = load_ratings(dcfg["dataset_dir"], min_rating=dcfg.get("min_rating", 0.0), max_rows=dcfg.get("max_rows"))
    train_df, val_df, test_df = temporal_leave_last_split(
        ratings,
        val_k=dcfg.get("val_k", 1),
        test_k=dcfg.get("test_k", 1),
        min_user_interactions=dcfg.get("min_user_interactions", 5),
        split_offset=dcfg.get("split_offset", 0),
    )
    recent_genres_n = int(ranker_cfg.get("recent_genres_n", 20))
    feature_ctx = prepare_feature_context(
        train_df=train_df,
        data_dir=dcfg.get("dataset_dir"),
        recent_genres_n=recent_genres_n,
        use_gpu=features_use_gpu,
    )

    tr_feats, cols = build_candidate_features(
        cval,
        train_df,
        val_df,
        data_dir=dcfg.get("dataset_dir"),
        recent_genres_n=recent_genres_n,
        use_gpu=features_use_gpu,
        feature_context=feature_ctx,
    )
    te_feats, _ = build_candidate_features(
        ctest,
        train_df,
        test_df,
        data_dir=dcfg.get("dataset_dir"),
        recent_genres_n=recent_genres_n,
        use_gpu=features_use_gpu,
        feature_context=feature_ctx,
    )
    model = train_ranker(tr_feats, cols, ranker_cfg)
    scored = score_ranker(model, te_feats, cols)
    scored.to_csv(out_dir / "predictions_ranker_baseline.csv", index=False)
    print(f"rank_rows={len(scored)}")


if __name__ == "__main__":
    main()
