#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.data.io import load_ratings, temporal_leave_last_split
from recsys_ml25m.eval.offline import evaluate_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ranker predictions")
    p.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    p.add_argument("--predictions", default=None, help="Path to predictions CSV with rank_score")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.get("outputs", {}).get("tables_dir", ROOT / "outputs" / "tables"))

    pred_path = Path(args.predictions) if args.predictions else out_dir / "predictions_ranker_baseline.csv"
    pred = pd.read_csv(pred_path)

    dcfg = cfg["data"]
    ecfg = cfg.get("evaluation", {})
    ratings = load_ratings(dcfg["dataset_dir"], min_rating=dcfg.get("min_rating", 0.0), max_rows=dcfg.get("max_rows"))
    _, _, test_df = temporal_leave_last_split(
        ratings,
        val_k=dcfg.get("val_k", 1),
        test_k=dcfg.get("test_k", 1),
        min_user_interactions=dcfg.get("min_user_interactions", 5),
        split_offset=dcfg.get("split_offset", 0),
    )

    metrics = evaluate_predictions(
        ground_truth_df=test_df,
        prediction_df=pred,
        score_col="rank_score",
        ks=[int(k) for k in ecfg.get("ks", [10, 20, 50])],
        model_name="ranker_eval",
    )
    metrics.to_csv(out_dir / "metrics_eval.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
