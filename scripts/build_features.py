#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.data.io import load_ratings, temporal_leave_last_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train/val/test splits and basic stats")
    p.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.get("outputs", {}).get("tables_dir", ROOT / "outputs" / "tables"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    ratings = load_ratings(dcfg["dataset_dir"], min_rating=dcfg.get("min_rating", 0.0), max_rows=dcfg.get("max_rows"))
    train_df, val_df, test_df = temporal_leave_last_split(
        ratings,
        val_k=dcfg.get("val_k", 1),
        test_k=dcfg.get("test_k", 1),
        min_user_interactions=dcfg.get("min_user_interactions", 5),
        split_offset=dcfg.get("split_offset", 0),
    )

    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    test_df.to_csv(out_dir / "test_split.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "rows_ratings": len(ratings),
                "rows_train": len(train_df),
                "rows_val": len(val_df),
                "rows_test": len(test_df),
                "users": ratings["userId"].nunique(),
                "items": ratings["movieId"].nunique(),
            }
        ]
    )
    summary.to_csv(out_dir / "feature_build_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
