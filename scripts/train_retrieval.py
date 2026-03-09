#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.data.io import load_ratings, temporal_leave_last_split
from recsys_ml25m.retrieval.als import build_retrieval_model, generate_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline retrieval and export candidates")
    p.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.get("outputs", {}).get("tables_dir", ROOT / "outputs" / "tables"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    ecfg = cfg.get("evaluation", {})

    ratings = load_ratings(dcfg["dataset_dir"], min_rating=dcfg.get("min_rating", 0.0), max_rows=dcfg.get("max_rows"))
    train_df, val_df, test_df = temporal_leave_last_split(
        ratings,
        val_k=dcfg.get("val_k", 1),
        test_k=dcfg.get("test_k", 1),
        min_user_interactions=dcfg.get("min_user_interactions", 5),
        split_offset=dcfg.get("split_offset", 0),
    )

    model = build_retrieval_model(train_df, cfg.get("retrieval", {}))
    k_candidates = int(ecfg.get("k_candidates", 200))

    users_val = sorted(set(val_df["userId"].tolist()) & set(train_df["userId"].tolist()))
    users_test = sorted(set(test_df["userId"].tolist()) & set(train_df["userId"].tolist()))

    cval = generate_candidates(model, users_val, k=k_candidates, filter_seen=True)
    ctest = generate_candidates(model, users_test, k=k_candidates, filter_seen=True)

    cval.to_csv(out_dir / "candidates_val_baseline.csv", index=False)
    ctest.to_csv(out_dir / "candidates_test_baseline.csv", index=False)
    print(f"algo={model.algorithm} val_rows={len(cval)} test_rows={len(ctest)}")


if __name__ == "__main__":
    main()
