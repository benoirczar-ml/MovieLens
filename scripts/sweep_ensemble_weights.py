#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.data.io import load_ratings, temporal_leave_last_split
from recsys_ml25m.eval.offline import evaluate_predictions
from recsys_ml25m.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep baseline/final weights in ranker ensemble")
    p.add_argument("--config", default=str(ROOT / "configs" / "final.yaml"))
    p.add_argument(
        "--baseline-weights",
        default="0.45,0.55,0.65",
        help="Comma-separated weights for baseline ranker in final_ranker_ensemble",
    )
    p.add_argument("--target-k", type=int, default=20)
    p.add_argument("--metric", default="ndcg", choices=["recall", "ndcg", "map", "mrr"])
    p.add_argument(
        "--scores-dir",
        default=None,
        help="Optional directory with baseline_scores.pkl/final_scores.pkl from a previous run",
    )
    return p.parse_args()


def _rank_norm_scores(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    ranked = df.sort_values(["userId", score_col], ascending=[True, False], kind="mergesort").copy()
    ranked["rank_pos"] = ranked.groupby("userId").cumcount() + 1
    ranked["rank_score_norm"] = 1.0 / ranked["rank_pos"].astype("float32")
    out = ranked[["userId", "movieId", "rank_score_norm"]].copy()
    out["userId"] = out["userId"].astype("int32")
    out["movieId"] = out["movieId"].astype("int32")
    out["rank_score_norm"] = out["rank_score_norm"].astype("float32")
    return out


def _load_ground_truth_from_cfg(cfg: dict, scores_dir: Path) -> pd.DataFrame:
    gt_path = scores_dir / "test_ground_truth.pkl"
    if gt_path.exists():
        gt = pd.read_pickle(gt_path)[["userId", "movieId"]].copy()
        gt["userId"] = gt["userId"].astype("int32")
        gt["movieId"] = gt["movieId"].astype("int32")
        return gt

    data_cfg = cfg.get("data", {})
    ratings = load_ratings(
        data_dir=data_cfg["dataset_dir"],
        min_rating=float(data_cfg.get("min_rating", 0.0)),
        max_rows=data_cfg.get("max_rows"),
    )
    _, _, test_df = temporal_leave_last_split(
        ratings,
        val_k=int(data_cfg.get("val_k", 1)),
        test_k=int(data_cfg.get("test_k", 1)),
        min_user_interactions=int(data_cfg.get("min_user_interactions", 5)),
        split_offset=int(data_cfg.get("split_offset", 0)),
    )
    gt = test_df[["userId", "movieId"]].drop_duplicates().copy()
    gt["userId"] = gt["userId"].astype("int32")
    gt["movieId"] = gt["movieId"].astype("int32")
    return gt


def _run_fast_sweep(
    cfg: dict,
    weights: list[float],
    target_k: int,
    metric: str,
    scores_dir: Path,
) -> pd.DataFrame:
    baseline_path = scores_dir / "baseline_scores.pkl"
    final_path = scores_dir / "final_scores.pkl"
    if not baseline_path.exists() or not final_path.exists():
        raise FileNotFoundError(
            f"Missing saved scores in {scores_dir}. Expected baseline_scores.pkl and final_scores.pkl."
        )

    baseline_scores = pd.read_pickle(baseline_path)
    final_scores = pd.read_pickle(final_path)
    gt = _load_ground_truth_from_cfg(cfg, scores_dir)

    b = _rank_norm_scores(baseline_scores, "rank_score").rename(columns={"rank_score_norm": "score_base"})
    f = _rank_norm_scores(final_scores, "rank_score").rename(columns={"rank_score_norm": "score_final"})
    merged = b.merge(f, on=["userId", "movieId"], how="outer")
    merged["score_base"] = merged["score_base"].fillna(0.0).astype("float32")
    merged["score_final"] = merged["score_final"].fillna(0.0).astype("float32")
    merged["userId"] = merged["userId"].astype("int32")
    merged["movieId"] = merged["movieId"].astype("int32")

    rows: list[dict] = []
    for wb in weights:
        wf = 1.0 - wb
        pred = merged[["userId", "movieId"]].copy()
        pred["rank_score"] = (wb * merged["score_base"] + wf * merged["score_final"]).astype("float32")
        m = evaluate_predictions(
            ground_truth_df=gt,
            prediction_df=pred,
            score_col="rank_score",
            ks=[int(target_k)],
            model_name="final_ranker_ensemble",
        ).iloc[0]
        rows.append(
            {
                "baseline_weight": wb,
                "final_weight": wf,
                "k": int(target_k),
                "ranker_recall": float(m["recall"]),
                "ranker_ndcg": float(m["ndcg"]),
                "ranker_map": float(m["map"]),
                "ranker_mrr": float(m["mrr"]),
                "mode": "fast_saved_scores",
                "scores_dir": str(scores_dir),
            }
        )
        print(
            f"fast wb={wb:.2f} wf={wf:.2f} k={int(target_k)} "
            f"ranker:{metric}={float(m[metric]):.6f} recall={float(m['recall']):.6f} ndcg={float(m['ndcg']):.6f}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)

    weights = [float(x.strip()) for x in args.baseline_weights.split(",") if x.strip()]
    if not weights:
        raise ValueError("No valid weights provided")

    if args.scores_dir:
        df = _run_fast_sweep(
            cfg=base_cfg,
            weights=weights,
            target_k=int(args.target_k),
            metric=args.metric,
            scores_dir=Path(args.scores_dir),
        )
        out_root = Path(base_cfg.get("outputs", {}).get("tables_dir", "outputs/tables"))
        out_root.mkdir(parents=True, exist_ok=True)
        out_csv = out_root / f"ensemble_sweep_summary_k{int(args.target_k)}_fast.csv"
        df = df.sort_values(
            [f"ranker_{args.metric}", "ranker_map", "ranker_mrr", "ranker_recall"],
            ascending=[False, False, False, False],
        )
        df.to_csv(out_csv, index=False)
        best = df.iloc[0]
        print("\nBEST:")
        print(
            f"baseline_weight={best['baseline_weight']:.2f} final_weight={best['final_weight']:.2f} "
            f"ranker_recall={best['ranker_recall']:.6f} ranker_ndcg={best['ranker_ndcg']:.6f} "
            f"ranker_map={best['ranker_map']:.6f} ranker_mrr={best['ranker_mrr']:.6f}"
        )
        print(f"saved={out_csv}")
        return

    rows: list[dict] = []

    for wb in weights:
        wf = 1.0 - wb
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("final_model", {}).setdefault("ranker_ensemble", {})["enabled"] = True
        cfg["final_model"]["ranker_ensemble"]["baseline_weight"] = wb
        cfg["final_model"]["ranker_ensemble"]["final_weight"] = wf

        tag = f"ens_b{int(round(wb * 100)):02d}_f{int(round(wf * 100)):02d}"
        base_tables = Path(base_cfg.get("outputs", {}).get("tables_dir", "outputs/tables"))
        base_figures = Path(base_cfg.get("outputs", {}).get("figures_dir", "outputs/figures"))
        cfg.setdefault("outputs", {})["tables_dir"] = str(base_tables / "ens_sweeps" / tag)
        cfg.setdefault("outputs", {})["figures_dir"] = str(base_figures / "ens_sweeps" / tag)

        out = run_pipeline(cfg)
        comp = out["comparison"]

        final_row = comp[(comp["model"] == "final_ranker_ensemble") & (comp["k"] == int(args.target_k))]
        if final_row.empty:
            continue

        fr = final_row.iloc[0]
        rows.append(
            {
                "baseline_weight": wb,
                "final_weight": wf,
                "k": int(args.target_k),
                "ranker_recall": float(fr["recall"]),
                "ranker_ndcg": float(fr["ndcg"]),
                "ranker_map": float(fr["map"]),
                "ranker_mrr": float(fr["mrr"]),
                "tables_dir": str(cfg["outputs"]["tables_dir"]),
            }
        )

        print(
            f"run={tag} k={args.target_k} ranker:{args.metric}={float(fr[args.metric]):.6f} "
            f"recall={float(fr['recall']):.6f} ndcg={float(fr['ndcg']):.6f}"
        )

    if not rows:
        raise RuntimeError("No sweep runs produced metrics")

    df = pd.DataFrame(rows).sort_values(
        [f"ranker_{args.metric}", "ranker_map", "ranker_mrr", "ranker_recall"],
        ascending=[False, False, False, False],
    )

    out_root = Path(base_cfg.get("outputs", {}).get("tables_dir", "outputs/tables"))
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "ensemble_sweep_summary.csv"
    df.to_csv(out_csv, index=False)

    best = df.iloc[0]
    print("\nBEST:")
    print(
        f"baseline_weight={best['baseline_weight']:.2f} final_weight={best['final_weight']:.2f} "
        f"ranker_recall={best['ranker_recall']:.6f} ranker_ndcg={best['ranker_ndcg']:.6f} "
        f"ranker_map={best['ranker_map']:.6f} ranker_mrr={best['ranker_mrr']:.6f}"
    )
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()
