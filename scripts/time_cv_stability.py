#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.pipeline import run_pipeline


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _run_one(base_cfg: dict, split_offset: int, run_tag: str) -> pd.DataFrame:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("data", {})["split_offset"] = int(split_offset)

    base_tables = Path(base_cfg.get("outputs", {}).get("tables_dir", "outputs/tables"))
    base_figs = Path(base_cfg.get("outputs", {}).get("figures_dir", "outputs/figures"))
    cfg.setdefault("outputs", {})["tables_dir"] = str(base_tables / "time_cv" / run_tag / f"split_{split_offset}")
    cfg.setdefault("outputs", {})["figures_dir"] = str(base_figs / "time_cv" / run_tag / f"split_{split_offset}")

    out = run_pipeline(cfg)
    comp = out["comparison"].copy()
    comp["split_offset"] = int(split_offset)
    return comp


def _metric_row(comp: pd.DataFrame, model: str, k: int) -> pd.Series:
    row = comp[(comp["model"] == model) & (comp["k"] == int(k))]
    if row.empty:
        raise RuntimeError(f"Missing row for model={model}, k={k}")
    return row.iloc[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal CV stability + ensemble calibration")
    p.add_argument("--config", default=str(ROOT / "configs" / "final_gpu.yaml"))
    p.add_argument("--eval-offsets", default="0,1,2", help="Comma-separated split offsets for final stability report")
    p.add_argument("--calib-offsets", default="1,2", help="Offsets used to calibrate ensemble weights")
    p.add_argument("--ensemble-baseline-weights", default="0.45,0.55,0.65", help="Candidate baseline weights")
    p.add_argument("--target-k", type=int, default=20)
    p.add_argument("--metric", choices=["recall", "ndcg", "map", "mrr"], default="ndcg")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)

    eval_offsets = _parse_ints(args.eval_offsets)
    calib_offsets = _parse_ints(args.calib_offsets)
    cand_wb = _parse_floats(args.ensemble_baseline_weights)

    base_cfg.setdefault("final_model", {}).setdefault("ranker_ensemble", {})["enabled"] = True

    calib_rows = []
    for wb in cand_wb:
        wf = 1.0 - wb
        cfg_w = copy.deepcopy(base_cfg)
        cfg_w["final_model"]["ranker_ensemble"]["baseline_weight"] = wb
        cfg_w["final_model"]["ranker_ensemble"]["final_weight"] = wf

        vals = []
        for off in calib_offsets:
            comp = _run_one(cfg_w, split_offset=off, run_tag=f"calib_b{int(round(wb*100)):02d}_f{int(round(wf*100)):02d}")
            row = _metric_row(comp, model="final_ranker_ensemble", k=args.target_k)
            vals.append(float(row[args.metric]))

        vals_s = pd.Series(vals)
        calib_rows.append(
            {
                "baseline_weight": wb,
                "final_weight": wf,
                f"calib_{args.metric}_mean": float(vals_s.mean()),
                f"calib_{args.metric}_std": float(vals_s.std(ddof=0)),
            }
        )
        print(
            f"calib wb={wb:.2f} wf={wf:.2f} {args.metric}_mean={vals_s.mean():.6f} "
            f"{args.metric}_std={vals_s.std(ddof=0):.6f}"
        )

    calib_df = pd.DataFrame(calib_rows).sort_values(
        [f"calib_{args.metric}_mean", f"calib_{args.metric}_std"], ascending=[False, True]
    )
    best = calib_df.iloc[0]
    best_wb = float(best["baseline_weight"])
    best_wf = float(best["final_weight"])

    print(f"BEST_WEIGHTS baseline={best_wb:.2f} final={best_wf:.2f}")

    cfg_best = copy.deepcopy(base_cfg)
    cfg_best["final_model"]["ranker_ensemble"]["baseline_weight"] = best_wb
    cfg_best["final_model"]["ranker_ensemble"]["final_weight"] = best_wf

    split_metrics = []
    for off in eval_offsets:
        comp = _run_one(cfg_best, split_offset=off, run_tag="stability_best")
        keep = comp[comp["k"] == int(args.target_k)].copy()
        keep["split_offset"] = int(off)
        split_metrics.append(keep)

    split_df = pd.concat(split_metrics, ignore_index=True)

    summary = (
        split_df.groupby("model", as_index=False)
        .agg(
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            ndcg_mean=("ndcg", "mean"),
            ndcg_std=("ndcg", "std"),
            map_mean=("map", "mean"),
            map_std=("map", "std"),
            mrr_mean=("mrr", "mean"),
            mrr_std=("mrr", "std"),
        )
        .fillna(0.0)
        .sort_values([f"{args.metric}_mean", f"{args.metric}_std"], ascending=[False, True])
    )

    out_root = Path(base_cfg.get("outputs", {}).get("tables_dir", "outputs/tables")) / "time_cv"
    out_root.mkdir(parents=True, exist_ok=True)
    calib_path = out_root / "calibration_summary.csv"
    split_path = out_root / "split_metrics_k.csv"
    summary_path = out_root / "stability_summary.csv"

    calib_df.to_csv(calib_path, index=False)
    split_df.to_csv(split_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"saved={calib_path}")
    print(f"saved={split_path}")
    print(f"saved={summary_path}")
    print("\nSTABILITY SUMMARY:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
