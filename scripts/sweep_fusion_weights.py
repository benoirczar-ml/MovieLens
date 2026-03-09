#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep candidate fusion weights (2-way or 3-way)")
    p.add_argument("--config", default=str(ROOT / "configs" / "final.yaml"))
    p.add_argument(
        "--two-tower-weights",
        default="0.45,0.55,0.65,0.75,0.85",
        help="Comma-separated weights for two tower in candidate fusion",
    )
    p.add_argument(
        "--aux-weights",
        default="",
        help="Optional comma-separated weights for auxiliary retrieval source (e.g. 0.05,0.10).",
    )
    p.add_argument(
        "--aux-algorithm",
        default="cosine",
        choices=["als", "bpr", "bm25", "cosine", "tfidf", "popular"],
        help="Algorithm used for auxiliary retrieval when aux-weights are provided.",
    )
    p.add_argument("--aux-itemknn-k", type=int, default=200)
    p.add_argument("--aux-itemknn-num-threads", type=int, default=0)
    p.add_argument("--target-k", type=int, default=20)
    p.add_argument("--metric", default="ndcg", choices=["recall", "ndcg", "map", "mrr"])
    return p.parse_args()


def _pick_model_row(comp: pd.DataFrame, prefix: str, target_k: int) -> pd.Series | None:
    rows = comp[(comp["k"] == int(target_k)) & (comp["model"].astype(str).str.startswith(prefix))]
    if rows.empty:
        return None
    return rows.iloc[-1]


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)

    weights = [float(x.strip()) for x in args.two_tower_weights.split(",") if x.strip()]
    aux_weights = [float(x.strip()) for x in args.aux_weights.split(",") if x.strip()]
    if not aux_weights:
        aux_weights = [0.0]
    rows: list[dict] = []

    for w_aux in aux_weights:
        for w_tt in weights:
            w_als = 1.0 - w_tt - w_aux
            if w_als <= 0.0:
                continue

            cfg = copy.deepcopy(base_cfg)
            cf = cfg.setdefault("final_model", {}).setdefault("candidate_fusion", {})
            cf["enabled"] = True
            cf["two_tower_weight"] = w_tt
            cf["als_weight"] = w_als

            if w_aux > 0.0:
                cf["aux_retrieval"] = {
                    "enabled": True,
                    "algorithm": args.aux_algorithm,
                    "weight": w_aux,
                    "itemknn_k": int(args.aux_itemknn_k),
                    "itemknn_num_threads": int(args.aux_itemknn_num_threads),
                    "require_gpu": False,
                }
            else:
                cf["aux_retrieval"] = {"enabled": False, "weight": 0.0}

            run_tag = (
                f"tt_{int(round(w_tt * 100)):02d}_als_{int(round(w_als * 100)):02d}"
                f"_aux_{int(round(w_aux * 100)):02d}"
            )
            base_tables = Path(base_cfg.get("outputs", {}).get("tables_dir", "outputs/tables"))
            base_figures = Path(base_cfg.get("outputs", {}).get("figures_dir", "outputs/figures"))
            cfg.setdefault("outputs", {})["tables_dir"] = str(base_tables / "sweeps" / run_tag)
            cfg.setdefault("outputs", {})["figures_dir"] = str(base_figures / "sweeps" / run_tag)

            out = run_pipeline(cfg)
            comp = out["comparison"]

            fr = _pick_model_row(comp, "final_ranker_two_tower_hybrid", int(args.target_k))
            rr = _pick_model_row(comp, "final_retrieval_two_tower_faiss_hybrid", int(args.target_k))
            if fr is None or rr is None:
                continue

            rows.append(
                {
                    "two_tower_weight": w_tt,
                    "als_weight": w_als,
                    "aux_weight": w_aux,
                    "aux_algorithm": args.aux_algorithm if w_aux > 0.0 else "none",
                    "k": int(args.target_k),
                    "ranker_recall": float(fr["recall"]),
                    "ranker_ndcg": float(fr["ndcg"]),
                    "ranker_map": float(fr["map"]),
                    "ranker_mrr": float(fr["mrr"]),
                    "retrieval_recall": float(rr["recall"]),
                    "retrieval_ndcg": float(rr["ndcg"]),
                    "tables_dir": str(cfg["outputs"]["tables_dir"]),
                }
            )

            print(
                f"run={run_tag} k={args.target_k} ranker:{args.metric}={float(fr[args.metric]):.6f} "
                f"recall={float(fr['recall']):.6f} ndcg={float(fr['ndcg']):.6f}"
            )

    if not rows:
        raise RuntimeError("No sweep runs produced metrics")

    df = pd.DataFrame(rows).sort_values(
        [f"ranker_{args.metric}", "ranker_recall", "ranker_ndcg", "ranker_map", "ranker_mrr"],
        ascending=[False, False, False, False, False],
    )

    out_root = Path(base_cfg.get("outputs", {}).get("tables_dir", "outputs/tables"))
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "fusion_sweep_summary.csv"
    df.to_csv(out_csv, index=False)

    best = df.iloc[0]
    print("\nBEST:")
    print(
        f"two_tower_weight={best['two_tower_weight']:.2f} als_weight={best['als_weight']:.2f} "
        f"aux_weight={best['aux_weight']:.2f} aux_algorithm={best['aux_algorithm']} "
        f"ranker_recall={best['ranker_recall']:.6f} ranker_ndcg={best['ranker_ndcg']:.6f} "
        f"ranker_map={best['ranker_map']:.6f} ranker_mrr={best['ranker_mrr']:.6f}"
    )
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()
