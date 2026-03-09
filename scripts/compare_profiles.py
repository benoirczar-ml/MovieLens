#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare fullrun profile metrics and stage timings")
    p.add_argument(
        "--profile",
        action="append",
        required=True,
        help="Profile spec: name=<id>,metrics=<path>,log=<path>",
    )
    p.add_argument(
        "--model",
        default="final_ranker_ensemble",
        help="Model row to extract from metrics CSV (default: final_ranker_ensemble)",
    )
    p.add_argument("--k", type=int, default=20, help="Target K for metric extraction")
    p.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "tables" / "profile_compare_fullrun_k20.csv"),
        help="Output CSV path",
    )
    return p.parse_args()


def parse_spec(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise ValueError(f"Invalid profile part '{part}', expected key=value")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    for req in ("name", "metrics", "log"):
        if req not in out:
            raise ValueError(f"Missing '{req}' in profile spec: {spec}")
    return out


def _match_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return float(m.group(1))


def parse_log(log_path: Path) -> dict[str, float | int | None]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    gpu_vals = [int(x) for x in re.findall(r"gpu=(\d+)%", text)]
    vram_vals = [int(x) for x in re.findall(r"vram=(\d+)%", text)]
    return {
        "pipeline_s": _match_float(r"Pipeline done in ([0-9.]+)s", text),
        "twotower_train_s": _match_float(r"Stage final\.twotower\.train done in ([0-9.]+)s", text),
        "twotower_candidates_s": _match_float(r"Stage final\.twotower\.candidates done in ([0-9.]+)s", text),
        "candidate_fusion_s": _match_float(r"Stage final\.candidate_fusion done in ([0-9.]+)s", text),
        "final_ranker_features_s": _match_float(r"Stage final\.ranker\.features done in ([0-9.]+)s", text),
        "final_ranker_train_score_s": _match_float(r"Stage final\.ranker\.train\+score done in ([0-9.]+)s", text),
        "final_ranker_ensemble_s": _match_float(r"Stage final\.ranker\.ensemble done in ([0-9.]+)s", text),
        "max_gpu_pct": max(gpu_vals) if gpu_vals else None,
        "max_vram_pct": max(vram_vals) if vram_vals else None,
    }


def parse_metrics(metrics_path: Path, model: str, k: int) -> dict[str, float | int]:
    df = pd.read_csv(metrics_path)
    row = df[(df["model"] == model) & (df["k"] == k)]
    if row.empty:
        raise ValueError(f"No row for model='{model}' and k={k} in {metrics_path}")
    r = row.iloc[0]
    return {
        "recall": float(r["recall"]),
        "ndcg": float(r["ndcg"]),
        "map": float(r["map"]),
        "mrr": float(r["mrr"]),
        "users": int(r["users"]),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for spec in args.profile:
        p = parse_spec(spec)
        metrics_path = Path(p["metrics"])
        log_path = Path(p["log"])
        metrics = parse_metrics(metrics_path, model=args.model, k=args.k)
        log_stats = parse_log(log_path)
        rows.append(
            {
                "profile": p["name"],
                "metrics_path": str(metrics_path),
                "log_path": str(log_path),
                **metrics,
                **log_stats,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["ndcg", "recall"], ascending=False).reset_index(drop=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"saved={out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
