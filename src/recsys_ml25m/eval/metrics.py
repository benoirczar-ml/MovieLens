from __future__ import annotations

import math


def recall_at_k(ground_truth: dict[int, set[int]], predictions: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, gt in ground_truth.items():
        if not gt:
            continue
        pred = set(predictions.get(user, [])[:k])
        vals.append(len(pred & gt) / len(gt))
    return float(sum(vals) / len(vals)) if vals else 0.0


def hitrate_at_k(ground_truth: dict[int, set[int]], predictions: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, gt in ground_truth.items():
        if not gt:
            continue
        pred = set(predictions.get(user, [])[:k])
        vals.append(1.0 if (pred & gt) else 0.0)
    return float(sum(vals) / len(vals)) if vals else 0.0


def ndcg_at_k(ground_truth: dict[int, set[int]], predictions: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, gt in ground_truth.items():
        if not gt:
            continue
        pred = predictions.get(user, [])[:k]
        dcg = 0.0
        for rank, item in enumerate(pred, start=1):
            if item in gt:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_hits = min(len(gt), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        vals.append(dcg / idcg if idcg > 0 else 0.0)

    return float(sum(vals) / len(vals)) if vals else 0.0


def map_at_k(ground_truth: dict[int, set[int]], predictions: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, gt in ground_truth.items():
        if not gt:
            continue
        pred = predictions.get(user, [])[:k]
        hits = 0
        ap = 0.0
        for rank, item in enumerate(pred, start=1):
            if item in gt:
                hits += 1
                ap += hits / rank
        denom = min(len(gt), k)
        vals.append(ap / denom if denom > 0 else 0.0)

    return float(sum(vals) / len(vals)) if vals else 0.0


def mrr_at_k(ground_truth: dict[int, set[int]], predictions: dict[int, list[int]], k: int) -> float:
    vals = []
    for user, gt in ground_truth.items():
        if not gt:
            continue
        pred = predictions.get(user, [])[:k]
        rr = 0.0
        for rank, item in enumerate(pred, start=1):
            if item in gt:
                rr = 1.0 / rank
                break
        vals.append(rr)
    return float(sum(vals) / len(vals)) if vals else 0.0
