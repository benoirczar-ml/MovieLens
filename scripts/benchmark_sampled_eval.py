#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.data.io import load_ratings, temporal_leave_last_split
from recsys_ml25m.eval.metrics import hitrate_at_k, ndcg_at_k, recall_at_k
from recsys_ml25m.retrieval.als import build_retrieval_model, generate_candidates
from recsys_ml25m.retrieval.two_tower import train_two_tower_faiss


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _ranknorm(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    out = np.zeros_like(scores, dtype=np.float32)
    out[order] = 1.0 / (1.0 + np.arange(len(scores), dtype=np.float32))
    return out


def _sample_eval_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_ids: np.ndarray,
    n_negatives: int,
    max_users: int | None,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, set[int]]]:
    rng = np.random.default_rng(seed)
    pos_df = test_df.sort_values(["userId", "timestamp"], kind="mergesort").groupby("userId", as_index=False).tail(1)
    users = pos_df["userId"].to_numpy(dtype=np.int32)
    positives = pos_df["movieId"].to_numpy(dtype=np.int32)

    if max_users is not None and max_users > 0 and len(users) > max_users:
        idx = rng.choice(len(users), size=max_users, replace=False)
        users = users[idx]
        positives = positives[idx]

    seen_map = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    eval_candidates: dict[int, np.ndarray] = {}
    gt: dict[int, set[int]] = {}
    item_ids = item_ids.astype(np.int32)

    for uid, pos in zip(users, positives):
        seen = seen_map.get(int(uid), set())
        taken = {int(pos)}
        negs: list[int] = []

        # Rejection sampling is cheap here because user history is small vs item catalog.
        while len(negs) < n_negatives:
            draw = rng.choice(item_ids, size=max(256, n_negatives * 2), replace=True)
            for cand in draw:
                c = int(cand)
                if c in seen or c in taken:
                    continue
                taken.add(c)
                negs.append(c)
                if len(negs) >= n_negatives:
                    break

        eval_candidates[int(uid)] = np.asarray([int(pos)] + negs[:n_negatives], dtype=np.int32)
        gt[int(uid)] = {int(pos)}

    return eval_candidates, gt


def _score_als(artifacts, eval_candidates: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    if getattr(artifacts, "algorithm", "") == "lightgcn":
        pred_scores: dict[int, np.ndarray] = {}
        lg = artifacts.model
        u_map = lg.user_to_idx
        i_map = lg.item_to_idx
        item_pop = lg.item_popularity
        blend = float(lg.pop_blend_weight)
        for uid, items in eval_candidates.items():
            uidx = u_map.get(int(uid))
            scores = np.full(len(items), -1e9, dtype=np.float32)
            if uidx is not None:
                mapped = np.asarray([i_map.get(int(mid), -1) for mid in items], dtype=np.int64)
                valid = mapped >= 0
                if np.any(valid):
                    uemb = lg.user_emb[int(uidx)]
                    iemb = lg.item_emb[mapped[valid]]
                    base = (iemb @ uemb).astype(np.float32)
                    pop = item_pop[mapped[valid]].astype(np.float32)
                    scores[valid] = (1.0 - blend) * base + blend * pop
            pred_scores[int(uid)] = scores
        return pred_scores

    pred_scores: dict[int, np.ndarray] = {}
    model = artifacts.model
    user_to_idx = artifacts.user_to_idx
    item_to_idx = artifacts.item_to_idx

    has_factors = hasattr(model, "user_factors") and hasattr(model, "item_factors")
    if not has_factors:
        ranked = getattr(model, "ranked_items", np.array([], dtype=np.int32))
        rank_map = {int(mid): float(1.0 / (i + 1)) for i, mid in enumerate(ranked[:50000])}
        for uid, items in eval_candidates.items():
            scores = np.asarray([rank_map.get(int(mid), 0.0) for mid in items], dtype=np.float32)
            pred_scores[uid] = scores
        return pred_scores

    # implicit GPU stores factors as CUDA Matrix objects; convert once for fast numpy scoring.
    if not isinstance(model.user_factors, np.ndarray) and hasattr(model, "to_cpu"):
        model = model.to_cpu()

    for uid, items in eval_candidates.items():
        uidx = user_to_idx.get(int(uid))
        scores = np.full(len(items), -1e9, dtype=np.float32)
        if uidx is not None:
            mapped = np.asarray([item_to_idx.get(int(mid), -1) for mid in items], dtype=np.int64)
            valid = mapped >= 0
            if np.any(valid):
                uf = model.user_factors[int(uidx)]
                iv = model.item_factors[mapped[valid]]
                scores[valid] = (iv @ uf).astype(np.float32)
        pred_scores[uid] = scores
    return pred_scores


def _score_two_tower(tt_art, eval_candidates: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    pred_scores: dict[int, np.ndarray] = {}
    u_map = tt_art.user_to_idx
    i_map = tt_art.item_to_idx
    item_pop = tt_art.item_popularity
    blend = float(tt_art.pop_blend_weight)

    for uid, items in eval_candidates.items():
        uidx = u_map.get(int(uid))
        scores = np.full(len(items), -1e9, dtype=np.float32)
        if uidx is not None:
            mapped = np.asarray([i_map.get(int(mid), -1) for mid in items], dtype=np.int64)
            valid = mapped >= 0
            if np.any(valid):
                uemb = tt_art.user_emb[int(uidx)]
                iemb = tt_art.item_emb[mapped[valid]]
                base = (iemb @ uemb).astype(np.float32)
                pop = item_pop[mapped[valid]].astype(np.float32)
                scores[valid] = (1.0 - blend) * base + blend * pop
        pred_scores[uid] = scores
    return pred_scores


def _train_itemknn_models(user_item, bench_cfg: dict) -> dict[str, object]:
    from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender

    names = [str(x).lower() for x in bench_cfg.get("itemknn_models", ["bm25"])]
    k = int(bench_cfg.get("itemknn_k", 200))
    num_threads = int(bench_cfg.get("itemknn_num_threads", 0))
    user_items = user_item.tocsr()

    out: dict[str, object] = {}
    for name in names:
        if name == "bm25":
            model = BM25Recommender(
                K=k,
                K1=float(bench_cfg.get("bm25_k1", 1.2)),
                B=float(bench_cfg.get("bm25_b", 0.75)),
                num_threads=num_threads,
            )
        elif name == "cosine":
            model = CosineRecommender(K=k, num_threads=num_threads)
        elif name == "tfidf":
            model = TFIDFRecommender(K=k, num_threads=num_threads)
        else:
            continue
        _log(f"Train itemknn model={name} K={k}")
        model.fit(user_items)
        out[name] = model
    return out


def _score_itemknn(model, artifacts, eval_candidates: dict[int, np.ndarray], model_name: str = "itemknn") -> dict[int, np.ndarray]:
    similarity = model.similarity.tocsr()
    user_item = artifacts.user_item
    u_map = artifacts.user_to_idx
    i_map = artifacts.item_to_idx

    pred_scores: dict[int, np.ndarray] = {}
    total = len(eval_candidates)
    for n, (uid, items) in enumerate(eval_candidates.items(), start=1):
        uidx = u_map.get(int(uid))
        scores = np.full(len(items), -1e9, dtype=np.float32)
        if uidx is not None:
            seen_idx = user_item[int(uidx)].indices
            if len(seen_idx) > 0:
                mapped = np.asarray([i_map.get(int(mid), -1) for mid in items], dtype=np.int64)
                valid = mapped >= 0
                if np.any(valid):
                    # Score candidate by sum of item-item similarities against user history.
                    sub = similarity[mapped[valid]][:, seen_idx]
                    scores[valid] = np.asarray(sub.sum(axis=1)).ravel().astype(np.float32)
        pred_scores[int(uid)] = scores
        if n % 5000 == 0 or n == total:
            pct = 100.0 * n / max(1, total)
            _log(f"Score sampled candidate sets: {model_name} progress users={n:,}/{total:,} ({pct:.1f}%)")
    return pred_scores


def _to_prediction_lists(scores_by_user: dict[int, np.ndarray], eval_candidates: dict[int, np.ndarray]) -> dict[int, list[int]]:
    pred: dict[int, list[int]] = {}
    for uid, scores in scores_by_user.items():
        items = eval_candidates[uid]
        order = np.argsort(-scores, kind="mergesort")
        pred[uid] = items[order].tolist()
    return pred


def _score_from_retrieval_candidates(
    artifacts,
    eval_candidates: dict[int, np.ndarray],
    k_retrieval: int,
    model_name: str,
) -> dict[int, np.ndarray]:
    users = list(eval_candidates.keys())
    cand = generate_candidates(artifacts, user_ids=users, k=int(k_retrieval), filter_seen=True, log_fn=_log)
    if cand.empty:
        return {uid: np.full(len(items), -1e9, dtype=np.float32) for uid, items in eval_candidates.items()}

    cand_map: dict[int, dict[int, float]] = {}
    for uid, g in cand.groupby("userId", sort=False):
        cand_map[int(uid)] = {int(mid): float(sc) for mid, sc in zip(g["movieId"].tolist(), g["retrieval_score"].tolist())}

    pred_scores: dict[int, np.ndarray] = {}
    for uid, items in eval_candidates.items():
        src = cand_map.get(int(uid), {})
        arr = np.asarray([float(src.get(int(mid), -1e9)) for mid in items], dtype=np.float32)
        pred_scores[int(uid)] = arr
    _log(f"Score sampled candidate sets: {model_name} via generated topK={int(k_retrieval)}")
    return pred_scores


def _eval_rows(model_name: str, gt: dict[int, set[int]], pred: dict[int, list[int]], ks: list[int]) -> list[dict]:
    rows = []
    for k in ks:
        rows.append(
            {
                "model": model_name,
                "k": int(k),
                "recall": recall_at_k(gt, pred, int(k)),
                "hitrate": hitrate_at_k(gt, pred, int(k)),
                "ndcg": ndcg_at_k(gt, pred, int(k)),
                "users": len(gt),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark sampled-negative protocol (leave-one-out)")
    p.add_argument("--config", default=str(ROOT / "configs" / "fullrun_gpu_robust.yaml"))
    p.add_argument("--n-negatives", type=int, default=None)
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--save-parquet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    retrieval_cfg = cfg.get("retrieval", {})
    final_cfg = cfg.get("final_model", {})
    bench_cfg = cfg.get("benchmark", {})
    out_cfg = cfg.get("outputs", {})

    n_negatives = int(args.n_negatives if args.n_negatives is not None else bench_cfg.get("n_negatives", 100))
    max_users = args.max_users if args.max_users is not None else bench_cfg.get("max_users", 20000)
    seed = int(args.seed if args.seed is not None else bench_cfg.get("seed", 42))
    ks = [int(k) for k in bench_cfg.get("ks", [10, 20])]

    out_path = Path(args.output) if args.output else Path(out_cfg.get("tables_dir", ROOT / "outputs" / "tables")) / "benchmark_sampled_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _log(
        f"Benchmark sampled protocol start | n_negatives={n_negatives} max_users={max_users} ks={ks} seed={seed}"
    )
    ratings = load_ratings(
        data_dir=data_cfg["dataset_dir"],
        min_rating=float(data_cfg.get("min_rating", 0.0)),
        max_rows=data_cfg.get("max_rows"),
    )
    train_df, _, test_df = temporal_leave_last_split(
        ratings,
        val_k=int(data_cfg.get("val_k", 1)),
        test_k=int(data_cfg.get("test_k", 1)),
        min_user_interactions=int(data_cfg.get("min_user_interactions", 5)),
        split_offset=int(data_cfg.get("split_offset", 0)),
    )
    item_ids = np.sort(train_df["movieId"].unique())
    _log(
        f"Data ready | train={len(train_df):,} test={len(test_df):,} users={train_df['userId'].nunique():,} items={len(item_ids):,}"
    )

    eval_candidates, gt = _sample_eval_candidates(
        train_df=train_df,
        test_df=test_df,
        item_ids=item_ids,
        n_negatives=n_negatives,
        max_users=max_users,
        seed=seed,
    )
    _log(f"Sampled candidate sets built | users_eval={len(eval_candidates):,} candidates_per_user={n_negatives + 1}")

    _log("Train ALS retrieval for sampled benchmark")
    bas_art = build_retrieval_model(train_df, retrieval_cfg, log_fn=_log)
    _log("Train Two-Tower for sampled benchmark")
    tt_art = train_two_tower_faiss(train_df, final_cfg.get("two_tower", {}), log_fn=_log)

    _log("Score sampled candidate sets: ALS")
    als_scores = _score_als(bas_art, eval_candidates)
    _log("Score sampled candidate sets: Two-Tower")
    tt_scores = _score_two_tower(tt_art, eval_candidates)
    knn_models = _train_itemknn_models(bas_art.user_item, bench_cfg)
    knn_scores: dict[str, dict[int, np.ndarray]] = {}
    for knn_name, knn_model in knn_models.items():
        _log(f"Score sampled candidate sets: {knn_name}")
        knn_scores[knn_name] = _score_itemknn(knn_model, bas_art, eval_candidates, model_name=knn_name)

    fusion_cfg = final_cfg.get("candidate_fusion", {})
    w_tt = float(fusion_cfg.get("two_tower_weight", 0.45))
    w_als = float(fusion_cfg.get("als_weight", 0.55))
    retrieval_eval_k = int(bench_cfg.get("retrieval_eval_k", max(400, n_negatives + 1)))

    source_scores: dict[str, dict[int, np.ndarray]] = {
        "als": als_scores,
        "tt": tt_scores,
    }
    source_weights: dict[str, float] = {
        "als": w_als,
        "tt": w_tt,
    }

    aux_cfg_list = fusion_cfg.get("aux_retrievals")
    if isinstance(aux_cfg_list, list):
        raw_aux_cfgs = [x for x in aux_cfg_list if isinstance(x, dict)]
    else:
        single_aux = fusion_cfg.get("aux_retrieval", {})
        raw_aux_cfgs = [single_aux] if isinstance(single_aux, dict) else []

    for aux_idx, aux_cfg in enumerate(raw_aux_cfgs, start=1):
        use_aux = bool(aux_cfg.get("enabled", False))
        w_aux = float(aux_cfg.get("weight", 0.0))
        aux_algo = str(aux_cfg.get("algorithm", "none")).lower()
        if (not use_aux) or w_aux <= 0.0:
            continue

        aux_name = str(aux_cfg.get("name", f"aux{aux_idx}_{aux_algo}")).strip().lower().replace(" ", "_")
        if not aux_name:
            aux_name = f"aux{aux_idx}_{aux_algo}"

        aux_retr_cfg = dict(retrieval_cfg)
        aux_retr_cfg.update(
            {
                k: v
                for k, v in aux_cfg.items()
                if k not in {"enabled", "weight", "algorithm"}
            }
        )
        aux_retr_cfg["algorithm"] = aux_algo
        aux_retr_cfg["require_gpu"] = bool(aux_cfg.get("require_gpu", False))
        if aux_algo in {"bm25", "cosine", "tfidf"}:
            aux_retr_cfg["use_gpu"] = False

        _log(f"Train aux retrieval for sampled benchmark name={aux_name} algo={aux_algo}")
        aux_art = build_retrieval_model(train_df, aux_retr_cfg, log_fn=_log)
        aux_scores = _score_from_retrieval_candidates(
            artifacts=aux_art,
            eval_candidates=eval_candidates,
            k_retrieval=retrieval_eval_k,
            model_name=aux_name,
        )
        source_scores[aux_name] = aux_scores
        source_weights[aux_name] = w_aux

    weight_sum = sum(max(0.0, float(v)) for v in source_weights.values())
    if weight_sum <= 0.0:
        weight_sum = 1.0

    hyb_scores: dict[int, np.ndarray] = {}
    for uid in eval_candidates.keys():
        combined = None
        for name, scores in source_scores.items():
            w = float(source_weights.get(name, 0.0)) / weight_sum
            if w <= 0.0:
                continue
            normed = _ranknorm(scores[uid])
            combined = (w * normed) if combined is None else (combined + w * normed)
        if combined is None:
            combined = np.zeros(len(eval_candidates[uid]), dtype=np.float32)
        hyb_scores[uid] = combined.astype(np.float32)

    pred_als = _to_prediction_lists(als_scores, eval_candidates)
    pred_tt = _to_prediction_lists(tt_scores, eval_candidates)
    pred_hyb = _to_prediction_lists(hyb_scores, eval_candidates)
    pred_aux = {
        name: _to_prediction_lists(sc, eval_candidates)
        for name, sc in source_scores.items()
        if name not in {"als", "tt"}
    }
    pred_knn = {name: _to_prediction_lists(sc, eval_candidates) for name, sc in knn_scores.items()}

    rows = []
    rows += _eval_rows("als_sampled", gt, pred_als, ks)
    rows += _eval_rows("two_tower_sampled", gt, pred_tt, ks)
    rows += _eval_rows("hybrid_sampled", gt, pred_hyb, ks)
    for aux_name, pred in pred_aux.items():
        rows += _eval_rows(f"{aux_name}_sampled", gt, pred, ks)
    for knn_name, pred in pred_knn.items():
        rows += _eval_rows(f"{knn_name}_sampled", gt, pred, ks)

    out = pd.DataFrame(rows).sort_values(["model", "k"], kind="mergesort").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    _log(f"Saved benchmark: {out_path}")
    save_parquet = bool(args.save_parquet or out_cfg.get("save_parquet_tables", False))
    if save_parquet:
        parquet_dir = Path(out_cfg.get("parquet_tables_dir", out_path.parent.parent / "parquet"))
        parquet_dir.mkdir(parents=True, exist_ok=True)
        pq_path = parquet_dir / f"{out_path.stem}.parquet"
        try:
            out.to_parquet(pq_path, index=False)
            _log(f"Saved benchmark parquet: {pq_path}")
        except Exception as e:
            _log(f"benchmark parquet fallback reason={type(e).__name__}: {e}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
