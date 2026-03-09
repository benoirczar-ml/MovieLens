from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class PopularityModel:
    ranked_items: np.ndarray


@dataclass
class RetrievalArtifacts:
    algorithm: str
    model: object
    user_item: sparse.csr_matrix
    user_to_idx: Dict[int, int]
    idx_to_user: np.ndarray
    item_to_idx: Dict[int, int]
    idx_to_item: np.ndarray
    recommend_batch_size: int
    log_every_batches: int


def _build_matrix(train_df: pd.DataFrame) -> tuple[sparse.csr_matrix, Dict[int, int], np.ndarray, Dict[int, int], np.ndarray]:
    unique_users = np.sort(train_df["userId"].unique())
    unique_items = np.sort(train_df["movieId"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_to_idx = {iid: i for i, iid in enumerate(unique_items)}

    rows = train_df["userId"].map(user_to_idx).to_numpy(dtype=np.int32)
    cols = train_df["movieId"].map(item_to_idx).to_numpy(dtype=np.int32)
    data = np.ones(len(train_df), dtype=np.float32)

    mat = sparse.coo_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items))).tocsr()
    return mat, user_to_idx, unique_users, item_to_idx, unique_items


def build_retrieval_model(train_df: pd.DataFrame, cfg: dict, log_fn: Callable[[str], None] | None = None) -> RetrievalArtifacts:
    algo = str(cfg.get("algorithm", "als")).lower()
    require_gpu = bool(cfg.get("require_gpu", False))
    user_item, user_to_idx, idx_to_user, item_to_idx, idx_to_item = _build_matrix(train_df)
    if log_fn:
        log_fn(
            f"baseline.retrieval.setup algo={algo} users={len(idx_to_user):,} "
            f"items={len(idx_to_item):,} interactions={len(train_df):,}"
        )

    if algo in {"als", "bpr"}:
        try:
            if algo == "als":
                from implicit.als import AlternatingLeastSquares

                use_gpu = bool(cfg.get("use_gpu", False))
                model = AlternatingLeastSquares(
                    factors=int(cfg.get("factors", 64)),
                    regularization=float(cfg.get("regularization", 0.01)),
                    iterations=int(cfg.get("iterations", 20)),
                    random_state=int(cfg.get("seed", 42)),
                    use_gpu=use_gpu,
                )
                alpha = float(cfg.get("alpha", 40.0))
                try:
                    t_fit = time.perf_counter()
                    model.fit((user_item * alpha).astype(np.float32))
                    if log_fn:
                        log_fn(f"baseline.retrieval.fit algo=als use_gpu={use_gpu} done_in={time.perf_counter() - t_fit:.1f}s")
                except Exception:
                    if use_gpu:
                        if require_gpu:
                            raise RuntimeError("ALS GPU training failed and require_gpu=true") from None
                        # Fallback to CPU ALS if GPU backend is unavailable.
                        model = AlternatingLeastSquares(
                            factors=int(cfg.get("factors", 64)),
                            regularization=float(cfg.get("regularization", 0.01)),
                            iterations=int(cfg.get("iterations", 20)),
                            random_state=int(cfg.get("seed", 42)),
                            use_gpu=False,
                        )
                        t_fit = time.perf_counter()
                        model.fit((user_item * alpha).astype(np.float32))
                        if log_fn:
                            log_fn(
                                "baseline.retrieval.fit algo=als fallback=cpu "
                                f"done_in={time.perf_counter() - t_fit:.1f}s"
                            )
                    else:
                        raise
            else:
                from implicit.bpr import BayesianPersonalizedRanking

                if require_gpu:
                    raise RuntimeError("BPR path does not support strict require_gpu mode in this pipeline")
                model = BayesianPersonalizedRanking(
                    factors=int(cfg.get("factors", 64)),
                    learning_rate=float(cfg.get("learning_rate", 0.05)),
                    regularization=float(cfg.get("regularization", 0.01)),
                    iterations=int(cfg.get("iterations", 100)),
                    random_state=int(cfg.get("seed", 42)),
                    verify_negative_samples=True,
                )
                t_fit = time.perf_counter()
                model.fit(user_item.astype(np.float32))
                if log_fn:
                    log_fn(f"baseline.retrieval.fit algo=bpr done_in={time.perf_counter() - t_fit:.1f}s")
        except Exception:
            if require_gpu:
                raise
            algo = "popular"
            item_pop = train_df.groupby("movieId").size().sort_values(ascending=False).index.to_numpy(dtype=np.int32)
            model = PopularityModel(ranked_items=item_pop)
            if log_fn:
                log_fn("baseline.retrieval.fallback algo=popular")
    elif algo in {"bm25", "cosine", "tfidf"}:
        try:
            if require_gpu:
                raise RuntimeError("itemknn retrieval is CPU-only in this pipeline (require_gpu=true)")

            from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender

            knn_k = int(cfg.get("itemknn_k", 200))
            knn_threads = int(cfg.get("itemknn_num_threads", 0))

            if algo == "bm25":
                model = BM25Recommender(
                    K=knn_k,
                    K1=float(cfg.get("bm25_k1", 1.2)),
                    B=float(cfg.get("bm25_b", 0.75)),
                    num_threads=knn_threads,
                )
            elif algo == "cosine":
                model = CosineRecommender(K=knn_k, num_threads=knn_threads)
            else:
                model = TFIDFRecommender(K=knn_k, num_threads=knn_threads)

            t_fit = time.perf_counter()
            model.fit(user_item.tocsr())
            if log_fn:
                log_fn(
                    f"baseline.retrieval.fit algo={algo} K={knn_k} threads={knn_threads} "
                    f"done_in={time.perf_counter() - t_fit:.1f}s"
                )
        except Exception:
            if require_gpu:
                raise
            algo = "popular"
            item_pop = train_df.groupby("movieId").size().sort_values(ascending=False).index.to_numpy(dtype=np.int32)
            model = PopularityModel(ranked_items=item_pop)
            if log_fn:
                log_fn("baseline.retrieval.fallback algo=popular")
    else:
        algo = "popular"
        item_pop = train_df.groupby("movieId").size().sort_values(ascending=False).index.to_numpy(dtype=np.int32)
        model = PopularityModel(ranked_items=item_pop)
        if log_fn:
            log_fn("baseline.retrieval.algo popular")

    return RetrievalArtifacts(
        algorithm=algo,
        model=model,
        user_item=user_item,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        recommend_batch_size=int(cfg.get("recommend_batch_size", 1024)),
        log_every_batches=max(1, int(cfg.get("log_every_batches", 20))),
    )


def generate_candidates(
    artifacts: RetrievalArtifacts,
    user_ids: Iterable[int],
    k: int = 200,
    filter_seen: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    rows: List[dict] = []
    frames: List[pd.DataFrame] = []
    valid_users: List[int] = []
    valid_uidx: List[int] = []
    for user_id in user_ids:
        uidx = artifacts.user_to_idx.get(int(user_id))
        if uidx is None:
            continue
        valid_users.append(int(user_id))
        valid_uidx.append(int(uidx))

    if artifacts.algorithm in {"als", "bpr", "bm25", "cosine", "tfidf"}:
        bs = max(1, int(artifacts.recommend_batch_size))
        n_batches = max(1, (len(valid_users) + bs - 1) // bs)
        log_every = max(1, int(artifacts.log_every_batches))
        if log_fn:
            log_fn(f"baseline.retrieval.candidates start users={len(valid_users):,} batch_size={bs} batches={n_batches}")

        for batch_idx, start in enumerate(range(0, len(valid_users), bs), start=1):
            end = start + bs
            batch_users = np.asarray(valid_users[start:end], dtype=np.int32)
            batch_uidx = np.asarray(valid_uidx[start:end], dtype=np.int32)

            item_idx, scores = artifacts.model.recommend(
                userid=batch_uidx,
                user_items=artifacts.user_item[batch_uidx],
                N=int(k),
                filter_already_liked_items=filter_seen,
                recalculate_user=False,
            )

            if item_idx.ndim == 1:
                item_idx = item_idx.reshape(1, -1)
                scores = scores.reshape(1, -1)

            movie_ids = artifacts.idx_to_item[item_idx]
            user_rep = np.repeat(batch_users, movie_ids.shape[1])
            frame = pd.DataFrame(
                {
                    "userId": user_rep,
                    "movieId": movie_ids.reshape(-1),
                    "retrieval_score": scores.reshape(-1),
                }
            )
            frames.append(frame)
            if log_fn and (batch_idx % log_every == 0 or batch_idx == n_batches):
                done = min(end, len(valid_users))
                pct = 100.0 * done / max(1, len(valid_users))
                log_fn(f"baseline.retrieval.candidates progress users={done:,}/{len(valid_users):,} ({pct:.1f}%)")
    else:
        for user_id, uidx in zip(valid_users, valid_uidx):
            seen = set()
            if filter_seen:
                seen_idx = artifacts.user_item[uidx].indices
                seen = set(artifacts.idx_to_item[seen_idx].tolist())

            added = 0
            for rank, movie_id in enumerate(artifacts.model.ranked_items):
                if movie_id in seen:
                    continue
                rows.append(
                    {
                        "userId": int(user_id),
                        "movieId": int(movie_id),
                        "retrieval_score": float(1.0 / (rank + 1)),
                    }
                )
                added += 1
                if added >= k:
                    break
        if log_fn:
            log_fn(f"baseline.retrieval.candidates popular_done users={len(valid_users):,}")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(rows)
    if out.empty:
        return out
    out["userId"] = out["userId"].astype("int32")
    out["movieId"] = out["movieId"].astype("int32")
    out["retrieval_score"] = out["retrieval_score"].astype("float32")
    return out
