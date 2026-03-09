from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@lru_cache(maxsize=8)
def _load_movie_genre_masks(data_dir: str) -> tuple[dict[int, int], int]:
    movies_path = Path(data_dir) / "movies.csv"
    if not movies_path.exists():
        return {}, 0

    movies = pd.read_csv(movies_path, usecols=["movieId", "genres"], dtype={"movieId": "int32", "genres": "string"})

    genre_set: set[str] = set()
    for g in movies["genres"].fillna(""):
        for token in str(g).split("|"):
            if token and token != "(no genres listed)":
                genre_set.add(token)

    genres = sorted(genre_set)
    if len(genres) > 63:
        genres = genres[:63]

    genre_to_bit = {g: i for i, g in enumerate(genres)}
    mask_map: dict[int, int] = {}

    for movie_id, g in zip(movies["movieId"].to_numpy(), movies["genres"].fillna("")):
        mask = 0
        for token in str(g).split("|"):
            bit = genre_to_bit.get(token)
            if bit is not None:
                mask |= 1 << bit
        mask_map[int(movie_id)] = int(mask)

    return mask_map, len(genres)


def _bitcount_u64(arr: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(arr).astype(np.float32)
    return np.array([int(int(x).bit_count()) for x in arr], dtype=np.float32)


def _build_user_recent_genre_mask(train_df: pd.DataFrame, movie_mask_map: dict[int, int], recent_n: int) -> pd.Series:
    if not movie_mask_map:
        return pd.Series(dtype="uint64", name="user_recent_genre_mask")

    recent = train_df.groupby("userId", sort=False).tail(max(1, int(recent_n)))[["userId", "movieId"]].copy()
    recent["genre_mask"] = recent["movieId"].map(movie_mask_map).fillna(0).astype("uint64")

    user_mask = recent.groupby("userId", sort=False)["genre_mask"].agg(
        lambda s: int(np.bitwise_or.reduce(s.to_numpy(dtype=np.uint64))) if len(s) else 0
    )
    return user_mask.rename("user_recent_genre_mask")


def _prepare_base_context_pandas(train_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "user_cnt": train_df.groupby("userId").size().rename("user_interactions"),
        "item_cnt": train_df.groupby("movieId").size().rename("item_popularity"),
        "user_mean": train_df.groupby("userId")["rating"].mean().rename("user_mean_rating"),
        "item_mean": train_df.groupby("movieId")["rating"].mean().rename("item_mean_rating"),
        "user_last_ts": train_df.groupby("userId")["timestamp"].max().rename("user_last_ts"),
        "user_first_ts": train_df.groupby("userId")["timestamp"].min().rename("user_first_ts"),
        "item_last_ts": train_df.groupby("movieId")["timestamp"].max().rename("item_last_ts"),
    }


def _prepare_base_context_cudf(train_df: pd.DataFrame) -> dict[str, Any]:
    import cudf

    gtrain = cudf.from_pandas(train_df[["userId", "movieId", "rating", "timestamp"]].copy())
    return {
        "user_cnt": gtrain.groupby("userId").size().reset_index(name="user_interactions"),
        "item_cnt": gtrain.groupby("movieId").size().reset_index(name="item_popularity"),
        "user_mean": gtrain.groupby("userId")["rating"].mean().reset_index().rename(columns={"rating": "user_mean_rating"}),
        "item_mean": gtrain.groupby("movieId")["rating"].mean().reset_index().rename(columns={"rating": "item_mean_rating"}),
        "user_last_ts": gtrain.groupby("userId")["timestamp"].max().reset_index().rename(columns={"timestamp": "user_last_ts"}),
        "user_first_ts": gtrain.groupby("userId")["timestamp"].min().reset_index().rename(columns={"timestamp": "user_first_ts"}),
        "item_last_ts": gtrain.groupby("movieId")["timestamp"].max().reset_index().rename(columns={"timestamp": "item_last_ts"}),
    }


def prepare_feature_context(
    train_df: pd.DataFrame,
    data_dir: str | None = None,
    recent_genres_n: int = 20,
    use_gpu: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {}

    if use_gpu:
        try:
            ctx["base_backend"] = "cudf"
            ctx["base_ctx"] = _prepare_base_context_cudf(train_df)
        except Exception as e:
            if log_fn:
                log_fn(f"ranker.features context_gpu_fallback reason={type(e).__name__}: {e}")
            ctx["base_backend"] = "pandas"
            ctx["base_ctx"] = _prepare_base_context_pandas(train_df)
    else:
        ctx["base_backend"] = "pandas"
        ctx["base_ctx"] = _prepare_base_context_pandas(train_df)

    ctx["movie_mask_map"] = {}
    ctx["n_genres"] = 0
    ctx["user_recent_mask"] = pd.Series(dtype="uint64", name="user_recent_genre_mask")

    if data_dir:
        movie_mask_map, n_genres = _load_movie_genre_masks(str(data_dir))
        ctx["movie_mask_map"] = movie_mask_map
        ctx["n_genres"] = n_genres
        if n_genres > 0:
            ctx["user_recent_mask"] = _build_user_recent_genre_mask(train_df, movie_mask_map, recent_n=recent_genres_n)

    if log_fn:
        log_fn(
            "ranker.features context_ready "
            f"backend={ctx['base_backend']} genre_vocab={int(ctx['n_genres'])} "
            f"user_recent_masks={len(ctx['user_recent_mask']):,}"
        )
    return ctx


def _build_base_features_pandas(
    candidates: pd.DataFrame,
    target_df: pd.DataFrame,
    base_ctx: dict[str, pd.Series],
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    user_cnt = base_ctx["user_cnt"]
    item_cnt = base_ctx["item_cnt"]
    user_mean = base_ctx["user_mean"]
    item_mean = base_ctx["item_mean"]
    user_last_ts = base_ctx["user_last_ts"]
    user_first_ts = base_ctx["user_first_ts"]
    item_last_ts = base_ctx["item_last_ts"]

    if "timestamp" in target_df.columns:
        query_ts = target_df.groupby("userId")["timestamp"].max().rename("query_ts")
    else:
        query_ts = user_last_ts.rename("query_ts")

    feats = candidates
    if log_fn:
        log_fn(f"ranker.features merges_start backend=pandas rows={len(feats):,}")

    merge_plan = [
        ("user_cnt", "userId", user_cnt),
        ("item_cnt", "movieId", item_cnt),
        ("user_mean", "userId", user_mean),
        ("item_mean", "movieId", item_mean),
        ("user_last_ts", "userId", user_last_ts),
        ("user_first_ts", "userId", user_first_ts),
        ("item_last_ts", "movieId", item_last_ts),
        ("query_ts", "userId", query_ts),
    ]
    for name, key, frame in merge_plan:
        t0 = time.perf_counter()
        feats = feats.merge(frame, on=key, how="left")
        if log_fn:
            log_fn(
                f"ranker.features merge_done backend=pandas name={name} "
                f"rows={len(feats):,} in={time.perf_counter() - t0:.1f}s"
            )

    feats["age_gap"] = (feats["user_last_ts"] - feats["item_last_ts"]).abs()
    feats["time_since_last_user_event"] = (feats["query_ts"] - feats["user_last_ts"]).clip(lower=0)
    feats["time_since_last_item_event"] = (feats["query_ts"] - feats["item_last_ts"]).clip(lower=0)

    day_sec = 24.0 * 3600.0
    week_sec = 7.0 * day_sec
    feats["item_freshness_1d"] = np.exp(-feats["time_since_last_item_event"] / day_sec)
    feats["item_freshness_7d"] = np.exp(-feats["time_since_last_item_event"] / week_sec)

    user_lifespan = (feats["user_last_ts"] - feats["user_first_ts"]).clip(lower=0) + 1.0
    feats["user_activity_rate"] = feats["user_interactions"] / user_lifespan

    query_dt = pd.to_datetime(feats["query_ts"], unit="s", errors="coerce")
    query_hour = query_dt.dt.hour.fillna(0).astype("float32")
    query_dow = query_dt.dt.dayofweek.fillna(0).astype("float32")
    feats["query_hour_sin"] = np.sin(2.0 * math.pi * query_hour / 24.0)
    feats["query_hour_cos"] = np.cos(2.0 * math.pi * query_hour / 24.0)
    feats["query_dow_sin"] = np.sin(2.0 * math.pi * query_dow / 7.0)
    feats["query_dow_cos"] = np.cos(2.0 * math.pi * query_dow / 7.0)

    feats["user_item_mean_gap"] = feats["user_mean_rating"] - feats["item_mean_rating"]
    feats["user_item_mean_abs_gap"] = feats["user_item_mean_gap"].abs()
    feats["item_pop_per_user"] = feats["item_popularity"] / (feats["user_interactions"] + 1.0)
    if log_fn:
        log_fn("ranker.features base_signals_done")
    return feats


def _build_base_features_cudf(
    candidates: pd.DataFrame,
    target_df: pd.DataFrame,
    base_ctx: dict[str, Any],
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    import cudf
    import cupy as cp

    if log_fn:
        log_fn("ranker.features backend=cudf")

    gcands = cudf.from_pandas(candidates.copy())
    user_cnt = base_ctx["user_cnt"]
    item_cnt = base_ctx["item_cnt"]
    user_mean = base_ctx["user_mean"]
    item_mean = base_ctx["item_mean"]
    user_last_ts = base_ctx["user_last_ts"]
    user_first_ts = base_ctx["user_first_ts"]
    item_last_ts = base_ctx["item_last_ts"]

    if "timestamp" in target_df.columns:
        gtarget = cudf.from_pandas(target_df[["userId", "timestamp"]].copy())
        query_ts = gtarget.groupby("userId")["timestamp"].max().reset_index().rename(columns={"timestamp": "query_ts"})
    else:
        query_ts = user_last_ts.rename(columns={"user_last_ts": "query_ts"})

    feats = gcands
    if log_fn:
        log_fn(f"ranker.features merges_start backend=cudf rows={len(feats):,}")
    merge_plan = [
        ("user_cnt", "userId", user_cnt),
        ("item_cnt", "movieId", item_cnt),
        ("user_mean", "userId", user_mean),
        ("item_mean", "movieId", item_mean),
        ("user_last_ts", "userId", user_last_ts),
        ("user_first_ts", "userId", user_first_ts),
        ("item_last_ts", "movieId", item_last_ts),
        ("query_ts", "userId", query_ts),
    ]
    for name, key, frame in merge_plan:
        t0 = time.perf_counter()
        feats = feats.merge(frame, on=key, how="left")
        if log_fn:
            log_fn(
                f"ranker.features merge_done backend=cudf name={name} "
                f"rows={len(feats):,} in={time.perf_counter() - t0:.1f}s"
            )

    feats["age_gap"] = (feats["user_last_ts"] - feats["item_last_ts"]).abs()
    feats["time_since_last_user_event"] = (feats["query_ts"] - feats["user_last_ts"]).clip(lower=0)
    feats["time_since_last_item_event"] = (feats["query_ts"] - feats["item_last_ts"]).clip(lower=0)

    day_sec = np.float32(24.0 * 3600.0)
    week_sec = np.float32(7.0 * 24.0 * 3600.0)
    ts_item = feats["time_since_last_item_event"].astype("float32").to_cupy()
    feats["item_freshness_1d"] = cudf.Series(cp.exp(-ts_item / day_sec), index=feats.index)
    feats["item_freshness_7d"] = cudf.Series(cp.exp(-ts_item / week_sec), index=feats.index)

    user_lifespan = (feats["user_last_ts"] - feats["user_first_ts"]).clip(lower=0).astype("float32") + 1.0
    feats["user_activity_rate"] = feats["user_interactions"] / user_lifespan

    query_ts_safe = feats["query_ts"].fillna(0).astype("int64")
    query_dt = cudf.to_datetime(query_ts_safe, unit="s")
    query_hour = query_dt.dt.hour.fillna(0).astype("float32")
    query_dow = query_dt.dt.weekday.fillna(0).astype("float32")
    qh = query_hour.to_cupy()
    qd = query_dow.to_cupy()
    feats["query_hour_sin"] = cudf.Series(cp.sin(2.0 * math.pi * qh / 24.0), index=feats.index)
    feats["query_hour_cos"] = cudf.Series(cp.cos(2.0 * math.pi * qh / 24.0), index=feats.index)
    feats["query_dow_sin"] = cudf.Series(cp.sin(2.0 * math.pi * qd / 7.0), index=feats.index)
    feats["query_dow_cos"] = cudf.Series(cp.cos(2.0 * math.pi * qd / 7.0), index=feats.index)

    feats["user_item_mean_gap"] = feats["user_mean_rating"] - feats["item_mean_rating"]
    feats["user_item_mean_abs_gap"] = feats["user_item_mean_gap"].abs()
    feats["item_pop_per_user"] = feats["item_popularity"] / (feats["user_interactions"] + 1.0)

    t0 = time.perf_counter()
    if log_fn:
        log_fn(f"ranker.features to_pandas_start rows={len(feats):,}")
    out = feats.to_pandas()
    if log_fn:
        log_fn(f"ranker.features to_pandas_done rows={len(out):,} in={time.perf_counter() - t0:.1f}s")
    if log_fn:
        log_fn("ranker.features base_signals_done")
    return out


def build_candidate_features(
    candidates: pd.DataFrame,
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    data_dir: str | None = None,
    recent_genres_n: int = 20,
    log_fn: Callable[[str], None] | None = None,
    use_gpu: bool = False,
    feature_context: dict[str, Any] | None = None,
    chunk_size_users: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if candidates.empty:
        return candidates.copy(), []

    chunk_n = int(chunk_size_users or 0)
    if chunk_n > 0:
        unique_users = np.sort(candidates["userId"].unique())
        if len(unique_users) > chunk_n:
            total_chunks = (len(unique_users) + chunk_n - 1) // chunk_n
            parts: list[pd.DataFrame] = []
            feature_cols_ref: list[str] | None = None
            for idx, start in enumerate(range(0, len(unique_users), chunk_n), start=1):
                user_chunk = unique_users[start : start + chunk_n]
                cand_chunk = candidates[candidates["userId"].isin(user_chunk)].copy()
                target_chunk = target_df[target_df["userId"].isin(user_chunk)].copy()
                if log_fn:
                    log_fn(
                        "ranker.features chunk_start "
                        f"{idx}/{total_chunks} users={len(user_chunk):,} candidate_rows={len(cand_chunk):,}"
                    )
                chunk_feats, chunk_cols = build_candidate_features(
                    candidates=cand_chunk,
                    train_df=train_df,
                    target_df=target_chunk,
                    data_dir=data_dir,
                    recent_genres_n=recent_genres_n,
                    log_fn=log_fn,
                    use_gpu=use_gpu,
                    feature_context=feature_context,
                    chunk_size_users=None,
                )
                parts.append(chunk_feats)
                if feature_cols_ref is None:
                    feature_cols_ref = chunk_cols
                if log_fn:
                    log_fn(
                        "ranker.features chunk_done "
                        f"{idx}/{total_chunks} rows={len(chunk_feats):,}"
                    )
            feats_all = pd.concat(parts, ignore_index=True)
            if log_fn:
                log_fn(f"ranker.features chunk_all_done rows={len(feats_all):,}")
            return feats_all, (feature_cols_ref or [])

    if log_fn:
        log_fn(
            "ranker.features start "
            f"candidate_rows={len(candidates):,} train_rows={len(train_df):,} target_rows={len(target_df):,}"
        )

    if feature_context is None:
        feature_context = prepare_feature_context(
            train_df,
            data_dir=data_dir,
            recent_genres_n=recent_genres_n,
            use_gpu=use_gpu,
            log_fn=log_fn,
        )

    base_backend = str(feature_context.get("base_backend", "pandas"))
    base_ctx = feature_context.get("base_ctx", {})

    if base_backend == "cudf":
        try:
            feats = _build_base_features_cudf(candidates, target_df, base_ctx, log_fn=log_fn)
        except Exception as e:
            if log_fn:
                log_fn(f"ranker.features gpu_fallback reason={type(e).__name__}: {e}")
            if not base_ctx:
                base_ctx = _prepare_base_context_pandas(train_df)
            feats = _build_base_features_pandas(candidates, target_df, base_ctx, log_fn=log_fn)
    else:
        if not base_ctx:
            base_ctx = _prepare_base_context_pandas(train_df)
        feats = _build_base_features_pandas(candidates, target_df, base_ctx, log_fn=log_fn)

    movie_mask_map = feature_context.get("movie_mask_map") or {}
    n_genres = int(feature_context.get("n_genres", 0))
    if n_genres > 0:
        user_recent_mask = feature_context.get("user_recent_mask")
        if isinstance(user_recent_mask, pd.Series) and not user_recent_mask.empty:
            feats = feats.merge(user_recent_mask, on="userId", how="left")
            user_mask_arr = feats["user_recent_genre_mask"].fillna(0).to_numpy(dtype=np.uint64)
        else:
            user_mask_arr = np.zeros(len(feats), dtype=np.uint64)

        item_mask_arr = feats["movieId"].map(movie_mask_map).fillna(0).to_numpy(dtype=np.uint64)
        overlap_mask = np.bitwise_and(user_mask_arr, item_mask_arr)

        overlap_bits = _bitcount_u64(overlap_mask)
        item_bits = np.maximum(_bitcount_u64(item_mask_arr), 1.0)
        user_bits = np.maximum(_bitcount_u64(user_mask_arr), 1.0)

        feats["user_recent_genre_overlap"] = overlap_bits / item_bits
        feats["user_recent_genre_coverage"] = overlap_bits / user_bits
        feats["item_genre_count"] = item_bits
        feats = feats.drop(columns=["user_recent_genre_mask"], errors="ignore")
        if log_fn:
            log_fn("ranker.features genre_signals_done")

    labels = target_df[["userId", "movieId"]].drop_duplicates().copy()
    if log_fn:
        log_fn(f"ranker.features labels_ready rows={len(labels):,}")
    labels["label"] = 1
    t0 = time.perf_counter()
    feats = feats.merge(labels, on=["userId", "movieId"], how="left")
    if log_fn:
        log_fn(f"ranker.features labels_merged rows={len(feats):,} in={time.perf_counter() - t0:.1f}s")
    feats["label"] = feats["label"].fillna(0).astype("int8")

    feature_cols = [
        "retrieval_score",
        "user_interactions",
        "item_popularity",
        "user_mean_rating",
        "item_mean_rating",
        "user_item_mean_gap",
        "user_item_mean_abs_gap",
        "item_pop_per_user",
        "age_gap",
        "time_since_last_user_event",
        "time_since_last_item_event",
        "item_freshness_1d",
        "item_freshness_7d",
        "user_activity_rate",
        "query_hour_sin",
        "query_hour_cos",
        "query_dow_sin",
        "query_dow_cos",
    ]

    if "user_recent_genre_overlap" in feats.columns:
        feature_cols.append("user_recent_genre_overlap")
    if "user_recent_genre_coverage" in feats.columns:
        feature_cols.append("user_recent_genre_coverage")
    if "item_genre_count" in feats.columns:
        feature_cols.append("item_genre_count")
    if "source_votes" in feats.columns:
        feature_cols.append("source_votes")
    score_cols = sorted(
        c
        for c in feats.columns
        if c.startswith("score_") and c not in {"score_col", "score", "score_label"}
    )
    for c in score_cols:
        if c not in feature_cols:
            feature_cols.append(c)

    feats[feature_cols] = feats[feature_cols].fillna(0.0)
    feats[feature_cols] = feats[feature_cols].astype("float32")
    if log_fn:
        pos_rate = float(feats["label"].mean()) if len(feats) else 0.0
        log_fn(
            "ranker.features done "
            f"rows={len(feats):,} feature_cols={len(feature_cols)} positive_rate={pos_rate:.4f}"
        )
    return feats, feature_cols
