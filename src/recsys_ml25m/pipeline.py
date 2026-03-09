from __future__ import annotations

import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import ensure_dirs
from .data.io import load_ratings, temporal_leave_last_split
from .eval.offline import evaluate_predictions
from .ranking.features import build_candidate_features, prepare_feature_context
from .ranking.ranker import score_ranker, train_ranker
from .retrieval.als import build_retrieval_model, generate_candidates
from .retrieval.two_tower import generate_candidates_faiss, train_two_tower_faiss
from .utils.monitoring import ResourceMonitor


def _log(msg: str, monitor: ResourceMonitor | None = None, include_resources: bool = False) -> None:
    suffix = ""
    if include_resources and monitor is not None:
        suffix = f" | {monitor.summary()}"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}{suffix}", flush=True)


def _rank_norm_scores(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    ranked = df.sort_values(["userId", score_col], ascending=[True, False], kind="mergesort").copy()
    ranked["rank_pos"] = ranked.groupby("userId").cumcount() + 1
    ranked["rank_score"] = 1.0 / ranked["rank_pos"].astype("float32")
    out = ranked[["userId", "movieId", "rank_score"]].copy()
    out = out.astype({"userId": "int32", "movieId": "int32", "rank_score": "float32"})
    return out


def _fuse_candidates_multi(
    sources: list[tuple[str, pd.DataFrame, float]],
    k: int,
    log_fn: Callable[[str], None] | None = None,
    stage_label: str = "candidate_fusion",
    use_gpu: bool = False,
    chunk_users: int | None = None,
) -> pd.DataFrame:
    def _progress(msg: str) -> None:
        if log_fn is not None:
            log_fn(f"{stage_label} {msg}")

    chunk_n = int(chunk_users or 0)
    if chunk_n > 0:
        user_sets: list[np.ndarray] = []
        for _, source_df, weight in sources:
            if source_df is None or source_df.empty or float(weight) <= 0.0:
                continue
            user_sets.append(source_df["userId"].to_numpy(dtype=np.int32, copy=False))

        if user_sets:
            unique_users = np.unique(np.concatenate(user_sets))
            if len(unique_users) > chunk_n:
                total_chunks = (len(unique_users) + chunk_n - 1) // chunk_n
                parts: list[pd.DataFrame] = []
                for idx, start in enumerate(range(0, len(unique_users), chunk_n), start=1):
                    users_chunk = unique_users[start : start + chunk_n]
                    _progress(f"chunk_start {idx}/{total_chunks} users={len(users_chunk):,}")
                    chunk_sources: list[tuple[str, pd.DataFrame, float]] = []
                    for name, source_df, weight in sources:
                        if source_df is None or source_df.empty or float(weight) <= 0.0:
                            continue
                        part = source_df[source_df["userId"].isin(users_chunk)].copy()
                        chunk_sources.append((name, part, weight))

                    chunk_out = _fuse_candidates_multi(
                        sources=chunk_sources,
                        k=k,
                        log_fn=log_fn,
                        stage_label=f"{stage_label}.chunk{idx}",
                        use_gpu=use_gpu,
                        chunk_users=None,
                    )
                    parts.append(chunk_out)
                    _progress(f"chunk_done {idx}/{total_chunks} rows={len(chunk_out):,}")

                out = pd.concat(parts, ignore_index=True)
                _progress(f"chunk_all_done rows={len(out):,}")
                return out

    if use_gpu:
        try:
            import cudf

            _progress("backend=cudf")
            prepared_gpu: list[cudf.DataFrame] = []
            score_cols_gpu: list[str] = []
            weights_gpu: dict[str, float] = {}

            for name, source_df, weight in sources:
                if source_df is None or source_df.empty:
                    continue
                w = float(weight)
                if w <= 0.0:
                    continue
                _progress(f"source_prepare name={name} rows={len(source_df):,} weight={w:.2f}")
                t0 = time.perf_counter()
                col = f"score_{name}"
                gsrc = cudf.from_pandas(source_df[["userId", "movieId", "retrieval_score"]])
                gsrc = gsrc.sort_values(["userId", "retrieval_score"], ascending=[True, False])
                gsrc["rank_pos"] = gsrc.groupby("userId").cumcount() + 1
                part = gsrc[["userId", "movieId"]].copy()
                part[col] = (1.0 / gsrc["rank_pos"].astype("float32")).astype("float32")
                _progress(f"source_ready name={name} rows={len(part):,} in={time.perf_counter() - t0:.1f}s")
                prepared_gpu.append(part)
                score_cols_gpu.append(col)
                weights_gpu[col] = w

            if not prepared_gpu:
                raise RuntimeError("candidate_fusion produced no valid input sources")

            _progress(f"merge_start inputs={len(prepared_gpu)}")
            merged_gpu = prepared_gpu[0]
            for idx, part in enumerate(prepared_gpu[1:], start=1):
                t0 = time.perf_counter()
                merged_gpu = merged_gpu.merge(part, on=["userId", "movieId"], how="outer")
                _progress(
                    f"merge_step={idx}/{len(prepared_gpu) - 1} rows={len(merged_gpu):,} in={time.perf_counter() - t0:.1f}s"
                )

            total_w = sum(weights_gpu.values())
            if total_w <= 0.0:
                uniform_w = 1.0 / float(len(score_cols_gpu))
                norm_weights = {col: uniform_w for col in score_cols_gpu}
            else:
                norm_weights = {col: float(w / total_w) for col, w in weights_gpu.items()}

            for col in score_cols_gpu:
                merged_gpu[col] = merged_gpu[col].fillna(0.0)

            source_votes = (merged_gpu[score_cols_gpu[0]] > 0.0).astype("int8")
            for col in score_cols_gpu[1:]:
                source_votes = (source_votes + (merged_gpu[col] > 0.0).astype("int8")).astype("int8")
            merged_gpu["source_votes"] = source_votes

            retrieval_score = merged_gpu[score_cols_gpu[0]].astype("float32") * float(norm_weights[score_cols_gpu[0]])
            for col in score_cols_gpu[1:]:
                retrieval_score = retrieval_score + merged_gpu[col].astype("float32") * float(norm_weights[col])
            merged_gpu["retrieval_score"] = retrieval_score.astype("float32")

            merged_gpu = merged_gpu.sort_values(["userId", "retrieval_score"], ascending=[True, False])
            merged_gpu["rownum"] = merged_gpu.groupby("userId").cumcount()
            merged_gpu = merged_gpu[merged_gpu["rownum"] < int(k)]

            keep_cols = ["userId", "movieId", "retrieval_score", "source_votes"] + score_cols_gpu
            out_gpu = merged_gpu[keep_cols].copy()
            cast_map = {"userId": "int32", "movieId": "int32", "retrieval_score": "float32", "source_votes": "int8"}
            cast_map.update({c: "float32" for c in score_cols_gpu})
            out_gpu = out_gpu.astype(cast_map)
            _progress(f"topk_done rows={len(out_gpu):,} k={int(k)}")
            return out_gpu.to_pandas()
        except Exception as e:
            _progress(f"gpu_fallback reason={type(e).__name__}: {e}")

    prepared: list[pd.DataFrame] = []
    score_cols: list[str] = []
    weights: dict[str, float] = {}

    for name, source_df, weight in sources:
        if source_df is None or source_df.empty:
            continue
        w = float(weight)
        if w <= 0.0:
            continue
        _progress(f"source_prepare name={name} rows={len(source_df):,} weight={w:.2f}")
        t0 = time.perf_counter()
        col = f"score_{name}"
        part = _rank_norm_scores(source_df, "retrieval_score").rename(columns={"rank_score": col})
        _progress(f"source_ready name={name} rows={len(part):,} in={time.perf_counter() - t0:.1f}s")
        prepared.append(part)
        score_cols.append(col)
        weights[col] = w

    if not prepared:
        raise RuntimeError("candidate_fusion produced no valid input sources")

    _progress(f"merge_start inputs={len(prepared)}")
    merged = prepared[0]
    for idx, part in enumerate(prepared[1:], start=1):
        t0 = time.perf_counter()
        merged = merged.merge(part, on=["userId", "movieId"], how="outer")
        _progress(
            f"merge_step={idx}/{len(prepared) - 1} rows={len(merged):,} in={time.perf_counter() - t0:.1f}s"
        )

    total_w = sum(weights.values())
    if total_w <= 0.0:
        uniform_w = 1.0 / float(len(score_cols))
        norm_weights = {col: uniform_w for col in score_cols}
    else:
        norm_weights = {col: float(w / total_w) for col, w in weights.items()}

    for col in score_cols:
        merged[col] = merged[col].fillna(0.0)

    merged["source_votes"] = sum((merged[c] > 0.0).astype("int8") for c in score_cols).astype("int8")
    merged["retrieval_score"] = sum(float(norm_weights[c]) * merged[c] for c in score_cols)

    merged = merged.sort_values(["userId", "retrieval_score"], ascending=[True, False], kind="mergesort")
    merged["rownum"] = merged.groupby("userId").cumcount()
    merged = merged[merged["rownum"] < int(k)].copy()

    keep_cols = ["userId", "movieId", "retrieval_score", "source_votes"] + score_cols
    out = merged[keep_cols].copy()

    cast_map = {"userId": "int32", "movieId": "int32", "retrieval_score": "float32", "source_votes": "int8"}
    cast_map.update({c: "float32" for c in score_cols})
    out = out.astype(cast_map)
    _progress(f"topk_done rows={len(out):,} k={int(k)}")
    return out


def _fuse_candidates(
    two_tower_df: pd.DataFrame,
    als_df: pd.DataFrame,
    k: int,
    two_tower_weight: float = 0.7,
    als_weight: float = 0.3,
    use_gpu: bool = False,
) -> pd.DataFrame:
    return _fuse_candidates_multi(
        sources=[
            ("tt", two_tower_df, two_tower_weight),
            ("als", als_df, als_weight),
        ],
        k=k,
        use_gpu=use_gpu,
    )


def _fuse_ranker_scores(
    base_scores: pd.DataFrame,
    final_scores: pd.DataFrame,
    w_base: float = 0.5,
    w_final: float = 0.5,
    log_fn: Callable[[str], None] | None = None,
    stage_label: str = "ranker_ensemble",
) -> pd.DataFrame:
    def _progress(msg: str) -> None:
        if log_fn is not None:
            log_fn(f"{stage_label} {msg}")

    _progress(f"start base_rows={len(base_scores):,} final_rows={len(final_scores):,} w_base={w_base:.2f} w_final={w_final:.2f}")
    b = _rank_norm_scores(base_scores, "rank_score").rename(columns={"rank_score": "score_base"})
    f = _rank_norm_scores(final_scores, "rank_score").rename(columns={"rank_score": "score_final"})
    _progress(f"normalized base_rows={len(b):,} final_rows={len(f):,}")
    merged = b.merge(f, on=["userId", "movieId"], how="outer")
    _progress(f"merged rows={len(merged):,}")
    merged["score_base"] = merged["score_base"].fillna(0.0)
    merged["score_final"] = merged["score_final"].fillna(0.0)
    merged["rank_score"] = w_base * merged["score_base"] + w_final * merged["score_final"]
    out = merged[["userId", "movieId", "rank_score"]].copy()
    out = out.astype({"userId": "int32", "movieId": "int32", "rank_score": "float32"})
    _progress(f"done rows={len(out):,}")
    return out


def _save_plots(metrics: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    keep = metrics[metrics["k"] == metrics["k"].max()].copy()
    if keep.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    x = range(len(keep))
    ax.bar([i - 0.2 for i in x], keep["recall"], width=0.2, label="Recall")
    ax.bar(x, keep["ndcg"], width=0.2, label="NDCG")
    ax.bar([i + 0.2 for i in x], keep["mrr"], width=0.2, label="MRR")

    ax.set_xticks(list(x))
    ax.set_xticklabels(keep["model"], rotation=20, ha="right")
    ax.set_title(f"Offline metrics @K={int(keep['k'].iloc[0])}")
    ax.set_ylim(0, max(0.01, float(keep[["recall", "ndcg", "mrr"]].max().max()) * 1.2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_comparison_kmax.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    for model, g in metrics.groupby("model"):
        g2 = g.sort_values("k")
        ax.plot(g2["k"], g2["recall"], marker="o", label=model)
    ax.set_title("Recall@K by model")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "recall_at_k_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    for model, g in metrics.groupby("model"):
        g2 = g.sort_values("k")
        ax.plot(g2["k"], g2["ndcg"], marker="o", label=model)
    ax.set_title("NDCG@K by model")
    ax.set_xlabel("K")
    ax.set_ylabel("NDCG@K")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ndcg_at_k_curves.png", dpi=150)
    plt.close(fig)


def run_pipeline(cfg: dict) -> dict[str, pd.DataFrame]:
    t_all = time.perf_counter()
    log_cfg = cfg.get("logging", {})
    stage_resources = bool(log_cfg.get("resource_snapshot", True))
    progress_resources = bool(log_cfg.get("progress_resources", False))
    monitor = ResourceMonitor(enabled=stage_resources or progress_resources, gpu_index=int(log_cfg.get("gpu_index", 0)))

    def log(msg: str, with_resources: bool = False) -> None:
        _log(msg, monitor=monitor, include_resources=with_resources)

    def progress(msg: str) -> None:
        _log(msg, monitor=monitor, include_resources=progress_resources)

    log("Pipeline start", with_resources=True)

    data_cfg = cfg.get("data", {})
    out_cfg = cfg.get("outputs", {})
    retrieval_cfg = cfg.get("retrieval", {})
    ranker_cfg = cfg.get("ranker", {})
    final_cfg = cfg.get("final_model", {})
    eval_cfg = cfg.get("evaluation", {})

    out_tables = Path(out_cfg.get("tables_dir", "outputs/tables"))
    out_figures = Path(out_cfg.get("figures_dir", "outputs/figures"))
    ensure_dirs(out_tables, out_figures)
    save_ranker_scores = bool(out_cfg.get("save_ranker_scores", False))
    ranker_scores_dir = Path(out_cfg.get("ranker_scores_dir", out_tables / "ranker_scores"))
    if save_ranker_scores:
        ensure_dirs(ranker_scores_dir)
        log(f"Outputs: ranker_scores={ranker_scores_dir}")
    log(f"Outputs: tables={out_tables} figures={out_figures}")

    def _save_rank_scores(df: pd.DataFrame, name: str) -> None:
        if not save_ranker_scores:
            return
        t0 = time.perf_counter()
        out_path = ranker_scores_dir / f"{name}.pkl"
        payload = df[["userId", "movieId", "rank_score"]].copy()
        payload["userId"] = payload["userId"].astype("int32")
        payload["movieId"] = payload["movieId"].astype("int32")
        payload["rank_score"] = payload["rank_score"].astype("float32")
        payload.to_pickle(out_path)
        log(
            "Stage ranker.scores.save done in "
            f"{time.perf_counter() - t0:.1f}s | name={name} rows={len(payload):,} path={out_path}"
        )

    t0 = time.perf_counter()
    ratings = load_ratings(
        data_dir=data_cfg["dataset_dir"],
        min_rating=float(data_cfg.get("min_rating", 0.0)),
        max_rows=data_cfg.get("max_rows"),
    )
    log(
        "Stage data.load done in "
        f"{time.perf_counter() - t0:.1f}s | rows={len(ratings):,} "
        f"users={ratings['userId'].nunique():,} items={ratings['movieId'].nunique():,}"
        ,
        with_resources=stage_resources,
    )

    t0 = time.perf_counter()
    train_df, val_df, test_df = temporal_leave_last_split(
        ratings,
        val_k=int(data_cfg.get("val_k", 1)),
        test_k=int(data_cfg.get("test_k", 1)),
        min_user_interactions=int(data_cfg.get("min_user_interactions", 5)),
        split_offset=int(data_cfg.get("split_offset", 0)),
    )
    log(
        "Stage data.split done in "
        f"{time.perf_counter() - t0:.1f}s | "
        f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} "
        f"users_train={train_df['userId'].nunique():,} items_train={train_df['movieId'].nunique():,}"
        ,
        with_resources=stage_resources,
    )

    users_val = sorted(set(val_df["userId"].tolist()) & set(train_df["userId"].tolist()))
    users_test = sorted(set(test_df["userId"].tolist()) & set(train_df["userId"].tolist()))
    same_eval_users = users_val == users_test
    log(f"Eval users: val={len(users_val):,} test={len(users_test):,}")
    if save_ranker_scores:
        gt_path = ranker_scores_dir / "test_ground_truth.pkl"
        test_df[["userId", "movieId"]].copy().to_pickle(gt_path)
        log(f"Saved: {gt_path}")

    k_candidates = int(eval_cfg.get("k_candidates", 200))
    ks = [int(k) for k in eval_cfg.get("ks", [10, 20, 50])]
    log(f"Eval setup: k_candidates={k_candidates} ks={ks}")

    run_final = bool(final_cfg.get("enabled", True))
    baseline_enabled = bool(cfg.get("baseline", {}).get("enabled", True))
    fusion_cfg = final_cfg.get("candidate_fusion", {})
    ens_cfg = final_cfg.get("ranker_ensemble", {})
    use_fusion = run_final and bool(fusion_cfg.get("enabled", True))
    use_ens = run_final and bool(ens_cfg.get("enabled", True))

    if (use_fusion or use_ens) and not baseline_enabled:
        raise ValueError("baseline.enabled=false is incompatible with candidate_fusion/ranker_ensemble enabled")

    baseline_metrics = pd.DataFrame()
    all_metrics: list[pd.DataFrame] = []
    scored_bas: pd.DataFrame | None = None
    cand_val_bas: pd.DataFrame | None = None
    cand_test_bas: pd.DataFrame | None = None
    feature_context_cache: dict[tuple[bool, int], dict] = {}

    def _get_feature_context(use_gpu_flag: bool, recent_genres_n: int) -> dict:
        key = (bool(use_gpu_flag), int(recent_genres_n))
        if key not in feature_context_cache:
            feature_context_cache[key] = prepare_feature_context(
                train_df=train_df,
                data_dir=data_cfg.get("dataset_dir"),
                recent_genres_n=int(recent_genres_n),
                use_gpu=bool(use_gpu_flag),
                log_fn=progress,
            )
        return feature_context_cache[key]

    if baseline_enabled:
        t0 = time.perf_counter()
        bas_ret = build_retrieval_model(train_df, retrieval_cfg, log_fn=progress)
        log(
            f"Stage baseline.retrieval.train done in {time.perf_counter() - t0:.1f}s | algo={bas_ret.algorithm}",
            with_resources=stage_resources,
        )

        t0 = time.perf_counter()
        cand_val_bas = generate_candidates(bas_ret, users_val, k=k_candidates, filter_seen=True, log_fn=progress)
        if same_eval_users:
            cand_test_bas = cand_val_bas.copy(deep=False)
            progress("baseline.retrieval.candidates reuse test_from_val=true")
        else:
            cand_test_bas = generate_candidates(bas_ret, users_test, k=k_candidates, filter_seen=True, log_fn=progress)
        log(
            "Stage baseline.retrieval.candidates done in "
            f"{time.perf_counter() - t0:.1f}s | val_rows={len(cand_val_bas):,} test_rows={len(cand_test_bas):,} "
            f"reused_from_val={same_eval_users}"
            ,
            with_resources=stage_resources,
        )

        t0 = time.perf_counter()
        metrics_retrieval_bas = evaluate_predictions(
            ground_truth_df=test_df,
            prediction_df=cand_test_bas,
            score_col="retrieval_score",
            ks=ks,
            model_name=f"baseline_retrieval_{bas_ret.algorithm}",
        )
        log(f"Stage baseline.retrieval.eval done in {time.perf_counter() - t0:.1f}s")

        t0 = time.perf_counter()
        baseline_features_use_gpu = bool(ranker_cfg.get("features_use_gpu", True))
        baseline_recent_n = int(ranker_cfg.get("recent_genres_n", 20))
        baseline_chunk_users = int(ranker_cfg.get("chunk_users", 0)) or None
        baseline_feature_ctx = _get_feature_context(
            use_gpu_flag=baseline_features_use_gpu,
            recent_genres_n=baseline_recent_n,
        )
        train_feats_bas, feat_cols_bas = build_candidate_features(
            cand_val_bas,
            train_df,
            val_df,
            data_dir=data_cfg.get("dataset_dir"),
            recent_genres_n=baseline_recent_n,
            log_fn=progress,
            use_gpu=baseline_features_use_gpu,
            feature_context=baseline_feature_ctx,
            chunk_size_users=baseline_chunk_users,
        )
        test_feats_bas, _ = build_candidate_features(
            cand_test_bas,
            train_df,
            test_df,
            data_dir=data_cfg.get("dataset_dir"),
            recent_genres_n=baseline_recent_n,
            log_fn=progress,
            use_gpu=baseline_features_use_gpu,
            feature_context=baseline_feature_ctx,
            chunk_size_users=baseline_chunk_users,
        )
        log(
            "Stage baseline.ranker.features done in "
            f"{time.perf_counter() - t0:.1f}s | train_rows={len(train_feats_bas):,} test_rows={len(test_feats_bas):,}"
            ,
            with_resources=stage_resources,
        )

        t0 = time.perf_counter()
        ranker_bas = train_ranker(train_feats_bas, feat_cols_bas, ranker_cfg, log_fn=progress)
        scored_bas = score_ranker(ranker_bas, test_feats_bas, feat_cols_bas, log_fn=progress)
        log(
            f"Stage baseline.ranker.train+score done in {time.perf_counter() - t0:.1f}s | "
            f"scored_rows={len(scored_bas):,}"
            ,
            with_resources=stage_resources,
        )
        _save_rank_scores(scored_bas, "baseline_scores")

        del train_feats_bas, test_feats_bas
        gc.collect()

        t0 = time.perf_counter()
        metrics_ranker_bas = evaluate_predictions(
            ground_truth_df=test_df,
            prediction_df=scored_bas,
            score_col="rank_score",
            ks=ks,
            model_name="baseline_ranker",
        )
        log(f"Stage baseline.ranker.eval done in {time.perf_counter() - t0:.1f}s")

        baseline_metrics = pd.concat([metrics_retrieval_bas, metrics_ranker_bas], ignore_index=True)
        baseline_metrics.to_csv(out_tables / "metrics_baseline.csv", index=False)
        log(f"Saved: {out_tables / 'metrics_baseline.csv'}")
        all_metrics.append(baseline_metrics)
    else:
        log("Baseline stages are disabled (baseline.enabled=false)")

    if baseline_enabled and (not use_fusion):
        del cand_val_bas, cand_test_bas
        cand_val_bas = None
        cand_test_bas = None
        gc.collect()

    if run_final:
        t0 = time.perf_counter()
        tt_art = train_two_tower_faiss(train_df, final_cfg.get("two_tower", {}), log_fn=progress)
        log(f"Stage final.twotower.train done in {time.perf_counter() - t0:.1f}s", with_resources=stage_resources)

        t0 = time.perf_counter()
        cand_val_final = generate_candidates_faiss(
            tt_art,
            train_df=train_df,
            user_ids=users_val,
            k=k_candidates,
            filter_seen=True,
            log_fn=progress,
        )
        if same_eval_users:
            cand_test_final = cand_val_final.copy(deep=False)
            progress("final.twotower.candidates reuse test_from_val=true")
        else:
            cand_test_final = generate_candidates_faiss(
                tt_art,
                train_df=train_df,
                user_ids=users_test,
                k=k_candidates,
                filter_seen=True,
                log_fn=progress,
            )
        log(
            "Stage final.twotower.candidates done in "
            f"{time.perf_counter() - t0:.1f}s | val_rows={len(cand_val_final):,} test_rows={len(cand_test_final):,} "
            f"reused_from_val={same_eval_users}"
            ,
            with_resources=stage_resources,
        )

        if use_fusion:
            aux_cfg_list = fusion_cfg.get("aux_retrievals")
            if isinstance(aux_cfg_list, list):
                raw_aux_cfgs = [x for x in aux_cfg_list if isinstance(x, dict)]
            else:
                # Backward compatibility: support single aux_retrieval dict.
                single_aux = fusion_cfg.get("aux_retrieval", {})
                raw_aux_cfgs = [single_aux] if isinstance(single_aux, dict) else []

            aux_sources: list[tuple[str, pd.DataFrame, pd.DataFrame, float, str]] = []
            for aux_idx, aux_cfg in enumerate(raw_aux_cfgs, start=1):
                use_aux = bool(aux_cfg.get("enabled", False))
                aux_weight = float(aux_cfg.get("weight", 0.0))
                aux_algo = str(aux_cfg.get("algorithm", "none")).lower()
                if (not use_aux) or aux_weight <= 0.0:
                    continue

                aux_retr_cfg = dict(retrieval_cfg)
                aux_retr_cfg.update(
                    {
                        k: v
                        for k, v in aux_cfg.items()
                        if k
                        not in {
                            "enabled",
                            "weight",
                            "algorithm",
                        }
                    }
                )
                aux_retr_cfg["algorithm"] = aux_algo
                aux_retr_cfg["require_gpu"] = bool(aux_cfg.get("require_gpu", False))
                if aux_algo in {"bm25", "cosine", "tfidf"}:
                    aux_retr_cfg["use_gpu"] = False

                aux_name = str(aux_cfg.get("name", f"aux{aux_idx}_{aux_algo}")).strip().lower().replace(" ", "_")
                if not aux_name:
                    aux_name = f"aux{aux_idx}_{aux_algo}"

                t0 = time.perf_counter()
                aux_ret = build_retrieval_model(train_df, aux_retr_cfg, log_fn=progress)
                log(
                    f"Stage final.{aux_name}.train done in "
                    f"{time.perf_counter() - t0:.1f}s | algo={aux_ret.algorithm}",
                    with_resources=stage_resources,
                )

                t0 = time.perf_counter()
                cand_val_aux = generate_candidates(aux_ret, users_val, k=k_candidates, filter_seen=True, log_fn=progress)
                if same_eval_users:
                    cand_test_aux = cand_val_aux.copy(deep=False)
                    progress(f"final.{aux_name}.candidates reuse test_from_val=true")
                else:
                    cand_test_aux = generate_candidates(aux_ret, users_test, k=k_candidates, filter_seen=True, log_fn=progress)
                log(
                    f"Stage final.{aux_name}.candidates done in "
                    f"{time.perf_counter() - t0:.1f}s | val_rows={len(cand_val_aux):,} "
                    f"test_rows={len(cand_test_aux):,} reused_from_val={same_eval_users}",
                    with_resources=stage_resources,
                )
                aux_sources.append((aux_name, cand_val_aux, cand_test_aux, aux_weight, aux_algo))

            tt_w = float(fusion_cfg.get("two_tower_weight", 0.7))
            als_w = float(fusion_cfg.get("als_weight", 0.3))
            fusion_use_gpu = bool(fusion_cfg.get("use_gpu", True))
            val_sources: list[tuple[str, pd.DataFrame, float]] = [
                ("tt", cand_val_final, tt_w),
                ("als", cand_val_bas, als_w),
            ]
            test_sources: list[tuple[str, pd.DataFrame, float]] = [
                ("tt", cand_test_final, tt_w),
                ("als", cand_test_bas, als_w),
            ]
            for aux_name, cand_val_aux, cand_test_aux, aux_weight, _ in aux_sources:
                val_sources.append((aux_name, cand_val_aux, aux_weight))
                test_sources.append((aux_name, cand_test_aux, aux_weight))

            t0 = time.perf_counter()
            fusion_chunk_users = int(fusion_cfg.get("chunk_users", 0)) or None
            progress(
                "final.candidate_fusion start "
                f"sources={len(val_sources)} users_val={len(users_val):,} users_test={len(users_test):,} "
                f"k={k_candidates} use_gpu={fusion_use_gpu}"
            )
            if fusion_chunk_users is not None:
                progress(f"final.candidate_fusion chunk_users={fusion_chunk_users}")
            cand_val_final = _fuse_candidates_multi(
                val_sources,
                k=k_candidates,
                log_fn=progress,
                stage_label="final.candidate_fusion.val",
                use_gpu=fusion_use_gpu,
                chunk_users=fusion_chunk_users,
            )
            cand_test_final = _fuse_candidates_multi(
                test_sources,
                k=k_candidates,
                log_fn=progress,
                stage_label="final.candidate_fusion.test",
                use_gpu=fusion_use_gpu,
                chunk_users=fusion_chunk_users,
            )

            aux_weight_sum = sum(w for _, _, _, w, _ in aux_sources)
            raw_weight_sum = tt_w + als_w + aux_weight_sum
            norm = raw_weight_sum if raw_weight_sum > 0.0 else 1.0
            tt_w_n = tt_w / norm
            als_w_n = als_w / norm
            aux_desc = ", ".join(
                f"{name}:{algo}:{(w / norm):.2f}" for name, _, _, w, algo in aux_sources
            ) if aux_sources else "none"
            log(
                "Stage final.candidate_fusion done in "
                f"{time.perf_counter() - t0:.1f}s | "
                f"tt_weight={tt_w_n:.2f} als_weight={als_w_n:.2f} "
                f"aux={aux_desc} "
                f"val_rows={len(cand_val_final):,} test_rows={len(cand_test_final):,}"
                ,
                with_resources=stage_resources,
            )
            n_sources = len(val_sources)
            if n_sources > 2:
                final_retrieval_name = f"final_retrieval_two_tower_faiss_hybrid{n_sources}"
                final_ranker_name = f"final_ranker_two_tower_hybrid{n_sources}"
            else:
                final_retrieval_name = "final_retrieval_two_tower_faiss_hybrid"
                final_ranker_name = "final_ranker_two_tower_hybrid"

            del cand_val_bas, cand_test_bas
            for _, cand_val_aux, cand_test_aux, _, _ in aux_sources:
                del cand_val_aux, cand_test_aux
            gc.collect()
        else:
            final_retrieval_name = "final_retrieval_two_tower_faiss"
            final_ranker_name = "final_ranker_two_tower"

        t0 = time.perf_counter()
        metrics_retrieval_final = evaluate_predictions(
            ground_truth_df=test_df,
            prediction_df=cand_test_final,
            score_col="retrieval_score",
            ks=ks,
            model_name=final_retrieval_name,
        )
        log(f"Stage final.retrieval.eval done in {time.perf_counter() - t0:.1f}s")

        t0 = time.perf_counter()
        final_ranker_cfg = final_cfg.get("ranker", ranker_cfg)
        final_features_use_gpu = bool(final_ranker_cfg.get("features_use_gpu", ranker_cfg.get("features_use_gpu", True)))
        final_recent_n = int(final_ranker_cfg.get("recent_genres_n", 20))
        final_chunk_users = int(final_ranker_cfg.get("chunk_users", ranker_cfg.get("chunk_users", 0))) or None
        final_feature_ctx = _get_feature_context(
            use_gpu_flag=final_features_use_gpu,
            recent_genres_n=final_recent_n,
        )
        train_feats_final, feat_cols_final = build_candidate_features(
            cand_val_final,
            train_df,
            val_df,
            data_dir=data_cfg.get("dataset_dir"),
            recent_genres_n=final_recent_n,
            log_fn=progress,
            use_gpu=final_features_use_gpu,
            feature_context=final_feature_ctx,
            chunk_size_users=final_chunk_users,
        )
        test_feats_final, _ = build_candidate_features(
            cand_test_final,
            train_df,
            test_df,
            data_dir=data_cfg.get("dataset_dir"),
            recent_genres_n=final_recent_n,
            log_fn=progress,
            use_gpu=final_features_use_gpu,
            feature_context=final_feature_ctx,
            chunk_size_users=final_chunk_users,
        )
        log(
            "Stage final.ranker.features done in "
            f"{time.perf_counter() - t0:.1f}s | train_rows={len(train_feats_final):,} test_rows={len(test_feats_final):,}"
            ,
            with_resources=stage_resources,
        )

        del cand_val_final, cand_test_final
        gc.collect()

        t0 = time.perf_counter()
        ranker_final = train_ranker(train_feats_final, feat_cols_final, final_ranker_cfg, log_fn=progress)
        scored_final = score_ranker(ranker_final, test_feats_final, feat_cols_final, log_fn=progress)
        log(
            f"Stage final.ranker.train+score done in {time.perf_counter() - t0:.1f}s | scored_rows={len(scored_final):,}",
            with_resources=stage_resources,
        )
        _save_rank_scores(scored_final, "final_scores")

        del train_feats_final, test_feats_final
        gc.collect()

        t0 = time.perf_counter()
        metrics_ranker_final = evaluate_predictions(
            ground_truth_df=test_df,
            prediction_df=scored_final,
            score_col="rank_score",
            ks=ks,
            model_name=final_ranker_name,
        )
        log(f"Stage final.ranker.eval done in {time.perf_counter() - t0:.1f}s")

        metrics_parts = [metrics_retrieval_final, metrics_ranker_final]

        if use_ens:
            if scored_bas is None:
                raise RuntimeError("ranker_ensemble requires baseline scores, but baseline pipeline is disabled")
            w_base = float(ens_cfg.get("baseline_weight", 0.55))
            w_final = float(ens_cfg.get("final_weight", 0.45))
            t0 = time.perf_counter()
            scored_ens = _fuse_ranker_scores(
                scored_bas,
                scored_final,
                w_base=w_base,
                w_final=w_final,
                log_fn=progress,
                stage_label="final.ranker.ensemble",
            )
            metrics_ens = evaluate_predictions(
                ground_truth_df=test_df,
                prediction_df=scored_ens,
                score_col="rank_score",
                ks=ks,
                model_name="final_ranker_ensemble",
            )
            log(
                "Stage final.ranker.ensemble done in "
                f"{time.perf_counter() - t0:.1f}s | baseline_weight={w_base:.2f} final_weight={w_final:.2f}"
            )
            metrics_parts.append(metrics_ens)
            del scored_ens
            gc.collect()

        final_metrics = pd.concat(metrics_parts, ignore_index=True)
        final_metrics.to_csv(out_tables / "metrics_final.csv", index=False)
        log(f"Saved: {out_tables / 'metrics_final.csv'}")
        all_metrics.append(final_metrics)

        del scored_final
        gc.collect()

    if not use_ens:
        del scored_bas
        gc.collect()

    if not all_metrics:
        raise RuntimeError("No metrics produced. Enable baseline and/or final_model.")

    comparison = pd.concat(all_metrics, ignore_index=True)
    comparison.to_csv(out_tables / "metrics_comparison.csv", index=False)
    log(f"Saved: {out_tables / 'metrics_comparison.csv'}")

    t0 = time.perf_counter()
    _save_plots(comparison, out_figures)
    log(f"Stage plot.save done in {time.perf_counter() - t0:.1f}s")

    split_stats = pd.DataFrame(
        [
            {
                "train_interactions": len(train_df),
                "val_interactions": len(val_df),
                "test_interactions": len(test_df),
                "users_train": train_df["userId"].nunique(),
                "items_train": train_df["movieId"].nunique(),
            }
        ]
    )
    split_stats.to_csv(out_tables / "split_stats.csv", index=False)
    log(f"Saved: {out_tables / 'split_stats.csv'}")
    log(f"Pipeline done in {time.perf_counter() - t_all:.1f}s", with_resources=stage_resources)

    return {
        "baseline": baseline_metrics,
        "comparison": comparison,
        "split_stats": split_stats,
    }
