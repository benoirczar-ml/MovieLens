from __future__ import annotations

import time
from typing import Callable

import pandas as pd


def _sample_ranker_train_df(
    train_feats: pd.DataFrame,
    cfg: dict,
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    max_rows = int(cfg.get("train_max_rows", 0) or 0)
    neg_ratio = float(cfg.get("train_negative_ratio", 0.0) or 0.0)
    seed = int(cfg.get("train_sample_seed", cfg.get("seed", 42)))

    if max_rows <= 0 and neg_ratio <= 0.0:
        return train_feats

    df = train_feats
    if "label" not in df.columns or df.empty:
        return df

    pos = df[df["label"] > 0]
    neg = df[df["label"] <= 0]

    if neg_ratio > 0.0 and len(pos) > 0 and len(neg) > 0:
        target_neg = int(len(pos) * neg_ratio)
        if 0 < target_neg < len(neg):
            neg = neg.sample(n=target_neg, random_state=seed)
        df = pd.concat([pos, neg], ignore_index=True)

    if max_rows > 0 and len(df) > max_rows:
        pos = df[df["label"] > 0]
        neg = df[df["label"] <= 0]
        if len(pos) >= max_rows:
            df = pos.sample(n=max_rows, random_state=seed)
        else:
            need_neg = max_rows - len(pos)
            if need_neg > 0 and len(neg) > need_neg:
                neg = neg.sample(n=need_neg, random_state=seed)
            df = pd.concat([pos, neg], ignore_index=True)

    if len(df) > 1:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if log_fn:
        pos_rate = float(df["label"].mean()) if len(df) else 0.0
        log_fn(
            "ranker.train sampled "
            f"rows={len(df):,} positive_rate={pos_rate:.4f} "
            f"max_rows={max_rows} neg_ratio={neg_ratio:.2f}"
        )
    return df


def train_ranker(
    train_feats: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    log_fn: Callable[[str], None] | None = None,
):
    train_df = _sample_ranker_train_df(train_feats, cfg, log_fn=log_fn)
    X = train_df[feature_cols]
    y = train_df["label"].astype(int)
    if log_fn:
        pos_rate = float(y.mean()) if len(y) else 0.0
        log_fn(f"ranker.train start rows={len(X):,} cols={len(feature_cols)} positive_rate={pos_rate:.4f}")

    use_lightgbm = str(cfg.get("framework", "xgboost")).lower() == "lightgbm"

    if use_lightgbm:
        try:
            from lightgbm import LGBMClassifier
            use_gpu = str(cfg.get("device", "cpu")).lower() in {"cuda", "gpu"}

            model = LGBMClassifier(
                n_estimators=int(cfg.get("n_estimators", 300)),
                learning_rate=float(cfg.get("learning_rate", 0.05)),
                num_leaves=int(cfg.get("num_leaves", 31)),
                subsample=float(cfg.get("subsample", 0.9)),
                colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
                random_state=int(cfg.get("seed", 42)),
                device_type="gpu" if use_gpu else "cpu",
            )
            t_fit = time.perf_counter()
            model.fit(X, y)
            if log_fn:
                log_fn(f"ranker.train done framework=lightgbm device={'gpu' if use_gpu else 'cpu'} in={time.perf_counter() - t_fit:.1f}s")
            return model
        except Exception:
            pass

    try:
        from xgboost import XGBClassifier
        device = str(cfg.get("device", "cpu")).lower()

        model = XGBClassifier(
            n_estimators=int(cfg.get("n_estimators", 300)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 6)),
            subsample=float(cfg.get("subsample", 0.9)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=str(cfg.get("tree_method", "hist")),
            random_state=int(cfg.get("seed", 42)),
            n_jobs=int(cfg.get("n_jobs", 4)),
            device="cuda" if device in {"cuda", "gpu"} else "cpu",
            max_bin=int(cfg.get("max_bin", 256)),
        )
        t_fit = time.perf_counter()
        model.fit(X, y)
        if log_fn:
            log_fn(
                f"ranker.train done framework=xgboost device={'cuda' if device in {'cuda', 'gpu'} else 'cpu'} "
                f"in={time.perf_counter() - t_fit:.1f}s"
            )
        return model
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(random_state=int(cfg.get("seed", 42)))
        t_fit = time.perf_counter()
        model.fit(X, y)
        if log_fn:
            log_fn(f"ranker.train done framework=sklearn_gb in={time.perf_counter() - t_fit:.1f}s")
        return model


def score_ranker(
    model,
    feats: pd.DataFrame,
    feature_cols: list[str],
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    X = feats[feature_cols]
    scores = None
    if log_fn:
        log_fn(f"ranker.score start rows={len(X):,} cols={len(feature_cols)}")

    # XGBoost with CUDA can warn and slow down if prediction input stays on CPU.
    # Try GPU-backed prediction first when device is set to CUDA.
    if hasattr(model, "get_params"):
        params = model.get_params()
        device = str(params.get("device", "cpu")).lower()
        if device in {"cuda", "gpu"}:
            try:
                import cupy as cp

                X_gpu = cp.asarray(X.to_numpy(dtype="float32"))
                if hasattr(model, "predict_proba"):
                    scores = cp.asnumpy(model.predict_proba(X_gpu)[:, 1])
                else:
                    scores = cp.asnumpy(model.predict(X_gpu))
            except Exception:
                scores = None

    if scores is None:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.decision_function(X)

    out = feats[["userId", "movieId"]].copy()
    out["rank_score"] = scores
    if log_fn:
        log_fn(f"ranker.score done rows={len(out):,}")
    return out
