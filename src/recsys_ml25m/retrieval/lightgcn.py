from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class LightGCNArtifacts:
    user_to_idx: Dict[int, int]
    idx_to_user: np.ndarray
    item_to_idx: Dict[int, int]
    idx_to_item: np.ndarray
    user_emb: np.ndarray
    item_emb: np.ndarray
    item_popularity: np.ndarray
    pop_blend_weight: float
    seen_items_by_user: Dict[int, set[int]]
    use_gpu: bool
    query_batch_size: int
    candidate_multiplier: int
    query_log_every_batches: int


def _build_norm_adj(
    n_users: int,
    n_items: int,
    uidx_edges: np.ndarray,
    iidx_edges: np.ndarray,
    device,
    torch,
):
    n_nodes = int(n_users + n_items)
    i_nodes = iidx_edges + n_users
    row = np.concatenate([uidx_edges, i_nodes]).astype(np.int64, copy=False)
    col = np.concatenate([i_nodes, uidx_edges]).astype(np.int64, copy=False)

    deg = np.bincount(row, minlength=n_nodes).astype(np.float32, copy=False)
    deg = np.clip(deg, 1.0, None)
    vals = (1.0 / np.sqrt(deg[row] * deg[col])).astype(np.float32, copy=False)

    indices_np = np.vstack([row, col]).astype(np.int64, copy=False)
    indices = torch.from_numpy(indices_np).to(device)
    values = torch.from_numpy(vals).to(device)
    adj = torch.sparse_coo_tensor(indices, values, size=(n_nodes, n_nodes), device=device).coalesce()
    return adj


def train_lightgcn(
    train_df: pd.DataFrame,
    cfg: dict,
    log_fn: Callable[[str], None] | None = None,
) -> LightGCNArtifacts:
    import torch

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_gpu = bool(cfg.get("use_gpu", False))
    require_gpu = bool(cfg.get("require_gpu", False))
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("LightGCN requested use_gpu=true but CUDA is not available")
    if require_gpu and not use_gpu:
        raise RuntimeError("LightGCN require_gpu=true but use_gpu=false")

    users = np.sort(train_df["userId"].unique())
    items = np.sort(train_df["movieId"].unique())
    user_to_idx = {int(u): i for i, u in enumerate(users)}
    item_to_idx = {int(i): j for j, i in enumerate(items)}

    mapped = train_df[["userId", "movieId"]].copy()
    mapped["uidx"] = mapped["userId"].map(user_to_idx).astype("int64")
    mapped["iidx"] = mapped["movieId"].map(item_to_idx).astype("int64")

    # Keep training positives from all interactions, but deduplicate graph edges.
    uidx_train = mapped["uidx"].to_numpy(dtype=np.int64, copy=False)
    iidx_train = mapped["iidx"].to_numpy(dtype=np.int64, copy=False)
    unique_pairs = mapped[["uidx", "iidx"]].drop_duplicates(ignore_index=True)
    uidx_edges = unique_pairs["uidx"].to_numpy(dtype=np.int64, copy=False)
    iidx_edges = unique_pairs["iidx"].to_numpy(dtype=np.int64, copy=False)

    n_users = len(users)
    n_items = len(items)
    dim = int(cfg.get("embedding_dim", 96))
    layers = int(cfg.get("num_layers", 2))
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    lr = float(cfg.get("lr", 1e-3))
    epochs = int(cfg.get("epochs", 6))
    batch_size = int(cfg.get("batch_size", 8192))
    num_negatives = int(cfg.get("num_negatives", 8))
    weight_decay = float(cfg.get("weight_decay", 1e-6))
    grad_clip = float(cfg.get("grad_clip", 2.0))
    pop_pow = float(cfg.get("neg_sampling_pop_power", 0.75))
    pop_blend_weight = float(cfg.get("pop_blend_weight", 0.08))

    query_batch_size = int(cfg.get("query_batch_size", 2048))
    candidate_multiplier = int(cfg.get("candidate_multiplier", 3))
    train_log_every_batches = int(cfg.get("train_log_every_batches", 0))
    query_log_every_batches = max(1, int(cfg.get("query_log_every_batches", 10)))

    user_emb = torch.nn.Embedding(n_users, dim).to(device)
    item_emb = torch.nn.Embedding(n_items, dim).to(device)
    torch.nn.init.normal_(user_emb.weight, std=0.05)
    torch.nn.init.normal_(item_emb.weight, std=0.05)
    optim = torch.optim.Adam(
        list(user_emb.parameters()) + list(item_emb.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    item_counts = mapped["iidx"].value_counts().sort_index()
    item_pop = np.ones(n_items, dtype=np.float64)
    item_pop[item_counts.index.to_numpy(dtype=np.int64)] = item_counts.to_numpy(dtype=np.float64)
    sampling_prob = np.power(item_pop, pop_pow)
    sampling_prob = sampling_prob / sampling_prob.sum()

    rng = np.random.default_rng(seed)
    all_pairs = np.column_stack([uidx_train, iidx_train])

    if log_fn:
        log_fn(
            "retrieval.lightgcn.setup "
            f"device={device.type} users={n_users:,} items={n_items:,} interactions={len(all_pairs):,} "
            f"edges={len(uidx_edges):,} dim={dim} layers={layers} batch_size={batch_size}"
        )

    for epoch in range(epochs):
        rng.shuffle(all_pairs)
        n_batches = max(1, (len(all_pairs) + batch_size - 1) // batch_size)
        log_every = max(1, train_log_every_batches) if train_log_every_batches > 0 else max(1, n_batches // 8)
        seen_pairs = 0
        loss_sum = 0.0

        for batch_idx, start in enumerate(range(0, len(all_pairs), batch_size), start=1):
            batch = all_pairs[start : start + batch_size]
            bu = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            bi_pos = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            bi_neg_np = rng.choice(n_items, size=(len(batch), num_negatives), p=sampling_prob, replace=True)
            bi_neg = torch.tensor(bi_neg_np, dtype=torch.long, device=device)

            u = user_emb(bu)
            i_pos = item_emb(bi_pos)
            i_neg = item_emb(bi_neg)
            pos_scores = (u * i_pos).sum(dim=1)
            neg_scores = (u.unsqueeze(1) * i_neg).sum(dim=2)
            loss = -torch.nn.functional.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(user_emb.parameters()) + list(item_emb.parameters()),
                max_norm=grad_clip,
            )
            optim.step()

            bsz = int(len(batch))
            seen_pairs += bsz
            loss_sum += float(loss.item()) * bsz
            if log_fn and (batch_idx % log_every == 0 or batch_idx == n_batches):
                pct = 100.0 * batch_idx / max(1, n_batches)
                log_fn(
                    "retrieval.lightgcn.train progress "
                    f"epoch={epoch + 1}/{epochs} batch={batch_idx}/{n_batches} "
                    f"loss={float(loss.item()):.4f} seen_pairs={seen_pairs:,} ({pct:.1f}%)"
                )

        if log_fn:
            avg_loss = loss_sum / max(1, seen_pairs)
            log_fn(f"retrieval.lightgcn.train epoch_done epoch={epoch + 1}/{epochs} avg_loss={avg_loss:.5f}")

    with torch.no_grad():
        adj = _build_norm_adj(
            n_users=n_users,
            n_items=n_items,
            uidx_edges=uidx_edges,
            iidx_edges=iidx_edges,
            device=device,
            torch=torch,
        )
        e0 = torch.cat([user_emb.weight, item_emb.weight], dim=0)
        embs = [e0]
        e = e0
        for layer_idx in range(layers):
            e = torch.sparse.mm(adj, e)
            embs.append(e)
            if log_fn:
                log_fn(f"retrieval.lightgcn.propagation layer={layer_idx + 1}/{layers}")

        efinal = torch.stack(embs, dim=0).mean(dim=0)
        user_final = efinal[:n_users]
        item_final = efinal[n_users:]

    user_np = user_final.detach().cpu().numpy().astype(np.float32)
    item_np = item_final.detach().cpu().numpy().astype(np.float32)

    user_norm = np.linalg.norm(user_np, axis=1, keepdims=True) + 1e-12
    item_norm = np.linalg.norm(item_np, axis=1, keepdims=True) + 1e-12
    user_np = user_np / user_norm
    item_np = item_np / item_norm

    item_popularity = np.log1p(item_pop.astype(np.float32))
    item_popularity = item_popularity / (float(item_popularity.max()) + 1e-12)

    seen_items_by_user = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    return LightGCNArtifacts(
        user_to_idx=user_to_idx,
        idx_to_user=users.astype(np.int32, copy=False),
        item_to_idx=item_to_idx,
        idx_to_item=items.astype(np.int32, copy=False),
        user_emb=user_np,
        item_emb=item_np,
        item_popularity=item_popularity.astype(np.float32, copy=False),
        pop_blend_weight=pop_blend_weight,
        seen_items_by_user=seen_items_by_user,
        use_gpu=use_gpu,
        query_batch_size=query_batch_size,
        candidate_multiplier=candidate_multiplier,
        query_log_every_batches=query_log_every_batches,
    )


def generate_candidates_lightgcn(
    artifacts: LightGCNArtifacts,
    user_ids: Iterable[int],
    k: int = 200,
    filter_seen: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    rows: List[dict] = []
    valid_users: list[int] = []
    valid_uidx: list[int] = []

    for uid in user_ids:
        uidx = artifacts.user_to_idx.get(int(uid))
        if uidx is None:
            continue
        valid_users.append(int(uid))
        valid_uidx.append(int(uidx))

    topn = min(int(k * artifacts.candidate_multiplier), artifacts.item_emb.shape[0])
    if artifacts.use_gpu:
        import torch

        device = torch.device("cuda")
        item_emb_t = torch.from_numpy(artifacts.item_emb).to(device=device, dtype=torch.float32)
        batch_size = max(1, int(artifacts.query_batch_size))
        n_batches = max(1, (len(valid_users) + batch_size - 1) // batch_size)
        log_every = max(1, int(artifacts.query_log_every_batches))
        if log_fn:
            log_fn(
                "retrieval.lightgcn.candidates start "
                f"users={len(valid_users):,} topn={topn} batch_size={batch_size} batches={n_batches}"
            )

        for batch_idx, start in enumerate(range(0, len(valid_users), batch_size), start=1):
            end = start + batch_size
            users_b = valid_users[start:end]
            uidx_b = valid_uidx[start:end]
            uemb_b = torch.from_numpy(artifacts.user_emb[np.array(uidx_b, dtype=np.int64)]).to(device=device, dtype=torch.float32)
            scores_b = uemb_b @ item_emb_t.T
            vals_b, idxs_b = torch.topk(scores_b, k=topn, dim=1)

            vals_np = vals_b.detach().cpu().numpy()
            idxs_np = idxs_b.detach().cpu().numpy()
            for row_i, uid in enumerate(users_b):
                seen = artifacts.seen_items_by_user.get(uid, set()) if filter_seen else set()
                taken = 0
                for it_idx, score in zip(idxs_np[row_i], vals_np[row_i]):
                    mid = int(artifacts.idx_to_item[int(it_idx)])
                    if mid in seen:
                        continue
                    base_score = float(score)
                    pop_score = float(artifacts.item_popularity[int(it_idx)])
                    final_score = (1.0 - artifacts.pop_blend_weight) * base_score + artifacts.pop_blend_weight * pop_score
                    rows.append({"userId": uid, "movieId": mid, "retrieval_score": final_score})
                    taken += 1
                    if taken >= k:
                        break

            if log_fn and (batch_idx % log_every == 0 or batch_idx == n_batches):
                done = min(end, len(valid_users))
                pct = 100.0 * done / max(1, len(valid_users))
                log_fn(f"retrieval.lightgcn.candidates progress users={done:,}/{len(valid_users):,} ({pct:.1f}%)")
    else:
        if log_fn:
            log_fn(f"retrieval.lightgcn.candidates cpu_start users={len(valid_users):,} topn={topn}")
        item_emb = artifacts.item_emb
        for uid, uidx in zip(valid_users, valid_uidx):
            seen = artifacts.seen_items_by_user.get(uid, set()) if filter_seen else set()
            scores = item_emb @ artifacts.user_emb[int(uidx)]
            order = np.argsort(-scores, kind="mergesort")[:topn]
            taken = 0
            for it_idx in order:
                mid = int(artifacts.idx_to_item[int(it_idx)])
                if mid in seen:
                    continue
                base_score = float(scores[int(it_idx)])
                pop_score = float(artifacts.item_popularity[int(it_idx)])
                final_score = (1.0 - artifacts.pop_blend_weight) * base_score + artifacts.pop_blend_weight * pop_score
                rows.append({"userId": uid, "movieId": mid, "retrieval_score": final_score})
                taken += 1
                if taken >= k:
                    break

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["userId"] = out["userId"].astype("int32")
    out["movieId"] = out["movieId"].astype("int32")
    out["retrieval_score"] = out["retrieval_score"].astype("float32")
    return out
