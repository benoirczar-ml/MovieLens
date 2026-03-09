from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class TwoTowerArtifacts:
    user_to_idx: Dict[int, int]
    idx_to_user: np.ndarray
    item_to_idx: Dict[int, int]
    idx_to_item: np.ndarray
    user_emb: np.ndarray
    item_emb: np.ndarray
    index: object | None
    item_popularity: np.ndarray
    pop_blend_weight: float
    use_gpu: bool
    query_batch_size: int
    candidate_multiplier: int
    query_log_every_batches: int


class _TwoTowerModel:
    def __init__(self, n_users: int, n_items: int, dim: int):
        import torch

        self.torch = torch
        self.user = torch.nn.Embedding(n_users, dim)
        self.item = torch.nn.Embedding(n_items, dim)
        torch.nn.init.normal_(self.user.weight, std=0.05)
        torch.nn.init.normal_(self.item.weight, std=0.05)

    def parameters(self):
        return list(self.user.parameters()) + list(self.item.parameters())


def train_two_tower_faiss(train_df: pd.DataFrame, cfg: dict, log_fn: Callable[[str], None] | None = None) -> TwoTowerArtifacts:
    import torch

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_gpu = bool(cfg.get("use_gpu", False))
    require_gpu = bool(cfg.get("require_gpu", False))
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("Two-Tower requested use_gpu=true but CUDA is not available")
    if require_gpu and not use_gpu:
        raise RuntimeError("Two-Tower require_gpu=true but use_gpu=false")

    users = np.sort(train_df["userId"].unique())
    items = np.sort(train_df["movieId"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {i: j for j, i in enumerate(items)}

    uidx = train_df["userId"].map(user_to_idx).to_numpy(dtype=np.int64)
    iidx = train_df["movieId"].map(item_to_idx).to_numpy(dtype=np.int64)
    item_counts = train_df["movieId"].map(item_to_idx).value_counts().sort_index()

    n_users = len(users)
    n_items = len(items)
    dim = int(cfg.get("embedding_dim", 64))

    model = _TwoTowerModel(n_users, n_items, dim)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    model.user = model.user.to(device)
    model.item = model.item.to(device)

    lr = float(cfg.get("lr", 1e-3))
    batch_size = int(cfg.get("batch_size", 4096))
    epochs = int(cfg.get("epochs", 4))
    num_negatives = int(cfg.get("num_negatives", 8))
    weight_decay = float(cfg.get("weight_decay", 1e-6))
    grad_clip = float(cfg.get("grad_clip", 2.0))
    pop_pow = float(cfg.get("neg_sampling_pop_power", 0.75))
    pop_blend_weight = float(cfg.get("pop_blend_weight", 0.1))
    query_batch_size = int(cfg.get("query_batch_size", 1024))
    candidate_multiplier = int(cfg.get("candidate_multiplier", 3))
    train_log_every_batches = int(cfg.get("train_log_every_batches", 0))
    query_log_every_batches = max(1, int(cfg.get("query_log_every_batches", 20)))

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    all_pairs = np.column_stack([uidx, iidx])
    item_pop = np.ones(n_items, dtype=np.float64)
    item_pop[item_counts.index.to_numpy(dtype=np.int64)] = item_counts.to_numpy(dtype=np.float64)
    sampling_prob = np.power(item_pop, pop_pow)
    sampling_prob = sampling_prob / sampling_prob.sum()

    if log_fn:
        log_fn(
            "final.twotower.setup "
            f"device={device.type} users={n_users:,} items={n_items:,} pairs={len(all_pairs):,} "
            f"batch_size={batch_size} negatives={num_negatives}"
        )

    for epoch in range(epochs):
        rng.shuffle(all_pairs)
        n_batches = max(1, (len(all_pairs) + batch_size - 1) // batch_size)
        log_every = max(1, train_log_every_batches) if train_log_every_batches > 0 else max(1, n_batches // 8)
        loss_sum = 0.0
        seen_pairs = 0
        for batch_idx, start in enumerate(range(0, len(all_pairs), batch_size), start=1):
            batch = all_pairs[start : start + batch_size]
            bu = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            bi_pos = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            bi_neg_np = rng.choice(n_items, size=(len(batch), num_negatives), p=sampling_prob, replace=True)
            bi_neg = torch.tensor(bi_neg_np, dtype=torch.long, device=device)

            u = model.user(bu)
            i_pos = model.item(bi_pos)
            i_neg = model.item(bi_neg)

            pos_scores = (u * i_pos).sum(dim=1)
            neg_scores = (u.unsqueeze(1) * i_neg).sum(dim=2)

            # Pairwise objective with multiple sampled negatives.
            loss = -torch.nn.functional.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optim.step()

            bsz = int(len(batch))
            seen_pairs += bsz
            loss_sum += float(loss.item()) * bsz
            if log_fn and (batch_idx % log_every == 0 or batch_idx == n_batches):
                pct = 100.0 * batch_idx / max(1, n_batches)
                log_fn(
                    "final.twotower.train progress "
                    f"epoch={epoch + 1}/{epochs} batch={batch_idx}/{n_batches} "
                    f"loss={float(loss.item()):.4f} seen_pairs={seen_pairs:,} ({pct:.1f}%)"
                )
        if log_fn:
            avg_loss = loss_sum / max(1, seen_pairs)
            log_fn(f"final.twotower.train epoch_done epoch={epoch + 1}/{epochs} avg_loss={avg_loss:.5f}")

    user_emb = model.user.weight.detach().cpu().numpy().astype(np.float32)
    item_emb = model.item.weight.detach().cpu().numpy().astype(np.float32)

    user_norm = np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-12
    item_norm = np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-12
    user_emb = user_emb / user_norm
    item_emb = item_emb / item_norm
    item_popularity = np.log1p(item_pop.astype(np.float32))
    item_popularity = item_popularity / (item_popularity.max() + 1e-12)

    index = None
    if not use_gpu:
        try:
            import faiss

            index = faiss.IndexFlatIP(dim)
            index.add(item_emb)
        except Exception:
            index = None

    return TwoTowerArtifacts(
        user_to_idx=user_to_idx,
        idx_to_user=users,
        item_to_idx=item_to_idx,
        idx_to_item=items,
        user_emb=user_emb,
        item_emb=item_emb,
        index=index,
        item_popularity=item_popularity.astype(np.float32),
        pop_blend_weight=pop_blend_weight,
        use_gpu=use_gpu,
        query_batch_size=query_batch_size,
        candidate_multiplier=candidate_multiplier,
        query_log_every_batches=query_log_every_batches,
    )


def generate_candidates_faiss(
    artifacts: TwoTowerArtifacts,
    train_df: pd.DataFrame,
    user_ids: Iterable[int],
    k: int = 200,
    filter_seen: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    rows: List[dict] = []
    seen_map = {}
    if filter_seen:
        seen_map = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    valid_users = []
    valid_uidx = []
    for uid in user_ids:
        uidx = artifacts.user_to_idx.get(int(uid))
        if uidx is None:
            continue
        valid_users.append(int(uid))
        valid_uidx.append(int(uidx))

    if artifacts.use_gpu:
        import torch

        device = torch.device("cuda")
        item_emb_t = torch.from_numpy(artifacts.item_emb).to(device=device, dtype=torch.float32)
        topn = min(int(k * artifacts.candidate_multiplier), artifacts.item_emb.shape[0])
        batch_size = max(1, int(artifacts.query_batch_size))
        n_batches = max(1, (len(valid_users) + batch_size - 1) // batch_size)
        log_every = max(1, int(artifacts.query_log_every_batches))
        if log_fn:
            log_fn(
                "final.twotower.candidates start "
                f"users={len(valid_users):,} topn={topn} batch_size={batch_size} batches={n_batches}"
            )

        for batch_idx, start in enumerate(range(0, len(valid_users), batch_size), start=1):
            end = start + batch_size
            users_b = valid_users[start:end]
            uidx_b = valid_uidx[start:end]

            user_emb_b = torch.from_numpy(artifacts.user_emb[np.array(uidx_b, dtype=np.int64)]).to(device=device, dtype=torch.float32)
            scores_b = user_emb_b @ item_emb_t.T
            vals_b, idxs_b = torch.topk(scores_b, k=topn, dim=1)

            vals_np = vals_b.detach().cpu().numpy()
            idxs_np = idxs_b.detach().cpu().numpy()

            for row_i, uid in enumerate(users_b):
                cand_idxs = idxs_np[row_i]
                cand_scores = vals_np[row_i]
                seen_items = seen_map.get(uid, set()) if filter_seen else set()
                taken = 0
                for it_idx, score in zip(cand_idxs, cand_scores):
                    mid = int(artifacts.idx_to_item[int(it_idx)])
                    if mid in seen_items:
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
                log_fn(f"final.twotower.candidates progress users={done:,}/{len(valid_users):,} ({pct:.1f}%)")
    else:
        for uid, uidx in zip(valid_users, valid_uidx):
            uq = artifacts.user_emb[uidx : uidx + 1]

            if artifacts.index is not None:
                scores, idxs = artifacts.index.search(uq.astype(np.float32), k * artifacts.candidate_multiplier)
                cand_idxs = idxs[0]
                cand_scores = scores[0]
            else:
                cand_scores = artifacts.item_emb @ uq[0]
                cand_idxs = np.argsort(-cand_scores)[: k * artifacts.candidate_multiplier]
                cand_scores = cand_scores[cand_idxs]

            seen_items = seen_map.get(uid, set()) if filter_seen else set()
            taken = 0
            for it_idx, score in zip(cand_idxs, cand_scores):
                mid = int(artifacts.idx_to_item[int(it_idx)])
                if mid in seen_items:
                    continue
                base_score = float(score)
                pop_score = float(artifacts.item_popularity[int(it_idx)])
                final_score = (1.0 - artifacts.pop_blend_weight) * base_score + artifacts.pop_blend_weight * pop_score
                rows.append({"userId": uid, "movieId": mid, "retrieval_score": final_score})
                taken += 1
                if taken >= k:
                    break
        if log_fn:
            log_fn(f"final.twotower.candidates cpu_done users={len(valid_users):,}")

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["userId"] = out["userId"].astype("int32")
    out["movieId"] = out["movieId"].astype("int32")
    out["retrieval_score"] = out["retrieval_score"].astype("float32")
    return out
