# Checkpoint: 2026-03-09 Quality-200 (k70)

## Goal
Podbić jakość przy runtime w widełkach ~150-200s.

## Result
- Profile: `configs/fullrun_gpu_quality_200_ep2_k70.yaml` (alias: `configs/fullrun_gpu_quality_best.yaml`)
- Pipeline: **198.4s**
- Two-Tower: `epochs=2`, `k_candidates=70`, `query_batch_size=12288`

## Metrics @20
- `final_retrieval_two_tower_faiss`: Recall `0.081589`, NDCG `0.031138`, MAP `0.017514`
- `final_ranker_two_tower`: Recall `0.103758`, NDCG `0.043015`, MAP `0.026312`

## Comparison (vs quality_200_ep2 k60)
- runtime: `187.4s` -> `198.4s`
- ranker Recall@20: `0.101115` -> `0.103758` (+0.002643)
- ranker NDCG@20: `0.042053` -> `0.043015` (+0.000962)

## Artifacts
- `outputs/fullrun_gpu_quality_200_ep2_k70/tables/metrics_comparison.csv`
- `outputs/tables/profile_ladder_2026-03-09.csv`
