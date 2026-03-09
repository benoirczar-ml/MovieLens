# Checkpoint: 2026-03-09 GPU v2 LightGCN Block1

## Scope
- Added LightGCN retrieval backend integrated with pipeline retrieval API.
- Added gpu_v2 dual-protocol runner (`full-catalog + sampled benchmark`).
- Enabled optional parquet table outputs alongside csv.
- Added smoke config and executed end-to-end smoke run.

## Smoke metrics (@20)
- full-catalog `final_ranker_ensemble`: Recall `0.147333`, NDCG `0.058149`, MAP `0.033668`
- benchmark `hybrid_sampled`: Recall `0.917000`, NDCG `0.593890` (3k users, 100 negatives)

## Runtime (smoke)
- `run_gpu_v2` full+benchmark done in approx 54s on current slice.
- full-catalog pipeline phase: `44.5s`.

## Artifacts
- `metrics_comparison.csv`
- `benchmark_sampled_metrics_gpu_v2_smoke.csv`
- `run_gpu_v2.log`
- `gpu_v2_lightgcn.yaml`
- `gpu_v2_lightgcn_smoke.yaml`
- `GPU_V2_PROTOCOLS.md`
