# Checkpoint: 2026-03-09 Quality Hybrid4 K220 Tuned

## Goal
Podniesc jakosc fullrun (MovieLens 25M) przez 4-way candidate fusion oraz strojenie wag ensemblu rankera.

## Result
- Config: `configs/fullrun_gpu_quality_goal_hybrid4_k220.yaml`
- Pipeline time: **2821.8s**
- Full data split, strict quality profile

## Metrics @20
- `final_ranker_ensemble` (run weights 0.35/0.65): Recall `0.178719`, NDCG `0.074670`, MAP `0.046138`
- `final_ranker_ensemble_tuned_fast` (fast sweep from saved scores, weights 0.25/0.75): Recall `0.179090`, NDCG `0.074772`, MAP `0.046167`

## Artifacts
- `metrics_comparison.csv`
- `metrics_final.csv`
- `ensemble_sweep_summary_k20_fast.csv`
- `metrics_ensemble_tuned_fast_from_scores.csv`
- `pipeline.log`
