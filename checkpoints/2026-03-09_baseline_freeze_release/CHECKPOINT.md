# Checkpoint: 2026-03-09 Baseline Freeze Release

## Intent
Zamrozenie etapu baseline/quality-max przed przejsciem do nowej rodziny modeli.

## Locked defaults
- `configs/production.yaml` -> hybrid4 candidate fusion + tuned ensemble (`baseline=0.25`, `final=0.75`)
- `configs/fullrun_gpu_quality_goal_hybrid4_k220.yaml`

## Locked metrics (@20)
- `final_ranker_ensemble_tuned_fast`: Recall `0.179090`, NDCG `0.074772`, MAP `0.046167`, MRR `0.046167`
- fullrun runtime: `2821.8s`

## Evidence
- `outputs/fullrun_gpu_quality_goal_hybrid4_k220/tables/metrics_comparison.csv`
- `outputs/fullrun_gpu_quality_goal_hybrid4_k220/tables/metrics_ensemble_tuned_fast_from_scores.csv`
- `outputs/tables/profile_ladder_2026-03-09.csv`
- `docs/BASELINE_FREEZE_2026-03-09.md`
