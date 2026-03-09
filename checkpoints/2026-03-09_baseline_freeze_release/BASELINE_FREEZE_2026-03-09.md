# Baseline Freeze - MovieLens 25M (2026-03-09)

## One-line summary
Projekt jest zamrozony w stabilnym stanie "quality-max" z czytelnym trade-offem czasu vs jakosc.
Domyslny profil produkcyjny: `configs/production.yaml` (hybrid4 + tuned ensemble 0.25/0.75).

## Final metrics (@20)
Dataset/protocol: MovieLens 25M, temporal holdout (`val_k=1`, `test_k=1`), `min_rating=3.5`.

| Model | Recall@20 | NDCG@20 | MAP@20 | MRR@20 |
|---|---:|---:|---:|---:|
| baseline_ranker | 0.1712 | 0.0708 | 0.0434 | 0.0434 |
| final_ranker_two_tower_hybrid4 | 0.1778 | 0.0743 | 0.0459 | 0.0459 |
| final_ranker_ensemble (run) | 0.1787 | 0.0747 | 0.0461 | 0.0461 |
| final_ranker_ensemble (tuned fast) | **0.1791** | **0.0748** | **0.0462** | **0.0462** |

Source files:
- `outputs/fullrun_gpu_quality_goal_hybrid4_k220/tables/metrics_comparison.csv`
- `outputs/fullrun_gpu_quality_goal_hybrid4_k220/tables/metrics_ensemble_tuned_fast_from_scores.csv`

## Runtime vs quality ladder

| Profile | Runtime (s) | Ranker Recall@20 | Ranker NDCG@20 | Ranker MAP@20 |
|---|---:|---:|---:|---:|
| speed150 | 134.3 | 0.0860 | 0.0363 | 0.0226 |
| speed150_quality | 145.2 | 0.0902 | 0.0379 | 0.0235 |
| quality_200_ep2_k70 | 198.4 | 0.1038 | 0.0430 | 0.0263 |
| quality_goal_hybrid4_k220_tuned_fast | **2821.8** | **0.1791** | **0.0748** | **0.0462** |

Source file:
- `outputs/tables/profile_ladder_2026-03-09.csv`

## Conclusions
- Dalsze mikrooptymalizacje GPU daja juz niski zwrot: jakosc rosnie wolno, runtime rosnie szybko.
- Ten checkpoint jest odpowiedni jako "mocny baseline" pod kolejny etap (architektura dedykowana: sequential/graph).
- Dla rekrutera wynik jest czytelny: jedna tabela, jeden profil produkcyjny, jeden zestaw artefaktow.

## Repro
```bash
python scripts/run_production.py
```

## Freeze artifacts
- Config: `configs/production.yaml`
- Quality run config: `configs/fullrun_gpu_quality_goal_hybrid4_k220.yaml`
- Metrics: `outputs/fullrun_gpu_quality_goal_hybrid4_k220/tables/`
- Profile ladder: `outputs/tables/profile_ladder_2026-03-09.csv`
- Checkpoint package: `checkpoints/2026-03-09_baseline_freeze_release/`
