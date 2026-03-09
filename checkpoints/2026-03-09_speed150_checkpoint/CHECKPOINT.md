# Checkpoint: 2026-03-09 Speed150 Milestone

## Goal
Zbić runtime pełnego runu (full data) do ~150s.

## Result
- Config: `configs/fullrun_gpu_speed150.yaml`
- Pipeline time: **134.3s**
- Dataset: MovieLens 25M full train/val/test split
- Baseline branch: disabled (`baseline.enabled=false`)
- Final branch: Two-Tower + Ranker (GPU)

## Key Metrics (@20)
- `final_retrieval_two_tower_faiss`: Recall `0.074416`, NDCG `0.028894`, MAP `0.016576`, MRR `0.016576`
- `final_ranker_two_tower`: Recall `0.086008`, NDCG `0.036331`, MAP `0.022622`, MRR `0.022622`

## Stage Timing Snapshot
- `data.load`: 16.7s
- `data.split`: 11.6s
- `final.twotower.train`: 40.4s
- `final.twotower.candidates`: 18.4s
- `final.retrieval.eval`: 9.7s
- `final.ranker.features`: 18.1s
- `final.ranker.train+score`: 5.5s
- `final.ranker.eval`: 10.8s
- `Pipeline done`: 134.3s

## Notes
To profil szybkościowy (speed-first), jakościowo słabszy od profilu `strict`.
Profil `strict` pozostaje domyślny dla jakości (`configs/production.yaml`).
