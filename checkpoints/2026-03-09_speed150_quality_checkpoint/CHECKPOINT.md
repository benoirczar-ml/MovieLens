# Checkpoint: 2026-03-09 Speed150 Quality

## Goal
Utrzymać runtime fullrun <150s i podnieść metryki względem profilu speed150.

## Result
- Config: `configs/fullrun_gpu_speed150_quality.yaml`
- Pipeline time: **145.2s**
- Dataset: MovieLens 25M full split
- Baseline: disabled
- Final path: Two-Tower + GPU ranker

## Metrics @20
- `final_retrieval_two_tower_faiss`: Recall `0.074948`, NDCG `0.029023`, MAP `0.016609`
- `final_ranker_two_tower`: Recall `0.090167`, NDCG `0.037892`, MAP `0.023490`

## Delta vs speed150 (previous)
- `speed150` time: `134.3s` (speed-first) vs `145.2s` (quality-boost under <150s)
- `final_ranker_two_tower` @20:
  - Recall: `0.085989` -> `0.090167` (+0.004178)
  - NDCG: `0.036123` -> `0.037892` (+0.001769)
  - MAP: `0.022374` -> `0.023490` (+0.001116)

## Runtime optimization included
Pipeline now reuses candidate generation for test split when `users_val == users_test`:
- `final.twotower.candidates reuse test_from_val=true`
- same mechanism added for baseline/aux retrieval.
