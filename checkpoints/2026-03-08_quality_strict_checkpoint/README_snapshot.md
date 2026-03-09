# Final Results (TL;DR)

MovieLens 25M recommender pipeline działa end-to-end (`retrieval -> ranking -> eval`) i uruchamia się jedną komendą.
Najlepszy wynik w obecnym setupie daje **final_ranker_ensemble** (3-way hybrid retrieval + ensemble rankera).

| Model | Recall@20 | NDCG@20 | MAP@20 | MRR@20 |
|---|---:|---:|---:|---:|
| baseline_retrieval_als | 0.1355 | 0.0513 | 0.0285 | 0.0285 |
| baseline_ranker | 0.1658 | 0.0681 | 0.0414 | 0.0414 |
| final_retrieval_two_tower_faiss_hybrid3 | 0.1334 | 0.0504 | 0.0280 | 0.0280 |
| final_ranker_two_tower_hybrid3 | 0.1698 | 0.0710 | 0.0439 | 0.0439 |
| final_ranker_ensemble | **0.1705** | **0.0712** | **0.0440** | **0.0440** |

Wyniki top-line pochodzą z profilu `strict` (`configs/fullrun_gpu_hybrid3_cosine.yaml`, run z 8 marca 2026) i są zapisane w [`outputs/fullrun_gpu_hybrid3_cosine/tables/metrics_comparison.csv`](outputs/fullrun_gpu_hybrid3_cosine/tables/metrics_comparison.csv).

## Problem

Budujemy system rekomendacji na MovieLens 25M:
1. retrieval (candidate generation),
2. ranking,
3. features user-item-session/context,
4. offline evaluation: Recall@K, HitRate@K, NDCG@K, MAP@K, MRR@K.

## Co jest zaimplementowane

- Retrieval baseline: `implicit` ALS (opcjonalnie BPR/popularity fallback).
- Retrieval final: Two-Tower (PyTorch) + ANN retrieval (FAISS, fallback brute-force dot-product).
- Ranker: XGBoost lub LightGBM (fallback sklearn).
- Holdout temporalny typu leave-last (`val_k=1`, `test_k=1`).
- Automatyczny zapis metryk i wykresów do `outputs/`.

## Project Structure

```text
recsys-movielens25m/
  src/recsys_ml25m/
    data/
    retrieval/
    ranking/
    eval/
    inference/
  configs/
  scripts/
  outputs/{tables,figures}
  tests/
```

## Quick Start

```bash
cd recsys-movielens25m
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/final.yaml
```

## GPU-Only Run

Pipeline ma teraz logowanie etapowe (`start/stop`, czasy, liczności) oraz logi pośrednie w długich etapach (batch/epoch progress) + snapshot zasobów (`cpu/ram/gpu/vram/temp`). Tryb strict GPU działa bez fallbacku CPU.

```bash
./scripts/install_implicit_cuda.sh
python scripts/run_pipeline.py --config configs/final_gpu.yaml
python scripts/run_pipeline.py --config configs/final_gpu_cosine.yaml
python scripts/run_pipeline.py --config configs/final_gpu_bm25.yaml
python scripts/run_pipeline.py --config configs/final_gpu_hybrid3_cosine.yaml
python scripts/run_pipeline.py --config configs/final_gpu_hybrid3_cosine_balanced.yaml
python scripts/run_pipeline.py --config configs/final_gpu_hybrid3_cosine_gpuheavy.yaml
python scripts/run_pipeline.py --config configs/final_gpu_aggressive.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_best.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_robust.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_hybrid3_cosine.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_hybrid3_cosine_balanced.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_hybrid3_cosine_gpuheavy.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_best.yaml --log-file outputs/fullrun_gpu_best/logs/pipeline.log
python scripts/time_cv_stability.py --config configs/final_gpu_hybrid3_cosine.yaml --eval-offsets 0,1,2 --calib-offsets 1,2 --ensemble-baseline-weights 0.45,0.55,0.65 --target-k 20 --metric ndcg
```

Fullrun profile compare @K=20 (runs z 8 marca 2026):
- `strict` (`configs/fullrun_gpu_hybrid3_cosine.yaml`): `pipeline=1288.8s`, Recall `0.170487`, NDCG `0.071237`, MAP `0.044046`
- `balanced` (`configs/fullrun_gpu_hybrid3_cosine_balanced.yaml`): `pipeline=1128.4s`, Recall `0.170822`, NDCG `0.071145`, MAP `0.043808`
- `gpuheavy` (`configs/fullrun_gpu_hybrid3_cosine_gpuheavy.yaml`): `pipeline=1097.8s`, Recall `0.170871`, NDCG `0.071223`, MAP `0.043903`

Artefakty fullrun:
- `outputs/fullrun_gpu_hybrid3_cosine/tables/metrics_comparison.csv`
- `outputs/fullrun_gpu_hybrid3_cosine_balanced/tables/metrics_comparison.csv`
- `outputs/fullrun_gpu_hybrid3_cosine_gpuheavy/tables/metrics_comparison.csv`
- `outputs/tables/profile_compare_fullrun_k20.csv`

Profil wydajności po przeniesieniu na GPU (8 marca 2026, fullrun):
- `Stage final.candidate_fusion`: ~`211s` -> ~`12.8s`
- profil `strict GPU` (features + fusion na GPU): pipeline ~`1289s` (najwyższy NDCG/MAP, największy latency)
- profil `balanced` (features na CPU + fusion na GPU): pipeline ~`1128s`
- profil `gpuheavy` (większe batch-e retrieval/Two-Tower): pipeline ~`1098s` (najszybszy end-to-end, jakość bardzo blisko strict)

Profile uruchomieniowe:
- `configs/fullrun_gpu_hybrid3_cosine.yaml` -> `strict GPU` (`features_use_gpu=true`, minimalizacja CPU kosztem czasu)
- `configs/fullrun_gpu_hybrid3_cosine_balanced.yaml` -> `balanced` (`features_use_gpu=false`)
- `configs/fullrun_gpu_hybrid3_cosine_gpuheavy.yaml` -> `gpuheavy` (większe `recommend_batch_size`, `two_tower.batch_size`, `query_batch_size`)
- `configs/fullrun_gpu_best.yaml` -> top-line (`baseline_weight=0.65`, `final_weight=0.35`)
- `configs/fullrun_gpu_robust.yaml` -> stability/CV (`baseline_weight=0.55`, `final_weight=0.45`)

Retrieval ablation (slice 1.2M, `final_ranker_ensemble` @20):
- `als_gpu`: Recall `0.1476`, NDCG `0.0574`, MAP `0.0329`
- `cosine`: Recall `0.1373`, NDCG `0.0530`, MAP `0.0299`
- `bm25`: Recall `0.1352`, NDCG `0.0520`, MAP `0.0294`

Wniosek: dla full-catalog pipeline na tym protokole ALS nadal wygrywa jako retrieval bazowy.
Porównanie zapisane w:
- `outputs/gpu/tables/retrieval_algo_compare_k20.csv`

3-way candidate fusion (ALS + Two-Tower + cosine ItemKNN, slice 1.2M):
- config: `configs/final_gpu_hybrid3_cosine.yaml` (`tt=0.40`, `als=0.50`, `aux=0.10`)
- `final_retrieval_two_tower_faiss_hybrid3` @20: Recall `0.1298`, NDCG `0.0499`, MAP `0.0283`
- `final_ranker_two_tower_hybrid3` @20: Recall `0.1488`, NDCG `0.0576`, MAP `0.0327`
- `final_ranker_ensemble` @20: Recall `0.1495`, NDCG `0.0579`, MAP `0.0331`

Delta 3-way vs 2-way (`outputs/gpu` -> `outputs/gpu_hybrid3_cosine`, @20):
- retrieval: Recall `+0.0055`, NDCG `+0.0020`, MAP `+0.0010`
- final ranker hybrid: Recall `+0.0046`, NDCG `+0.0020`, MAP `+0.0012`
- final ranker ensemble: Recall `+0.0019`, NDCG `+0.0005`, MAP `+0.0002`

Ensemble sweep (slice 1.2M, `configs/final_gpu_hybrid3_cosine.yaml`, @20):
- `baseline_weight=0.45` -> NDCG `0.057866`
- `baseline_weight=0.55` -> NDCG `0.058285` (best)
- `baseline_weight=0.65` -> NDCG `0.056595`
- `baseline_weight=0.75` -> NDCG `0.055996`

Wniosek: na slice 1.2M najlepsza kalibracja ensemble to `baseline=0.55`, `final=0.45` (w Time-CV fullrun wygrało `0.45/0.55`).

Artefakty:
- `outputs/gpu_hybrid3_cosine/tables/metrics_comparison.csv`
- `outputs/gpu_hybrid3_cosine/tables/fusion_sweep_summary.csv`
- `outputs/gpu_hybrid3_cosine/tables/ensemble_sweep_summary.csv`
- `outputs/gpu_hybrid3_cosine/tables/hybrid2_vs_hybrid3_k20.csv`

Hybrid3 vs historical runs (@20, run 8 marca 2026):
- `final_ranker_ensemble` (`hybrid3`) vs (`robust`): Recall `+0.002779`, NDCG `+0.001759`, MAP/MRR `+0.001426`
- `final_ranker_ensemble` (`hybrid3`) vs (`best`): Recall `+0.003045`, NDCG `+0.002040`, MAP/MRR `+0.001696`
- `final_ranker_two_tower_hybrid3` (`hybrid3`) vs (`robust two_tower`): Recall `+0.004982`, NDCG `+0.003431`, MAP/MRR `+0.002969`

Porównanie zapisane w:
- `outputs/fullrun_gpu_hybrid3_cosine/tables/hybrid3_vs_best_robust_k20.csv`

Benchmark protocol (leave-one-out + sampled negatives, bliżej raportowania z paperów):
```bash
python scripts/benchmark_sampled_eval.py --config configs/benchmark_sampled.yaml --output outputs/benchmark/tables/benchmark_sampled_metrics_20k_knn.csv
```

Benchmark result (8 marca 2026, 20k users, 100 negatives/user):
- `als_sampled`: Recall@20 `0.9676`, NDCG@10 `0.8036`, HitRate@10 `0.9560`
- `bm25_sampled`: Recall@20 `0.9936`, NDCG@10 `0.7723`, HitRate@10 `0.9747`
- `cosine_sampled`: Recall@20 `0.9968`, NDCG@10 `0.7898`, HitRate@10 `0.9801`
- `tfidf_sampled`: Recall@20 `0.9962`, NDCG@10 `0.7928`, HitRate@10 `0.9790`
- `hybrid_sampled`: Recall@20 `0.9831`, NDCG@10 `0.7963`, HitRate@10 `0.9673`
- `two_tower_sampled`: Recall@20 `0.9466`, NDCG@10 `0.6815`, HitRate@10 `0.8886`

Uwaga: to metryki w protokole sampled-negatives (`1 positive + 100 negatives`) i nie są 1:1 porównywalne z full-catalog runem z góry README.
Szybki wariant debug/smoke:
- `outputs/benchmark/tables/benchmark_sampled_metrics_quick.csv` (`max_users=3000`)

## Stability (Time CV)

Temporal CV (offsety `0,1,2`, metryki @20) z kalibracją wag ensemble na offsetach `1,2` wybrał:
- `baseline_weight=0.45`, `final_weight=0.55` (best mean NDCG na kalibracji)

Stability summary (`mean ± std`):
- `final_ranker_ensemble`: Recall `0.1603 ± 0.0113`, NDCG `0.0617 ± 0.0041`, MAP `0.0349 ± 0.0023`
- `final_ranker_two_tower_hybrid3`: Recall `0.1584 ± 0.0088`, NDCG `0.0611 ± 0.0032`, MAP `0.0347 ± 0.0017`
- `baseline_ranker`: Recall `0.1542 ± 0.0125`, NDCG `0.0589 ± 0.0048`, MAP `0.0330 ± 0.0028`

Artefakty stability:
- `outputs/gpu_hybrid3_cosine/tables/time_cv/calibration_summary.csv`
- `outputs/gpu_hybrid3_cosine/tables/time_cv/split_metrics_k.csv`
- `outputs/gpu_hybrid3_cosine/tables/time_cv/stability_summary.csv`

Nowe cechy (feature extension):
- time/context: `time_since_last_user_event`, `time_since_last_item_event`, `item_freshness_1d/7d`, `query_hour_sin/cos`, `query_dow_sin/cos`
- interaction: `user_item_mean_gap`, `user_item_mean_abs_gap`, `item_pop_per_user`, `user_activity_rate`
- session/genre: `user_recent_genre_overlap`, `user_recent_genre_coverage`, `item_genre_count` (bitmask + popcount, bez ciężkich joinów one-hot)

## Reproducible Commands

```bash
python scripts/download_data.py --output-dir data --local-source ../LENS/ml-25m
python scripts/build_features.py --config configs/baseline.yaml
python scripts/train_retrieval.py --config configs/baseline.yaml
python scripts/train_ranker.py --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml
python scripts/run_pipeline.py --config configs/final.yaml
./scripts/install_implicit_cuda.sh
python scripts/run_pipeline.py --config configs/final_gpu.yaml
python scripts/run_pipeline.py --config configs/final_gpu_cosine.yaml
python scripts/run_pipeline.py --config configs/final_gpu_bm25.yaml
python scripts/run_pipeline.py --config configs/final_gpu_hybrid3_cosine.yaml
python scripts/run_pipeline.py --config configs/final_gpu_hybrid3_cosine_balanced.yaml
python scripts/run_pipeline.py --config configs/final_gpu_hybrid3_cosine_gpuheavy.yaml
python scripts/run_pipeline.py --config configs/final_gpu_aggressive.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_best.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_robust.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_hybrid3_cosine.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_hybrid3_cosine_balanced.yaml
python scripts/run_pipeline.py --config configs/fullrun_gpu_hybrid3_cosine_gpuheavy.yaml
python scripts/compare_profiles.py --profile "name=strict,metrics=outputs/fullrun_gpu_hybrid3_cosine/tables/metrics_comparison.csv,log=outputs/fullrun_gpu_hybrid3_cosine/logs/pipeline_cudf_features_fullrun.log" --profile "name=balanced,metrics=outputs/fullrun_gpu_hybrid3_cosine_balanced/tables/metrics_comparison.csv,log=outputs/fullrun_gpu_hybrid3_cosine_balanced/logs/pipeline_refresh.log" --profile "name=gpuheavy,metrics=outputs/fullrun_gpu_hybrid3_cosine_gpuheavy/tables/metrics_comparison.csv,log=outputs/fullrun_gpu_hybrid3_cosine_gpuheavy/logs/pipeline.log" --output outputs/tables/profile_compare_fullrun_k20.csv
python scripts/benchmark_sampled_eval.py --config configs/benchmark_sampled.yaml --output outputs/benchmark/tables/benchmark_sampled_metrics_20k_knn.csv
python scripts/time_cv_stability.py --config configs/final_gpu_hybrid3_cosine.yaml --eval-offsets 0,1,2 --calib-offsets 1,2 --ensemble-baseline-weights 0.45,0.55,0.65 --target-k 20 --metric ndcg
python scripts/sweep_fusion_weights.py --config configs/final.yaml --two-tower-weights 0.35,0.45,0.55 --target-k 20 --metric ndcg
python scripts/sweep_fusion_weights.py --config configs/final_gpu_hybrid3_cosine.yaml --two-tower-weights 0.40,0.45 --aux-weights 0.05,0.10 --aux-algorithm cosine --target-k 20 --metric ndcg
python scripts/sweep_ensemble_weights.py --config configs/final.yaml --baseline-weights 0.45,0.55,0.65 --target-k 20 --metric ndcg
python scripts/sweep_ensemble_weights.py --config configs/final_gpu_hybrid3_cosine.yaml --baseline-weights 0.45,0.55,0.65,0.75 --target-k 20 --metric ndcg
```

## Outputs

- Tables:
  - `outputs/tables/metrics_baseline.csv`
  - `outputs/tables/metrics_final.csv`
  - `outputs/tables/metrics_comparison.csv`
  - `outputs/tables/fusion_sweep_summary.csv`
  - `outputs/tables/ensemble_sweep_summary.csv`
  - `outputs/tables/split_stats.csv`
  - `outputs/gpu/tables/profile_compare_k20.csv`
  - `outputs/gpu/tables/retrieval_algo_compare_k20.csv`
  - `outputs/gpu_hybrid3_cosine/tables/metrics_comparison.csv`
  - `outputs/gpu_hybrid3_cosine/tables/fusion_sweep_summary.csv`
  - `outputs/gpu_hybrid3_cosine/tables/ensemble_sweep_summary.csv`
  - `outputs/gpu_hybrid3_cosine/tables/hybrid2_vs_hybrid3_k20.csv`
  - `outputs/fullrun_gpu_best/tables/best_vs_robust_k20.csv`
  - `outputs/fullrun_gpu_hybrid3_cosine/tables/metrics_comparison.csv`
  - `outputs/fullrun_gpu_hybrid3_cosine/tables/hybrid3_vs_best_robust_k20.csv`
  - `outputs/fullrun_gpu_hybrid3_cosine_balanced/tables/metrics_comparison.csv`
  - `outputs/fullrun_gpu_hybrid3_cosine_gpuheavy/tables/metrics_comparison.csv`
  - `outputs/tables/profile_compare_fullrun_k20.csv`
  - `outputs/benchmark/tables/benchmark_sampled_metrics_20k_knn.csv`
  - `outputs/benchmark/tables/benchmark_sampled_metrics_20k.csv`
  - `outputs/benchmark/tables/benchmark_sampled_metrics_quick.csv`
- Figures:
  - `outputs/figures/metrics_comparison_kmax.png`
  - `outputs/figures/recall_at_k_curves.png`
  - `outputs/figures/ndcg_at_k_curves.png`

## Data Slice Used In This Run

`split_stats.csv` (current run):
- train interactions: 15304489
- val interactions: 161578
- test interactions: 161578
- users: 161578
- items: 47187

Konfiguracja (`configs/fullrun_gpu_hybrid3_cosine.yaml`) używa:
- `min_rating = 3.5`
- `max_rows = null` (pełny filtered set)

## Why This Works

Najlepszy model to `final_ranker_ensemble`, który łączy:
- 3-way candidate fusion (`two_tower + als + cosine`) na etapie retrieval,
- rank-level fusion `baseline_ranker` + `final_ranker_two_tower_hybrid3` (w `configs/fullrun_gpu_hybrid3_cosine.yaml`: `baseline_weight=0.45`, `final_weight=0.55`).

Dzięki temu top-line przebija poprzednie profile `best/robust` na Recall/NDCG/MAP/MRR przy pełnym `max_rows=null`.

## Testing

```bash
python -m pytest -q
```
