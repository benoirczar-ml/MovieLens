# GPU v2 Protocols

## Why two protocols
- `main` (full-catalog temporal holdout) keeps continuity with existing pipeline history.
- `benchmark` (sampled negatives) is the target track for higher ranking metrics.

## Current target bands (benchmark track)
- Recall@20: `0.22 - 0.24`
- NDCG@10: `0.14 - 0.16`
- NDCG@20: `0.17 - 0.19`

## Run modes
- Full + benchmark in one command:
```bash
python scripts/run_gpu_v2.py --config configs/gpu_v2_lightgcn.yaml
```
- Smoke:
```bash
python scripts/run_gpu_v2.py --config configs/gpu_v2_lightgcn_smoke.yaml
```

## Output layout
- CSV tables: `outputs/gpu_v2_lightgcn/tables/`
- Parquet tables: `outputs/gpu_v2_lightgcn/parquet/`

## Notes
- Parquet is enabled for `gpu_v2` configs (`outputs.save_parquet_tables=true`).
- `benchmark` run writes both CSV and Parquet when available.
