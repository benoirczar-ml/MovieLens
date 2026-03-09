#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import ROOT
from recsys_ml25m.config import load_config
from recsys_ml25m.pipeline import run_pipeline


class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full recommender pipeline")
    p.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    p.add_argument("--log-file", default=None, help="Optional path to tee stdout/stderr logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_fh = None
    try:
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "a", encoding="utf-8")
            sys.stdout = _TeeIO(old_stdout, log_fh)
            sys.stderr = _TeeIO(old_stderr, log_fh)
            print(f"log_file={log_path}")

        outputs = run_pipeline(cfg)
        summary = outputs["comparison"]
        out_path = Path(cfg.get("outputs", {}).get("tables_dir", ROOT / "outputs" / "tables")) / "metrics_comparison.csv"
        print(f"saved={out_path}")
        print(summary.to_string(index=False))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if log_fh is not None:
            log_fh.flush()
            log_fh.close()


if __name__ == "__main__":
    main()
