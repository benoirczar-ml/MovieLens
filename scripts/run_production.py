#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from _bootstrap import ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run production profile (strict quality)")
    p.add_argument("--config", default=str(ROOT / "configs" / "production.yaml"))
    p.add_argument(
        "--log-file",
        default=None,
        help="Optional explicit log path. If omitted, uses outputs/production_strict/logs/pipeline_<timestamp>.log",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_file or str(ROOT / "outputs" / "production_strict" / "logs" / f"pipeline_{ts}.log")
    # Delegate to the existing pipeline runner to keep behavior identical.
    cmd = ["python", "scripts/run_pipeline.py", "--config", args.config, "--log-file", log_path]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
