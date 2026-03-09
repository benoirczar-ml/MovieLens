#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT
from recsys_ml25m.data.io import copy_local_dataset, download_movielens_25m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download or copy MovieLens 25M dataset")
    p.add_argument("--output-dir", default=str(ROOT / "data"), help="Target dir for dataset")
    p.add_argument("--local-source", default=None, help="Optional local path to existing ml-25m directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    if args.local_source:
        target = copy_local_dataset(args.local_source, out)
    else:
        target = download_movielens_25m(out)
    print(target)


if __name__ == "__main__":
    main()
