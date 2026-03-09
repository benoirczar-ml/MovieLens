from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens_25m(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = output_dir / "ml-25m"
    if (target / "ratings.csv").exists():
        return target

    zip_path = output_dir / "ml-25m.zip"
    urlretrieve(DATASET_URL, zip_path)  # nosec B310

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    zip_path.unlink(missing_ok=True)
    return target


def load_ratings(data_dir: str | Path, min_rating: float = 0.0, max_rows: int | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir)
    ratings_path = data_dir / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

    usecols = ["userId", "movieId", "rating", "timestamp"]
    df = pd.read_csv(
        ratings_path,
        usecols=usecols,
        nrows=max_rows,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    if min_rating > 0:
        df = df[df["rating"] >= float(min_rating)].copy()

    df = df.sort_values(["userId", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def temporal_leave_last_split(
    ratings: pd.DataFrame,
    val_k: int = 1,
    test_k: int = 1,
    min_user_interactions: int = 5,
    split_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not {"userId", "movieId", "timestamp"}.issubset(ratings.columns):
        raise ValueError("ratings must have userId, movieId, timestamp columns")
    if split_offset < 0:
        raise ValueError("split_offset must be >= 0")

    df = ratings.sort_values(["userId", "timestamp"], kind="mergesort").copy()
    user_sizes = df.groupby("userId").size()
    needed = split_offset + val_k + test_k + 1
    valid_users = user_sizes[user_sizes >= max(min_user_interactions, needed)].index
    df = df[df["userId"].isin(valid_users)].copy()

    if df.empty:
        raise ValueError("No users left after min_user_interactions filtering")

    ord_desc = df.groupby("userId").cumcount(ascending=False)

    test_lo = split_offset
    test_hi = split_offset + test_k
    val_lo = test_hi
    val_hi = val_lo + val_k

    test_mask = (ord_desc >= test_lo) & (ord_desc < test_hi)
    val_mask = (ord_desc >= val_lo) & (ord_desc < val_hi)
    # Train only on interactions strictly older than val/test window.
    train_mask = ord_desc >= val_hi

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    return train_df, val_df, test_df


def copy_local_dataset(src_dir: str | Path, dst_dir: str | Path) -> Path:
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / src.name
    if target.exists():
        return target
    shutil.copytree(src, target)
    return target
