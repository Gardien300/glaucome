"""
Data loading utilities for the deep learning phase (PyTorch).

This module builds on top of src.data_loading and src.config:
- Uses train_labels.csv and the 0–5 folders.
- Provides PyTorch Dataset / DataLoader objects.
- Encodes labels as 0 (NRG) / 1 (RG).
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
from PIL import Image

from .config import (
    PROJECT_ROOT,
    TRAIN_FOLDERS,
    EVAL_FOLDERS,
    HOLDOUT_FOLDER,
    RANDOM_STATE,
)
from .data_loading import get_multi_folder_df, get_subset_for_folder


try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:  # pragma: no cover - only raised if torch missing
    raise ImportError(
        "PyTorch is required for the deep learning data pipeline "
        "(pip install torch torchvision)."
    ) from e


LABEL_MAP = {"NRG": 0, "RG": 1}


class FundusDataset(Dataset):
    """
    Basic PyTorch Dataset for fundus images.

    Expects a DataFrame with at least:
    - 'path': relative path from PROJECT_ROOT to the image file
    - 'class': 'NRG' or 'RG'
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Optional[Callable] = None,
        root: Path = PROJECT_ROOT,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = Path(row["path"])
        img_path = self.root / rel_path

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        label_str = row["class"]
        label = LABEL_MAP.get(label_str, 0)
        return img, label


def _split_train_val(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val split on the 'class' column.
    """
    from sklearn.model_selection import train_test_split

    y = df["class"]
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=y,
        random_state=random_state,
    )
    return train_df, val_df


def build_dl_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build global splits for deep learning, aligned with AIROGS-style usage:

    - Train: folders in TRAIN_FOLDERS (0+1+2 by default), with internal train/val split.
    - Eval: folders in EVAL_FOLDERS (3+4 by default).
    - Holdout: folder HOLDOUT_FOLDER (5 by default), never seen during tuning.

    Returns:
        train_df, val_df, eval_df, holdout_df
    """
    print("Building deep learning splits (multi-folder)...")
    train_full = get_multi_folder_df(TRAIN_FOLDERS)
    train_df, val_df = _split_train_val(train_full, val_ratio=0.15, random_state=RANDOM_STATE)

    print(f"  Train DL: {len(train_df)} | Val DL: {len(val_df)}")

    if EVAL_FOLDERS:
        print("  Loading eval (EVAL_FOLDERS)...")
        eval_df = get_multi_folder_df(EVAL_FOLDERS)
    else:
        print("  Eval = Val (EVAL_FOLDERS vide, train max sur 0+1+2+3+4)")
        eval_df = val_df.copy()

    print("  Loading holdout (HOLDOUT_FOLDER)...")
    holdout_df = get_subset_for_folder(HOLDOUT_FOLDER)
    n_rg = (holdout_df["class"] == "RG").sum()
    print(f"  Holdout folder {HOLDOUT_FOLDER}: {len(holdout_df)} images ({n_rg} RG)")

    return train_df, val_df, eval_df, holdout_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    holdout_df: Optional[pd.DataFrame] = None,
    train_transforms: Optional[Callable] = None,
    eval_transforms: Optional[Callable] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create DataLoader objects for the different splits.

    Args:
        train_df, val_df, eval_df, holdout_df: DataFrames as returned by build_dl_splits().
        train_transforms: transforms/augmentations for training.
        eval_transforms: transforms for validation/test/eval.
        batch_size: batch size for all loaders.
        num_workers: DataLoader workers.
        pin_memory: whether to pin memory (recommended for GPU training).

    Returns:
        dict with keys 'train', 'val', and optionally 'eval', 'holdout'.
    """
    if eval_transforms is None:
        eval_transforms = train_transforms

    loader_kw = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        if persistent_workers:
            loader_kw["persistent_workers"] = True
        if prefetch_factor is not None:
            loader_kw["prefetch_factor"] = prefetch_factor

    loaders: Dict[str, DataLoader] = {}

    train_ds = FundusDataset(train_df, transforms=train_transforms)
    val_ds = FundusDataset(val_df, transforms=eval_transforms)

    loaders["train"] = DataLoader(train_ds, shuffle=True, **loader_kw)
    loaders["val"] = DataLoader(val_ds, shuffle=False, **loader_kw)

    if eval_df is not None:
        eval_ds = FundusDataset(eval_df, transforms=eval_transforms)
        loaders["eval"] = DataLoader(eval_ds, shuffle=False, **loader_kw)

    if holdout_df is not None:
        holdout_ds = FundusDataset(holdout_df, transforms=eval_transforms)
        loaders["holdout"] = DataLoader(holdout_ds, shuffle=False, **loader_kw)

    return loaders


