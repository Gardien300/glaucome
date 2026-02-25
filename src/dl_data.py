"""
Data loading utilities for the deep learning phase (PyTorch).

This module builds on top of src.data_loading and src.config:
- Uses train_labels.csv and the 0–5 folders.
- Provides PyTorch Dataset / DataLoader objects.
- Encodes labels as 0 (NRG) / 1 (RG).
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    PROJECT_ROOT,
    TRAIN_FOLDERS,
    EVAL_FOLDERS,
    HOLDOUT_FOLDER,
    RANDOM_STATE,
    SPLIT_STRATEGY,
    RANDOM_SPLIT_RATIOS,
    ALL_DL_FOLDERS,
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


# ---------------------------------------------------------------------------
# Medical normalisation (Bloc A — shared between training and inference)
# ---------------------------------------------------------------------------

def normalize_fundus(pil_image: Image.Image, clahe_clip: float = 2.0) -> Image.Image:
    """
    Standard pre-processing applied to every fundus image.

    Steps:
    1. Convert to RGB.
    2. Center-crop to a square (remove non-standard aspect ratio borders).
    3. CLAHE on the L-channel in LAB space (soft contrast correction).

    This function is used identically in training and inference to avoid
    domain mismatch.
    """
    img = pil_image.convert("RGB")

    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))

    np_img = np.array(img)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge((l_eq, a_ch, b_ch))
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return Image.fromarray(rgb_eq)


class FundusDataset(Dataset):
    """
    PyTorch Dataset for fundus images with optional medical pre-processing
    and dual-branch (global + papilla crop) support.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Optional[Callable] = None,
        root: Path = PROJECT_ROOT,
        apply_normalize: bool = True,
        dual_branch: bool = False,
        papilla_transforms: Optional[Callable] = None,
        papilla_crop_ratio: float = 0.22,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.root = Path(root)
        self.apply_normalize = apply_normalize
        self.dual_branch = dual_branch
        self.papilla_transforms = papilla_transforms
        self.papilla_crop_ratio = papilla_crop_ratio

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = Path(row["path"])
        img_path = self.root / rel_path

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.apply_normalize:
            img = normalize_fundus(img)

        if self.dual_branch:
            from .dl_models import detect_optic_disc_center, crop_optic_disc

            cx, cy = detect_optic_disc_center(img)
            papilla_crop = crop_optic_disc(img, cx, cy, self.papilla_crop_ratio)

            global_img = self.transforms(img) if self.transforms else img
            papilla_img = (
                self.papilla_transforms(papilla_crop)
                if self.papilla_transforms
                else (self.transforms(papilla_crop) if self.transforms else papilla_crop)
            )

            label = LABEL_MAP.get(row["class"], 0)
            return global_img, papilla_img, label

        if self.transforms is not None:
            img = self.transforms(img)

        label = LABEL_MAP.get(row["class"], 0)
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


def _build_dl_splits_random() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge tous les dossiers 0-5 et split stratifié aléatoire.
    Évite le biais d'un holdout venant d'une seule machine/site.
    """
    from sklearn.model_selection import train_test_split

    print("Building deep learning splits (RANDOM: merge 0-5 + stratified split)...")
    all_df = get_multi_folder_df(ALL_DL_FOLDERS)
    n_total = len(all_df)
    n_rg = (all_df["class"] == "RG").sum()
    print(f"  Total: {n_total} images ({n_rg} RG)")

    train_ratio, val_ratio, holdout_ratio = RANDOM_SPLIT_RATIOS
    assert abs(train_ratio + val_ratio + holdout_ratio - 1.0) < 1e-6

    # 1) Séparer holdout
    df_trainval, holdout_df = train_test_split(
        all_df,
        test_size=holdout_ratio,
        stratify=all_df["class"],
        random_state=RANDOM_STATE,
    )
    # 2) Séparer train / val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        df_trainval,
        test_size=val_size,
        stratify=df_trainval["class"],
        random_state=RANDOM_STATE,
    )

    n_train = len(train_df)
    n_val = len(val_df)
    n_hold = len(holdout_df)
    print(f"  Train: {n_train} | Val: {n_val} | Holdout: {n_hold}")

    eval_df = val_df.copy()
    return train_df, val_df, eval_df, holdout_df


def _build_dl_splits_folders() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split par dossiers (stratégie AIROGS originale)."""
    print("Building deep learning splits (FOLDERS)...")
    train_full = get_multi_folder_df(TRAIN_FOLDERS)
    train_df, val_df = _split_train_val(train_full, val_ratio=0.15, random_state=RANDOM_STATE)

    print(f"  Train DL: {len(train_df)} | Val DL: {len(val_df)}")

    if EVAL_FOLDERS:
        print("  Loading eval (EVAL_FOLDERS)...")
        eval_df = get_multi_folder_df(EVAL_FOLDERS)
    else:
        print("  Eval = Val")
        eval_df = val_df.copy()

    print("  Loading holdout (HOLDOUT_FOLDER)...")
    holdout_df = get_subset_for_folder(HOLDOUT_FOLDER)
    n_rg = (holdout_df["class"] == "RG").sum()
    print(f"  Holdout folder {HOLDOUT_FOLDER}: {len(holdout_df)} images ({n_rg} RG)")

    return train_df, val_df, eval_df, holdout_df


def build_dl_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build global splits for deep learning.

    Stratégie contrôlée par SPLIT_STRATEGY dans config:
    - "random": merge 0-5 + split stratifié (évite biais machine unique)
    - "folders": split par dossier AIROGS (train 0-4, holdout 5)

    Returns:
        train_df, val_df, eval_df, holdout_df
    """
    if SPLIT_STRATEGY == "random":
        return _build_dl_splits_random()
    return _build_dl_splits_folders()


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
    dual_branch: bool = False,
    papilla_transforms: Optional[Callable] = None,
    papilla_crop_ratio: float = 0.22,
) -> Dict[str, DataLoader]:
    """Create DataLoader objects for the different splits."""
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

    ds_kw = dict(dual_branch=dual_branch, papilla_crop_ratio=papilla_crop_ratio)

    loaders: Dict[str, DataLoader] = {}

    train_ds = FundusDataset(
        train_df, transforms=train_transforms,
        papilla_transforms=papilla_transforms, **ds_kw,
    )
    val_ds = FundusDataset(
        val_df, transforms=eval_transforms,
        papilla_transforms=papilla_transforms if dual_branch else None, **ds_kw,
    )

    loaders["train"] = DataLoader(train_ds, shuffle=True, **loader_kw)
    loaders["val"] = DataLoader(val_ds, shuffle=False, **loader_kw)

    if eval_df is not None:
        eval_ds = FundusDataset(
            eval_df, transforms=eval_transforms,
            papilla_transforms=papilla_transforms if dual_branch else None, **ds_kw,
        )
        loaders["eval"] = DataLoader(eval_ds, shuffle=False, **loader_kw)

    if holdout_df is not None:
        holdout_ds = FundusDataset(
            holdout_df, transforms=eval_transforms,
            papilla_transforms=papilla_transforms if dual_branch else None, **ds_kw,
        )
        loaders["holdout"] = DataLoader(holdout_ds, shuffle=False, **loader_kw)

    return loaders


