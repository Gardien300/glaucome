"""
Chargement des données pour la phase ML — mono ou multi-dossier.
Construit le sous-ensemble (IDs + labels) et permet de charger les chemins pour train/val/test.
"""

from pathlib import Path
import os
import pandas as pd

from .config import (
    PROJECT_ROOT,
    LABELS_CSV,
    DATA_DIR,
    ML_IMAGES_PATH,
    TRAIN_ML_SUBSET_CSV,
    TRAIN_FOLDERS,
    EVAL_FOLDERS,
    HOLDOUT_FOLDER,
    RANDOM_STATE,
)


def get_ml_subset_df(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Retourne le DataFrame du sous-ensemble ML (dossier 0 uniquement).
    Colonnes : challenge_id, class, path (chemin vers l'image).
    Crée data/train_ml_subset.csv si absent ou si force_rebuild=True.
    """
    # Utilise le CSV mis en cache si disponible pour éviter de relister les fichiers.
    if TRAIN_ML_SUBSET_CSV.exists() and not force_rebuild:
        df = pd.read_csv(TRAIN_ML_SUBSET_CSV)
        return df

    image_dir = Path(ML_IMAGES_PATH)
    if not image_dir.exists():
        raise FileNotFoundError(f"Dossier images ML introuvable: {image_dir}")

    available_ids = set()
    for name in os.listdir(image_dir):
        if name.lower().endswith(".jpg"):
            available_ids.add(Path(name).stem)

    labels = pd.read_csv(LABELS_CSV)
    subset = labels[labels["challenge_id"].isin(available_ids)].copy()
    subset["path"] = str(ML_IMAGES_PATH.name) + "/" + subset["challenge_id"] + ".jpg"

    TRAIN_ML_SUBSET_CSV.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(TRAIN_ML_SUBSET_CSV, index=False)
    return subset


def sample_df(df: pd.DataFrame, max_samples: int, random_state: int | None = None) -> pd.DataFrame:
    """
    Sous-échantillonne un DataFrame à max_samples lignes au maximum.

    - Si len(df) <= max_samples, retourne df tel quel.
    - Sinon, effectue un tirage aléatoire sans remise.

    Utile pour limiter la taille des jeux utilisés dans les notebooks (expériences rapides).
    """
    if max_samples is None or max_samples <= 0:
        return df
    if len(df) <= max_samples:
        return df
    rs = random_state if random_state is not None else RANDOM_STATE
    return df.sample(n=max_samples, random_state=rs)


def get_subset_for_folder(folder: str) -> pd.DataFrame:
    """
    Retourne un DataFrame (challenge_id, class, path) pour les images
    présentes dans le dossier donné ("0", "1", "2", etc.).
    """
    image_dir = DATA_DIR / folder
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Dossier introuvable: {image_dir}")

    available_ids = set()
    for name in os.listdir(image_dir):
        if name.lower().endswith(".jpg"):
            available_ids.add(Path(name).stem)

    labels = pd.read_csv(LABELS_CSV)
    subset = labels[labels["challenge_id"].isin(available_ids)].copy()
    subset["path"] = folder + "/" + subset["challenge_id"] + ".jpg"
    return subset


def get_multi_folder_df(folders: list = None) -> pd.DataFrame:
    """
    Concatène les DataFrames de plusieurs dossiers.
    Par défaut, utilise TRAIN_FOLDERS depuis config.py.
    Retourne un DataFrame (challenge_id, class, path).
    """
    if folders is None:
        folders = TRAIN_FOLDERS

    dfs = []
    for folder in folders:
        try:
            df = get_subset_for_folder(folder)
            dfs.append(df)
            n_rg = (df["class"] == "RG").sum()
            n_total = len(df)
            print(f"  Dossier {folder}: {n_total} images ({n_rg} RG, {n_total - n_rg} NRG)")
        except FileNotFoundError as e:
            print(f"  Dossier {folder}: IGNORÉ ({e})")
            continue

    if not dfs:
        raise ValueError(f"Aucun dossier valide parmi {folders}")

    combined = pd.concat(dfs, ignore_index=True)
    n_rg = (combined["class"] == "RG").sum()
    print(f"  Total: {len(combined)} images ({n_rg} RG, {len(combined) - n_rg} NRG)")
    return combined


def get_train_val_test_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Découpe stratifié en train / val / test.
    Retourne (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    y = df["class"]

    df_trainval, df_test = train_test_split(
        df, test_size=test_ratio, stratify=y, random_state=RANDOM_STATE
    )
    val_size = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size,
        stratify=df_trainval["class"],
        random_state=RANDOM_STATE,
    )
    return df_train, df_val, df_test


def get_multi_folder_splits() -> tuple:
    """
    Charge les données multi-dossier avec la stratégie optimale :
    - Train : dossiers 0+1+2 (split stratifié 85/15 train/val)
    - Eval : dossiers 3+4
    - Test final : dossier 5

    Retourne (train_df, val_df, eval_df, holdout_df).
    """
    from sklearn.model_selection import train_test_split

    print("Chargement multi-dossier:")
    train_full = get_multi_folder_df(TRAIN_FOLDERS)

    # Split train/val (85/15)
    train_df, val_df = train_test_split(
        train_full,
        test_size=0.15,
        stratify=train_full["class"],
        random_state=RANDOM_STATE,
    )
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    # Eval
    print("Chargement évaluation:")
    eval_df = get_multi_folder_df(EVAL_FOLDERS)

    # Holdout
    print("Chargement holdout:")
    holdout_df = get_subset_for_folder(HOLDOUT_FOLDER)
    n_rg = (holdout_df["class"] == "RG").sum()
    print(f"  Dossier {HOLDOUT_FOLDER}: {len(holdout_df)} images ({n_rg} RG)")

    return train_df, val_df, eval_df, holdout_df
