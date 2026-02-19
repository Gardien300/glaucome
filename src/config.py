"""
Configuration pour la phase Machine Learning — projet glaucome (AIROGS).
"""

from pathlib import Path

# Racine du projet (dossier glaucome)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Données
LABELS_CSV = PROJECT_ROOT / "train_labels.csv"
DATA_DIR = PROJECT_ROOT  # dossiers 0, 1, 2, 3, 4, 5 sont à la racine

# Phase ML : dossier d'images par défaut (rétrocompatible)
ML_IMAGE_FOLDER = "0"
ML_IMAGES_PATH = DATA_DIR / ML_IMAGE_FOLDER

# Multi-dossier : entraînement sur 0+1+2, éval sur 3+4, test final sur 5
TRAIN_FOLDERS = ["0", "1", "2"]
EVAL_FOLDERS = ["3", "4"]
HOLDOUT_FOLDER = "5"

# Fichiers dérivés (créés par les scripts)
TRAIN_ML_SUBSET_CSV = PROJECT_ROOT / "data" / "train_ml_subset.csv"

# Cache features
FEATURE_CACHE_DIR = PROJECT_ROOT / "data" / "features_cache"

# Modèles sauvegardés
MODELS_DIR = PROJECT_ROOT / "models"

# Prétraitement
IMAGE_SIZE_GLOBAL = (224, 224)
IMAGE_SIZE_TEXTURE = (512, 512)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Entraînement
CV_FOLDS = 5
N_TUNING_ITER = 50

# Reproductibilité
RANDOM_STATE = 42
