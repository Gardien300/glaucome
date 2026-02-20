"""
Configuration pour la phase Machine Learning — projet glaucome (AIROGS).
"""

import os
from pathlib import Path

# Racine du projet (dossier glaucome)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stockage persistant RunPod : si RUNPOD_VOLUME_PATH est défini (ex. /runpod-volume),
# models et reports y sont sauvegardés (survit à l'arrêt du pod / manque de crédits)
_OUTPUT_BASE = os.environ.get("RUNPOD_VOLUME_PATH", "")
OUTPUT_BASE = Path(_OUTPUT_BASE).resolve() if _OUTPUT_BASE else PROJECT_ROOT

# Données
LABELS_CSV = PROJECT_ROOT / "train_labels.csv"
DATA_DIR = PROJECT_ROOT  # dossiers 0, 1, 2, 3, 4, 5 sont à la racine

# Phase ML : dossier d'images par défaut (rétrocompatible)
ML_IMAGE_FOLDER = "0"
ML_IMAGES_PATH = DATA_DIR / ML_IMAGE_FOLDER

# Multi-dossier : entraînement au max sur 0+1+2+3+4 (~90k), holdout strict sur 5
TRAIN_FOLDERS = ["0", "1", "2", "3", "4"]
EVAL_FOLDERS = []  # eval = val split (15 % de train_full) ; holdout 5 jamais vu
HOLDOUT_FOLDER = "5"

# Fichiers dérivés (créés par les scripts)
TRAIN_ML_SUBSET_CSV = PROJECT_ROOT / "data" / "train_ml_subset.csv"

# Cache features
FEATURE_CACHE_DIR = PROJECT_ROOT / "data" / "features_cache"

# Modèles et rapports (sur volume persistant si RUNPOD_VOLUME_PATH défini)
MODELS_DIR = OUTPUT_BASE / "models"
REPORTS_DIR = OUTPUT_BASE / "reports"

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
