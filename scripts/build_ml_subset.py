#!/usr/bin/env python3
"""
Génère data/train_ml_subset.csv à partir du dossier 0 et de train_labels.csv.
À lancer depuis la racine du projet : python scripts/build_ml_subset.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import get_ml_subset_df

if __name__ == "__main__":
    df = get_ml_subset_df(force_rebuild=True)
    print("Écrit:", df.shape[0], "lignes dans data/train_ml_subset.csv")
    print(df["class"].value_counts())
