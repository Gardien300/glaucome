# Détection du glaucome référable (RG) — Machine Learning uniquement

Projet basé sur le dataset **AIROGS**. **Périmètre : ML classique uniquement** (aucun deep learning — pas de CNN, ViT, PyTorch/Keras pour les modèles). Entraînement sur le **dossier 0** (18k images), évaluation possible sur dossiers 1–5.

## Structure

- `train_labels.csv` : labels (challenge_id, class NRG/RG)
- `0/` : images d'entraînement (18k) ; `1/` … `5/` : évaluation / généralisation
- `src/` : config, chargement, prétraitement, features, métriques
- `notebooks/` : EDA (`01_eda`), entraînement (`03_training_ml`), éval. dossier 1 (`04_eval_dossier_1`), comparaison features/modèles (`05_pipelines_features_ml`)
- `data/` : `train_ml_subset.csv` (généré), splits
- `PLAN_PROJET.md` : plan détaillé

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

## Générer le sous-ensemble ML (dossier 0)

La première fois, créer le CSV des images du dossier 0 :

```bash
python scripts/build_ml_subset.py
```

Ou lancer le notebook `01_eda.ipynb` : le chargement appelle `get_ml_subset_df()` qui crée le fichier si besoin (peut prendre 1–2 min).

## Lancer les notebooks

**Dans Cursor / VS Code (recommandé)** : ouvrir le dossier `glaucome` comme workspace, puis ouvrir `notebooks/01_eda.ipynb` et `notebooks/03_training_ml.ipynb`. Exécuter les cellules avec le bouton « Run » ou Shift+Entrée. Le répertoire de travail est la racine du projet ; les notebooks détectent automatiquement la racine.

**En Jupyter (navigateur)** : lancer `jupyter notebook notebooks/` depuis la racine du projet, puis exécuter `01_eda.ipynb` puis `03_training_ml.ipynb`.

## Features et options ML

- **Features :** `simple` (histo + stats), `hog`, `lbp`, `haralick`, `combined` (toutes). Voir `src/feature_extraction.py`.
- **Pipeline :** SMOTE, RF / régression logistique / XGBoost, réglage du seuil pour 95 % spécificité. Voir `src/pipeline_ml.py` et `05_pipelines_features_ml.ipynb`.

## Métriques

Alignées sur le challenge AIROGS : AUC-ROC, sensibilité à 95 % spécificité, pAUC (90–100 % spécificité).
