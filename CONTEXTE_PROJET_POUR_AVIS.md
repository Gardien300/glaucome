# Contexte projet glaucome — résumé pour avis externe (Claude)

**Usage :** coller ce fichier à Claude (ou autre LLM) pour demander un avis critique sur le projet.

---

## 1. Objectif

Détection du **glaucome référable (RG)** à partir de photographies du fond d’œil (dataset **AIROGS**). Objectif final : **produit commercialisable** (SaMD) et **ultra efficace pour le patient** (ne pas manquer les RG, limiter les faux positifs, scores calibrés).

---

## 2. Données

- **Source :** `train_labels.csv` (~101k lignes), colonnes `challenge_id`, `class` (NRG / RG). Images dans dossiers `0/` à `5/` à la racine du projet.
- **Déséquilibre :** ~96,5 % NRG, ~3,5 % RG.
- **Splits :**
  - **ML classique :** sous-ensemble dossier `0` (~18k images), `data/train_ml_subset.csv`.
  - **DL :** train = dossiers **0+1+2** (~45 900 train, ~8 100 val), eval = **3+4** (~36k), holdout = **5** (~11 442). Séparation par dossier pour simuler multi-sites / généralisation.

---

## 3. Pipeline ML classique (phase 1)

- **Config :** `src/config.py` (PROJECT_ROOT, TRAIN_FOLDERS, EVAL_FOLDERS, HOLDOUT_FOLDER, RANDOM_STATE=42).
- **Chargement :** `src/data_loading.py`, `src/build_ml_subset.py` → CSV sous-ensemble dossier 0.
- **Features :** `src/feature_extraction.py` (LBP, Haralick, HOG, canaux, texture), prétraitement (CLAHE, canal vert), cache dans `data/features_cache`.
- **Modèles :** `src/pipeline_ml.py` — SMOTE dans CV, StandardScaler, XGBoost / RF / LR, stacking, CalibratedClassifierCV (sklearn).
- **Tuning :** `src/tuning.py` (RandomizedSearchCV), `src/feature_selection.py` (RFE, PCA optionnel).
- **Métriques :** `src/evaluation_metrics.py` — AUC-ROC, pAUC (90–100 % spec), Sens@95 % spec, Sens@90 % spec, bootstrap CI, courbes ROC.
- **Notebooks :** EDA (01), baseline (03), eval cross-dossier (04), labo features (05), pipeline complet (07).
- **Rapports :** `RAPPORT_ML_CLASSIQUE_GLAUCOME.txt`, `report_ml_business.tex`, `REPONSE_CRITIQUE_PROFESSEUR.md`.

---

## 4. Pipeline DL (phase 2)

- **Data :** `src/dl_data.py` — `FundusDataset`, `build_dl_splits()` (train/val depuis 0+1+2, eval 3+4, holdout 5), `create_dataloaders()`.
- **Modèles :** `src/dl_models.py` — torchvision backbones, tête binaire (1 logit), BCEWithLogitsLoss. Backbones : `resnet18`, `resnet50`, `efficientnet_b0`, `efficientnet_b4`, `efficientnet_v2_s`, `efficientnet_v2_m`, `convnext_tiny`, `convnext_small`, `vit_b_16`. Défaut : `efficientnet_b0`.
- **Entraînement :** `src/dl_train.py` — config YAML, AdamW, early stopping sur AUC val (patience 5 ou 7), **temperature scaling** sur le set de validation après entraînement ; évaluation sur eval et holdout avec probas calibrées.
- **Checkpoint :** `models/dl_<backbone>_best.pth` = `{state_dict, temperature, backbone}` pour inférence calibrée.
- **Configs :** `configs/dl_baseline.yaml` (efficientnet_b0, 224, batch 32, calibration on), `configs/dl_highperf.yaml` (efficientnet_v2_m, 384, batch 24, 40 epochs, patience 7).
- **Déploiement :** `Dockerfile.dl` pour entraînement sur pod GPU (ex. RunPod) ; commande : `python -m src.dl_train --config configs/dl_baseline.yaml`.

---

## 5. Cibles produit & vision

- **Fichier :** `PRODUCT_VISION.md` — efficacité patient (Sens@95 %, pAUC, AUC, calibration), commercialisation (SaMD, CE/FDA, validation clinique, intended use).
- **Cibles chiffrées :** `configs/product_targets.yaml` — AUC ≥ 0,90, pAUC ≥ 0,80, Sens@95 % ≥ 0,75 (idéaux : 0,93 / 0,85 / 0,80). À atteindre sur **eval** et **holdout**.
- **Vérification :** en fin de `dl_train`, chargement de `product_targets.yaml` et affichage OK / EN DESSOUS pour chaque métrique (eval + holdout).

---

## 6. Choix modèles CNN

- **Doc :** `RECO_MODELES_CNN.md` — état de l’art (EfficientNet, EfficientNetV2, ConvNeXt, ViT), pertinence fond d’œil, recommandation par défaut = EfficientNet-B0 ; config “haute perf” = EfficientNetV2-M en 384. Données (~46k train) jugées suffisantes pour modèles plus lourds.

---

## 7. Arborescence clé

```
glaucome/
├── configs/
│   ├── dl_baseline.yaml      # efficientnet_b0, 224, calibration
│   ├── dl_highperf.yaml      # efficientnet_v2_m, 384
│   └── product_targets.yaml  # seuils min/idéal (AUC, pAUC, Sens@95%)
├── data/
│   ├── train_ml_subset.csv   # sous-ensemble ML (dossier 0)
│   └── features_cache/
├── models/                   # checkpoints DL (state_dict + T + backbone)
├── notebooks/                # 01_eda, 03_training_ml, 04_eval_dossier_1, 05, 07_full_pipeline
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── dl_data.py, dl_models.py, dl_train.py
│   ├── evaluation_metrics.py
│   ├── feature_extraction.py, feature_selection.py
│   ├── pipeline_ml.py, tuning.py
│   └── preprocessing*.py, features_spatial.py
├── train_labels.csv          # labels AIROGS
├── 0/ .. 5/                  # images par dossier
├── PLAN_PROJET.md
├── PRODUCT_VISION.md
├── RECO_MODELES_CNN.md
├── RAPPORT_ML_CLASSIQUE_GLAUCOME.txt
├── requirements.txt
└── Dockerfile.dl
```

---

## 8. Points déjà discutés / critiques reçues

- **SMOTE :** utilisé dans le CV (pipeline ML), pas en test ; réponse dans `REPONSE_CRITIQUE_PROFESSEUR.md`.
- **Feature importance / SHAP :** partiellement fait en ML ; pas encore en DL.
- **Généralisation :** éval cross-dossier (ML) et splits 0+1+2 / 3+4 / 5 (DL) pour limiter le sur-ajustement.
- **Calibration :** temperature scaling (DL) ; CalibratedClassifierCV (ML).

---

## 9. Question pour Claude

**En te basant sur ce résumé :** quels sont les **points forts**, les **risques ou manques** (technique, clinique, réglementaire, produit), et les **3–5 recommandations prioritaires** pour que ce projet tienne la route comme base d’un **produit commercialisable et ultra efficace pour le patient** (dépistage du glaucome référable) ?
