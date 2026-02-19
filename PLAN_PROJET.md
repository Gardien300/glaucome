# Plan du projet — Détection du glaucome référable (RG) sur fond d’œil

**Contexte :** Identification du glaucome référable (RG) à partir d’images de fond d’œil. **Périmètre actuel : machine learning classique uniquement** (aucun deep learning — CNN, ViT, etc. — pour le moment, qu’importe le résultat). Présentation du travail devant un groupe.

**Référence :** Challenge AIROGS (Artificial Intelligence for RObust Glaucoma Screening) — dataset et papiers des gagnants dans `research/`.

---

## 1. Synthèse du contexte (dataset & littérature)

### 1.1 Dataset disponible

| Élément | Détail |
|--------|--------|
| **Labels** | `train_labels.csv` : colonnes `challenge_id`, `class` (NRG / RG) |
| **Effectifs** | ~101k lignes (déséquilibre ~3 % RG) |
| **Images** | Réparties dans les dossiers `0/`, `1/`, …, `5/` (~101k images, format `TRAINXXXXXX.jpg`) |
| **Phase ML (cette partie)** | On n'utilise que le dossier `0/` (18k images) pour l'entraînement. Les dossiers 1–5 servent à l’évaluation (généralisation) en ML. Pas de deep learning dans ce périmètre. |

**À faire en premier :** construire un chargement qui ne lit que les images du dossier `0/` et associe chaque fichier à son label (ex. `data/train_ml_subset.csv` ou filtre au chargement).

### 1.2 Ce que montrent le challenge et les papiers (AIROGS)

- **Objectif clinique :** Dépistage du glaucome référable (RG) à partir de photographies du fond d’œil (CFP), avec une idée de **robustesse** (images de qualité variable, voire non gradables).
- **Métriques du challenge :**
  - **α** : pAUC (aire partielle sous la courbe ROC) pour le glaucome référable (souvent 90–100 % spécificité).
  - **β** : Sensibilité à 95 % de spécificité (SE@95SP).
  - **γ** : Kappa sur la « non-gradabilité ».
  - **δ** : AUC pour la détection d’images non gradables.
- **Pistes des meilleures équipes :**
  - **Classification RG :** Vision Transformers (ViT) utilisés par le top 3.
  - **Prétraitement :** Détection / segmentation du disque optique → recadrage autour du disque (plusieurs gagnants ont fait de l’annotation manuelle du disque).
  - **Robustesse / non-gradabilité :** Confiance de la détection du disque optique, ou classifieur dédié, ou erreur de reconstruction (autoencodeur) pondérée par l’incertitude du modèle RG.

Pour la **phase ML classique**, on ne reproduit pas tout le deep learning des gagnants, mais on s’en inspire (prétraitement, métriques, déséquilibre) et on pose les bases reproductibles et présentables.

---

## 2. Grandes étapes du projet (machine learning → présentable)

### Phase 0 — Environnement et reproductibilité

- **0.1** Environnement Python : venv / conda, dépendances (pandas, numpy, scikit-learn, opencv, pillow, matplotlib, seaborn).
- **0.2** Structure du repo : dossiers `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `research/`.
- **0.3** Fichier `requirements.txt` (ou `environment.yml`) et README court pour cloner + installer + lancer une première analyse.

**Livrable :** Projet clonable, environnement unique, première exécution sans erreur.

---

### Phase 1 — Données : chargement, cohérence, EDA

- **1.1** **Alignement images / labels**
  - Lister les `challenge_id` pour lesquels une image existe dans le dossier **`0/`** (seul dossier utilisé en phase ML).
  - Créer un DataFrame (ou CSV dérivé) **images du dossier 0 + label** (ex. `data/train_ml_subset.csv`). Voir `src/config.py` (`ML_IMAGE_FOLDER = "0"`).
  - Vérifier la répartition NRG/RG sur ce sous-ensemble (nombre et %).

- **1.2** **Exploration (EDA)**
  - Distribution des classes (barplot, proportions).
  - Qualité / taille des images (résolution, format, exemples visuels).
  - Quelques exemples par classe (NRG vs RG) pour la présentation.

- **1.3** **Découpage train / validation / test**
  - Stratifié sur la cible (RG/NRG), avec seed fixe pour reproductibilité.
  - Proportions typiques : 70 % train, 15 % validation, 15 % test (ou 80/10/10).

**Livrable :** Notebook EDA + jeu de données « propre » (chemins, labels, splits) documenté.

---

### Phase 2 — Prétraitement des images (pipeline reproductible)

- **2.1** **Chargement et redimensionnement**
  - Charger les images (OpenCV ou Pillow), redimensionnement uniforme (ex. 224×224 ou 299×299) pour alimenter des descripteurs ou un extracteur de features.

- **2.2** **Extraction de features pour le ML classique** (au choix, par ordre de priorité)
  - **Option A — Histogrammes et texture :** couleurs (HSV/RGB), LBP, Haralick, gradient.
  - **Option B — Features « prêtes à l’emploi » :** HOG, descripteurs SIFT/ORB (puis agrégation type Bag of Visual Words).
  - **Option C — (hors périmètre actuel)** Features par CNN pré-entraîné : non utilisées dans cette phase ; on reste sur descripteurs classiques (A, B) uniquement.

- **2.3** **Gestion du déséquilibre (RG rare)**
  - Suréchantillonnage (SMOTE sur les features, ou duplication ciblée), sous-échantillonnage, ou pondération des classes dans le classifieur.
  - Documenter le choix (ex. « SMOTE sur train uniquement ») pour la reproductibilité.

**Livrable :** Script(s) ou notebook de prétraitement + extraction de features, sauvegarde des matrices (features + labels) pour les splits.

---

### Phase 3 — Modèles de machine learning

- **3.1** **Modèles de base**
  - Régression logistique, Random Forest, XGBoost (ou LightGBM) sur les features extraites.
  - Entraînement sur le **train**, réglage des hyperparamètres sur la **validation** (grid search ou random search, métrique : AUC-ROC ou pAUC / sensibilité à 95 % spécificité).

- **3.2** **Métriques alignées avec le challenge**
  - Courbe ROC, AUC-ROC.
  - **Sensibilité à 95 % de spécificité** (SE@95SP).
  - **pAUC (90–100 % spécificité)** si possible (scikit-learn ou code dédié).
  - Matrice de confusion, précision, rappel, F1 (en gardant en tête le déséquilibre).

- **3.3** **Sélection du meilleur pipeline**
  - Comparer les modèles sur l’ensemble de **test** (une seule fois), avec intervalles de confiance ou bootstrap si pertinent pour la présentation.

**Livrable :** Notebook(s) d’entraînement et d’évaluation, modèle(s) sauvegardés (joblib/pickle), tableau récapitulatif des métriques.

---

### Phase 4 — Analyse des résultats et interprétabilité

- **4.1** **Visualisations pour la présentation**
  - Courbes ROC par modèle.
  - Comparaison SE@95SP et pAUC (tableau ou barres).
  - Exemples de bonnes / mauvaises prédictions (images + prédiction + probabilité).

- **4.2** **Interprétabilité (ML classique)**
  - Importance des variables (Random Forest, XGBoost).
  - Si Bag of Words / régions : quelles régions ou types de features contribuent le plus.

- **4.3** **Limites et biais**
  - Effet du déséquilibre, impact du sous-ensemble 18k vs 101k. Rester en ML classique ; pas de proposition deep learning dans le périmètre actuel.

**Livrable :** Rapport court (PDF ou notebook exporté) + slides ou notebook « présentation » avec figures et messages clés.

---

### Phase 5 — Préparation de la présentation

- **5.1** **Structure du pitch**
  - Contexte clinique (glaucome, enjeu du dépistage).
  - Dataset (AIROGS, NRG/RG, déséquilibre, sous-ensemble utilisé).
  - Méthode (features + modèles ML).
  - Résultats (métriques, courbes ROC, exemples).
  - Limites et perspectives (deep learning, robustesse, non-gradabilité).

- **5.2** **Démo ou reproductibilité**
  - Option : script ou notebook « de la donnée brute au score » (chargement → prétraitement → prédiction) pour montrer le pipeline en direct.

- **5.3** **Documentation**
  - README à jour (objectif, structure, comment lancer EDA, train, évaluation).
  - `PLAN_PROJET.md` (ce document) comme feuille de route.

**Livrable :** Support de présentation + démo reproductible + repo propre et documenté.

---

## 3. Ordre d’exécution recommandé

```text
Phase 0 (env + structure)
    → Phase 1 (données + EDA + splits)
        → Phase 2 (prétraitement + features)
            → Phase 3 (entraînement + évaluation)
                → Phase 4 (analyse + interprétabilité)
                    → Phase 5 (présentation + doc)
```

---

## 4. Fichiers et dossiers suggérés

```text
glaucome/
├── data/
│   ├── train_ml_subset.csv       # IDs du dossier 0 + labels (phase ML)
│   ├── splits/                  # indices ou listes train/val/test
│   └── features/                # matrices de features (optionnel)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_training_ml.ipynb
│   └── 04_evaluation_presentation.ipynb
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── evaluation_metrics.py
├── models/                      # modèles sauvegardés
├── reports/                     # figures, tableaux, rapport court
├── research/                    # papiers PDF (existant)
├── PLAN_PROJET.md               # ce plan
├── README.md
└── requirements.txt
```

---

## 5. Prochaines actions immédiates

1. **Créer** la structure de dossiers ci-dessus (sans écraser `research/` ni `0/`).
2. **Implémenter** le chargement des images présentes et la construction de `data/train_available.csv` (ou équivalent).
3. **Lancer** le premier notebook EDA (Phase 1) et documenter la répartition des classes sur les 18k images.
4. **Choisir** une première stratégie de features (Option A, B ou C) et un premier modèle (ex. Random Forest ou XGBoost) pour avoir une baseline rapidement.

**Périmètre :** tout le projet reste en **machine learning classique** (features manuelles ou HOG/texture, RF, XGBoost, régression logistique, SMOTE, seuils, etc.). Aucun deep learning (CNN, ViT, PyTorch/TensorFlow pour les modèles) pour le moment.
