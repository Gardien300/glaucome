# Périmètre du projet — ML uniquement

**Règle :** tout le travail sur ce projet reste en **machine learning classique**. Aucun deep learning (CNN, ViT, réseaux de neurones, PyTorch/TensorFlow/Keras pour les modèles), qu’importe le résultat.

## Autorisé

- **Données :** chargement d’images (OpenCV, Pillow), redimensionnement, splits stratifiés.
- **Features :** histogrammes couleur, statistiques (moyenne, écart-type), HOG, LBP (scikit-image), Haralick (GLCM), combinaison `combined` (simple+HOG+LBP+Haralick), SIFT/ORB + Bag of Words.
- **Modèles :** Random Forest, XGBoost, LightGBM, régression logistique, SVM (scikit-learn / xgboost / imblearn).
- **Déséquilibre :** `class_weight`, SMOTE (imbalanced-learn), sous/sur-échantillonnage, réglage du seuil de décision (seuil optimisé pour 95 % spécificité).
- **Pipeline :** `src/pipeline_ml.py` — SMOTE, `get_classifier`, `find_threshold_for_specificity`, `threshold_predict`.
- **Évaluation :** AUC-ROC, pAUC, sensibilité à 95 % spécificité, matrice de confusion, courbes ROC.
- **Outils :** pandas, numpy, scikit-learn, scikit-image, OpenCV, imbalanced-learn, xgboost, joblib, matplotlib, seaborn.

## Hors périmètre (pour l’instant)

- CNN, Vision Transformer, tout réseau de neurones.
- PyTorch, TensorFlow, Keras pour l’entraînement de modèles.
- Utilisation de modèles pré-entraînés (ResNet, VGG, etc.) comme extracteurs de features ou classifieurs.

Ce fichier et `.cursor/rules/glaucome-ml-only.mdc` rappellent ce périmètre pour toutes les évolutions du projet.
