# Choix d’architecture CNN pour la détection du glaucome (fond d’œil)

Synthèse des modèles actuels et recommandation pour le projet (AIROGS, classification RG/NRG).

---

## 1. État de l’art (2024–2025)

### Architectures souvent citées

| Famille | Exemples | Points forts |
|--------|----------|--------------|
| **EfficientNet** | B0, B3, B4 | Très utilisés en imagerie rétinienne ; bon compromis précision / coût. |
| **EfficientNetV2** | V2-S, V2-M | Entraînement plus rapide, bon fine-tuning sur datasets de taille moyenne. |
| **ConvNeXt** | Tiny, Small, Base | “CNN pour les années 2020”, proche des perfs des transformers à taille égale. |
| **Vision Transformer (ViT)** | ViT-B/16, Swin | Très bonnes perfs avec beaucoup de données ; plus sensibles en faible données. |
| **ResNet / DenseNet** | ResNet50, DenseNet169 | Toujours solides en baseline et en transfer learning médical. |

Références : [1](https://hiringnet.com/image-classification-state-of-the-art-models-in-2025), [2](https://www.nature.com/articles/s41598-024-72752-x), [3](https://arxiv.org/abs/2405.00857) (Brighteye – ViT glaucome), [4](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296229) (EfficientNet-B3 + attention).

---

## 2. Spécificité fond d’œil / glaucome

- **EfficientNet** : très présent dans les benchmarks glaucome (AIROGS, REFUGE, RIM-ONE, etc.) ; B0/B3/B4 donnent d’excellents résultats (staging, screening).
- **ViT** : méthodes type “Brighteye” montrent des gains en sensibilité à 95 % spécificité, mais souvent avec prétraitement (localisation du disque, etc.) et besoin de plus de données pour être stables.
- **ConvNeXt** : peu de comparaisons directes fond d’œil dans la littérature ; en général très compétitif en classification ImageNet et proche des transformers.
- **Conclusion littérature** : la qualité et la diversité des données comptent souvent autant que l’architecture ; EfficientNet et ResNet restent des références solides pour le fond d’œil.

---

## 3. Recommandation pour ce projet

- **Par défaut : rester sur EfficientNet-B0.**  
  Justifications :
  - Prouvé en glaucome / fond d’œil.
  - Bon rapport précision / temps d’entraînement et coût GPU (RunPod, etc.).
  - Disponible dans torchvision, pipeline et calibration déjà en place.

- **Alternatives intégrées au code (optionnel)** :
  - **efficientnet_v2_s** : si tu veux tester une variante “moderne” EfficientNet (entraînement plus rapide, perfs souvent proches ou légèrement meilleures).
  - **convnext_tiny** : si tu veux un CNN récent type “post-ResNet” ; à comparer à EfficientNet-B0 sur ton jeu AIROGS (eval / holdout).

Tu peux lancer un même entraînement en changeant uniquement `backbone` dans `configs/dl_baseline.yaml` (ex. `efficientnet_v2_s` ou `convnext_tiny`) et comparer les métriques (AUC, pAUC, Sens@95 %) sur eval et holdout.

---

## 4. Nos données sont-elles suffisantes ?

**Oui.** On dispose d’environ **45 900 images d’entraînement** (dossiers 0+1+2) et **8 100 en validation**, soit ~54k images au total. En imagerie médicale, beaucoup d’études utilisent 5k–20k images ; au-delà de 30–40k, on est dans un régime où des modèles plus gros (EfficientNet-B4, EfficientNetV2-M, ConvNeXt, ViT) peuvent être entraînés sans sous-exploiter les données, à condition d’utiliser :

- **poids de classe** (`pos_weight`) pour le déséquilibre NRG/RG,
- **data augmentation** (déjà en place),
- **early stopping** sur la validation pour limiter l’overfitting.

Conclusion : le volume actuel est **suffisant pour viser une grosse perf** avec un modèle plus lourd et une résolution plus haute (ex. 384px).

---

## 5. Quel modèle si on vise la perf max (coût d’entraînement non limitant) ?

Recommandation **performance maximale** :

| Choix | Backbone | Résolution | Commentaire |
|-------|----------|------------|-------------|
| **Recommandé** | **EfficientNetV2-M** | **384** | Bon compromis capacité / stabilité, entraînement plus rapide qu’EfficientNet-B4, très bon sur ImageNet et en transfer. |
| Alternative 1 | EfficientNet-B4 | 384 | Prouvé en glaucome, ~19 M paramètres. |
| Alternative 2 | ConvNeXt-Small | 384 | CNN moderne, ~50 M paramètres. |
| Option “tout en un” | ViT-B/16 | 384 | Très forte capacité, mais plus de risque d’overfitting ; à tester si les autres ne suffisent pas. |

**Config prête à l’emploi :** `configs/dl_highperf.yaml`  
- `backbone: efficientnet_v2_m`  
- `image_size: 384`  
- `batch_size: 24` (réduire à 16 si OOM sur GPU 24 Go)  
- `num_epochs: 40`, `patience: 7`  
- Calibration activée.

Commande :

```bash
python -m src.dl_train --config configs/dl_highperf.yaml
```

À lancer sur **GPU** (RunPod A40 ou RTX 4090). Si la VRAM est juste, passer à `image_size: 224` ou `batch_size: 16` dans ce même fichier.

---

## 5bis. Plan d’expériences DL recommandé (AIROGS, A40)

Pour converger rapidement vers un modèle « produit » sans tuning manuel :

- **Expérience 1 – Baseline DL (comparaison ML vs DL)**  
  - Config : `configs/dl_baseline.yaml`  
  - `backbone: efficientnet_b0`, `image_size: 224`, `batch_size: 32`, `num_epochs: 30`, `patience: 5`  
  - Objectif : montrer le gain clair par rapport au meilleur pipeline ML (AUC, pAUC, Sens@95 % sur eval/holdout).

- **Expérience 2 – Modèle produit candidat (EfficientNetV2-M)**  
  - Config : `configs/dl_highperf.yaml` (A40 ou GPU ≥ 24 Go)  
  - Hyperparamètres principaux :  
    - `backbone: efficientnet_v2_m`  
    - `image_size: 384`  
    - `batch_size: 24` (adapter si VRAM différente)  
    - `num_epochs: 40`, `patience: 7`  
    - `lr: 8e-5`, `weight_decay: 1e-4`  
    - `pos_weight: "auto"` (calculé à partir du ratio NRG/RG sur les dossiers 0+1+2)  
    - `scheduler: cosine` avec `T_max = num_epochs`, `eta_min = 1e-6`  
  - Ces paramètres sont pensés pour laisser l’entraînement **s’auto‑ajuster** (LR dynamique, early stopping) une fois le pod lancé.

- **Expérience 3 – Stabilité (seed alternatif)**  
  - Rejouer l’Expérience 2 avec un `RANDOM_STATE` différent dans `src/config.py` (par ex. 123 au lieu de 42).  
  - But : vérifier la **stabilité des métriques** (AUC, pAUC, Sens@95 %) sur eval et holdout.

- **Expérience 4 (optionnelle) – ConvNeXt-Small 384**  
  - Dupliquer `configs/dl_highperf.yaml` en `configs/dl_convnext_small.yaml` avec :  
    - `backbone: convnext_small`, `image_size: 384`, mêmes `batch_size`, `num_epochs`, `lr`, `weight_decay`, `pos_weight: "auto"`, `scheduler: cosine`.  
  - Objectif : contrôler que l’on ne laisse pas 2–3 points de Sens@95 % sur la table par rapport à un CNN moderne alternatif.

À l’issue de ces 3–4 expériences, le **modèle « produit » v1.0** est choisi comme celui qui :

- Atteint ou dépasse les cibles `product_targets.yaml` (AUC, pAUC, Sens@95 %) sur eval,  
- Se comporte de façon acceptable sur le holdout,  
- Offre des heatmaps Grad‑CAM cliniquement plausibles (disque, anneau neurorétinien).

---

## 6. Modèles disponibles dans le projet (torchvision)

| Backbone | Paramètres (ordre de grandeur) | Usage typique |
|----------|-------------------------------|----------------|
| `efficientnet_b0` | ~5 M | **Baseline** – bon rapport coût/perf. |
| `efficientnet_b4` | ~19 M | Haute perf, éprouvé glaucome. |
| `efficientnet_v2_s` | ~24 M | Variante récente, rapide à entraîner. |
| `efficientnet_v2_m` | ~54 M | **Haute perf** – config `dl_highperf.yaml`. |
| `convnext_tiny` | ~28 M | CNN moderne. |
| `convnext_small` | ~50 M | Haute perf, alternative à V2-M. |
| `resnet18` / `resnet50` | ~11 M / ~25 M | Baselines. |
| `vit_b_16` | ~86 M | Très gros, à tester si données et GPU suffisants. |

La calibration (temperature scaling) et la config (YAML) s’appliquent à tous ces backbones.

---

## 7. Références

- AIROGS Challenge : [airogs.grand-challenge.org](https://airogs.grand-challenge.org/)
- EfficientNetV2 (torchvision) : [pytorch.org/vision/stable/models/efficientnetv2.html](https://pytorch.org/vision/stable/models/efficientnetv2.html)
- ConvNeXt (torchvision) : [docs.pytorch.org/vision/stable/models/convnext.html](https://docs.pytorch.org/vision/stable/models/convnext.html)
- Brighteye (ViT glaucome) : [arxiv.org/abs/2405.00857](https://arxiv.org/abs/2405.00857)
- Comparative evaluation ophthalmology (Nature 2024) : [nature.com/articles/s41598-024-72752-x](https://www.nature.com/articles/s41598-024-72752-x)
