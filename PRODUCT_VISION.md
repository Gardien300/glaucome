# Vision produit — Dépistage du glaucome référable

**Objectif :** Construire un produit **commercialisable** et **ultra efficace pour le patient**, capable d’aider au dépistage du glaucome référable (RG) à partir de photographies du fond d’œil.

---

## 1. Efficacité pour le patient

### 1.1 Ce que “ultra efficace” signifie en clinique

- **Ne pas manquer les patients à risque (RG)**  
  → Maximiser la **sensibilité** à un niveau de spécificité fixe (ex. 95 %), pour que les vrais glaucomes soient bien repérés et orientés.
- **Limiter les sur-reférés et l’anxiété inutile**  
  → Garder une **spécificité** élevée (ex. 95 %) pour ne pas surcharger les spécialistes ni inquiéter à tort.
- **Aider le clinicien à décider en confiance**  
  → **Probabilités calibrées** (temperature scaling) pour que le score reflète un vrai niveau de risque utilisable en pratique.
- **Fonctionner dans des conditions réalistes**  
  → Modèle évalué sur un **holdout indépendant** (dossier 5) et sur des **splits multi-sites** (dossiers 0–4) pour refléter la variabilité des appareils et des populations.

### 1.2 Objectifs chiffrés cibles (produit “prêt”)

À considérer comme **objectifs minimaux** pour viser un produit déployable et défendable cliniquement :

| Métrique | Cible minimale | Rôle pour le patient |
|----------|----------------|----------------------|
| **Sensibilité @ 95 % spécificité** | ≥ 0,75 (idéal ≥ 0,80) | Ne pas manquer les RG quand on accepte 5 % de faux positifs parmi les NRG. |
| **pAUC (90–100 % spec)** | ≥ 0,80 (idéal ≥ 0,85) | Bonne discrimination dans la zone clinique (peu de faux positifs). |
| **AUC-ROC** | ≥ 0,90 | Performance globale de tri RG / NRG. |
| **Calibration** | ECE faible, courbe fiable | Les scores servent à prioriser et à expliquer, pas seulement à classer. |

Ces seuils sont des **cibles produit** : les runs (ML et DL) doivent les atteindre sur **eval** et **holdout** avant de prétendre à un déploiement clinique.

### 1.3 Ce que le pipeline actuel apporte déjà

- **Métriques alignées AIROGS** : pAUC, Sens@95 %, Sens@90 % — directement exploitables pour le cahier des charges clinique.
- **Calibration (temperature scaling)** : probabilités utilisables pour le tri et la communication du risque.
- **Évaluation généralisation** : train (0+1+2), val, eval (3+4), holdout (5) — évite le sur-ajustement à un seul centre.
- **Traçabilité** : config YAML, checkpoint avec backbone + température, rapports (RAPPORT_ML_CLASSIQUE_GLAUCOME.txt, etc.).

---

## 2. Commercialisation : exigences produit

### 2.1 Réglementation (SaMD — Software as Medical Device)

- Le logiciel sera très probablement un **dispositif médical logiciel** (classe IIa ou IIb selon la réglementation).
- **CE (Europe)** : conformité MDR, documentation technique, validation clinique, gestion des risques (ISO 14971), qualité (ISO 13485).
- **FDA (États-Unis)** : voie 510(k) ou De Novo selon le niveau de preuve et l’intention d’usage (aide au diagnostic vs dépistage).

Pour avancer vers la commercialisation, il faudra à terme :
- Définir l’**intended use** (dépistage du glaucome référable, en aide à la décision, pas diagnostic seul).
- Documenter les **performances** sur des données représentatives (internal + si possible external validation).
- Mettre en place **traçabilité** des versions de modèle, des jeux d’évaluation et des métriques (déjà en partie fait avec config + checkpoints).

### 2.2 Validation clinique

- **Validation interne** : eval (dossiers 3+4) et holdout (dossier 5) déjà en place.
- **Validation externe** : idéalement tester sur un autre dataset (ex. REFUGE, RIM-ONE, ou partenaire hospitalier) pour renforcer le dossier réglementaire.
- **Interprétabilité / confiance** : probabilités calibrées, et à terme explications (cartes d’attention, SHAP si pertinent) pour renforcer l’acceptation clinique.

### 2.3 Déploiement et usage cible

- **Utilisateurs** : ophtalmologistes, services de dépistage, télémédecine, optométristes (selon juridiction).
- **Intégration** : API (score + probabilité + seuils), ou intégration dans une station de lecture / PACS.
- **Robustesse** : gestion des images non gradables (hors scope actuel mais prévu dans AIROGS) — à traiter en phase produit (pré-détection qualité, rejet avec explication).

---

## 3. Feuille de route “produit”

| Étape | Statut / action |
|-------|------------------|
| **Performances cibles** | Définies ci-dessus ; les atteindre sur eval + holdout. |
| **Calibration** | ✅ Intégrée (temperature scaling) en DL. |
| **Métriques cliniques** | ✅ pAUC, Sens@95 %, AUC, bootstrap CI. |
| **Documentation des runs** | Sauvegarder pour chaque run : config, métriques eval/holdout, chemin du checkpoint. |
| **Validation externe** | À planifier (autre dataset ou partenaire). |
| **Intended use & réglementation** | Rédiger le cahier des charges SaMD et l’aligner sur MDR/FDA. |
| **Déploiement (API / intégration)** | Après validation ; versioning modèle + seuils configurables. |

---

## 4. Résumé

- **Ultra efficace pour le patient** = **sensibilité élevée à haute spécificité** (ne pas manquer les RG, limiter les faux positifs) + **probabilités calibrées** (décision et communication du risque) + **évaluation sur holdout** (généralisation).
- **Commercialisable** = atteindre les **objectifs chiffrés** (Sens@95 %, pAUC, AUC), documenter les performances, puis engager la **validation clinique** et le **dossier réglementaire** (SaMD, CE/FDA).

Le pipeline actuel (ML + DL, calibration, métriques AIROGS, splits multi-dossiers) est conçu pour servir cette vision ; les prochaines étapes sont d’**atteindre et tracer les cibles produit** sur chaque run, puis d’enchaîner validation externe et cadre réglementaire.
