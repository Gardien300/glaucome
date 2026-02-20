## Rétinographes & compatibilité avec les algorithmes

Document de référence listant tous les rétinographes utilisés (ou potentiels) et les caractéristiques impactant la compatibilité avec nos pipelines ML (prétraitement, normalisation, entraînement, inférence).

### Structure par appareil

Pour chaque rétinographe, on documente :
- **Contexte d’usage**
- **Spécifications image** (champ, résolution, profondeur de couleur…)
- **Modalités disponibles** (couleur, red-free, IR…)
- **Formats & métadonnées**
- **Compatibilité avec nos algorithmes**
- **Risques / points d’attention**
- **Actions à prévoir**

---

### 1. Topcon Maestro2 (OCT-A multimodal)

#### 1.1. Description générale

- **Nom complet** : Topcon Maestro2 (OCT-A multimodal)
- **Type d’appareil** : OCT + rétinographe couleur
- **Lieu / centre** : _(à compléter)_
- **Utilisation prévue dans le projet** : 
  - Suivi glaucome ? Dépistage ? Screening masse ? _(à préciser)_

#### 1.2. Modalités de photographie fundus

- **Types de photographie** :
  - Couleur (true color, 24 bits, lumière blanche)
  - Red-free¹ (sans hémoglobine)
  - Infrarouge (IR)
- **Combinaisons possibles** : 
  - Multimodal avec OCT/OCTA (superposition précise fundus + OCT/OCTA)
  - Macula, disque optique (zone(s) supportée(s) pour superposition)

#### 1.3. Champ et résolution

- **Angle d’image** : 45° ± 5 % (standard fundus)
- **Résolution sur fundus** :
  - Centre ≥ 60 lignes/mm
  - Image simultanée avec OCT en couleur 24 bits
- **Taille d’image (pixels)** : _(à compléter, ex. 2048×2048)_
- **Taille physique du champ** : _(si connue, ex. 6×6 mm, 9×9 mm)_

#### 1.4. Conditions d’acquisition

- **Diamètre pupillaire requis** :
  - ≥ 4.0 mm (sans petit diaphragme)
  - ≥ 2.5 mm pour scans combinés (fundus + OCT/OCTA)
- **Mode de capture** :
  - Automatisé (single-touch) avec alignement robotisé et autofocus
  - Mode manuel également disponible
- **Qualité / artefacts fréquents** : _(à documenter selon retour terrain)_

#### 1.5. Formats, export et métadonnées

- **Logiciel de gestion** : `IMAGEnet6`
- **Formats d’export image fundus** :
  - Natif IMAGEnet6
  - Probablement TIFF/JPEG haute résolution (à confirmer)
- **Métadonnées disponibles** (liées au glaucome) :
  - Épaisseur RNFL
  - Épaisseur GCL
  - Hood Report (glaucome)
  - Coordonnées de superposition OCT/OCTA
- **Modalités d’export** :
  - Export batch possible ? _(oui/non/à confirmer)_
  - Export avec anonymisation ? _(oui/non/à préciser)_

#### 1.6. Compatibilité avec nos algorithmes

- **Couleur / dynamique** :
  - True color 24 bits → **compatible** avec modèles RGB classiques (ResNet, EfficientNet, ViT…).
  - Lumière blanche → comparable aux bases publiques type Kaggle/REFUGE ? _(à vérifier visuellement)_
- **Champ de vision (45°)** :
  - **Compatible** pour nos modèles centrés sur le disque optique et la macula.
  - Attention à la comparaison avec d’autres appareils (30°, widefield, etc.).
- **Résolution (≥ 60 lignes/mm)** :
  - A priori **suffisante** pour détection des structures glaucomateuses (anneau neurorétinien, RNFL, vaisseaux).
  - À vérifier :
    - Résolution finale en pixels
    - Impact de la compression (JPEG vs TIFF)
- **Modalités spéciales** :
  - Red-free : potentiellement **intéressant** pour segmentation des vaisseaux / RNFL.
  - IR : possiblement **utile** pour certaines tâches spécifiques, mais pas forcément pour les modèles actuels (à décider).
  - OCT/OCTA + Hood Report : **fort potentiel** pour modèles multimodaux, mais hors du scope des modèles purement fundus actuels.

##### Conclusion compatibilité (résumé)

- **Compatibilité globale fundus couleur** : ✅ **Compatible** avec nos modèles CNN actuels (entrée RGB, 24 bits).
- **Prétraitements nécessaires** :
  - Normalisation taille (ex. resize en 512×512 / 1024×1024)
  - Normalisation couleur (mean/std adaptées au dataset Maestro2 ou au dataset de pré-entraînement)
  - Vérification de la compression (éviter JPEG trop compressé si possible → privilégier TIFF ou JPEG qualité max).
- **Différences vs autres appareils** :
  - Angle 45° standard → facile à intégrer dans un dataset multi-appareils avec champs proches.
  - Multimodalité OCT/OCTA → nécessite pipelines ML spécifiques (non utilisés dans la V1 des modèles images 2D couleur).

#### 1.7. Risques / points d’attention

- **Risque de variabilité** :
  - Balance des blancs / couleur différente vs autres caméras → risque de domain shift.
  - Il faudra vérifier la performance spécifique des modèles sur des lots Maestro2 seuls.
- **Risque lié à la compression** :
  - Si export en JPEG qualité moyenne → perte de détails fins (RNFL, micro-hémorragies).
- **Alignement multimodal** :
  - Si un jour on utilise OCT/OCTA → bien conserver les métadonnées d’alignement / repères.

#### 1.8. Actions à prévoir

- **Court terme** :
  - [ ] Récupérer un petit lot de tests (ex. 50–100 images fundus couleur Maestro2, disque + macula).
  - [ ] Vérifier format exact (taille en pixels, TIFF/JPEG, profil couleur).
  - [ ] Tester inférence avec nos modèles actuels (sans ré-entraînement).
- **Moyen terme** :
  - [ ] Envisager un fine-tuning dédié Maestro2 si domain shift important.
  - [ ] Documenter un pipeline standard d’import IMAGEnet6 → dataset ML (scripts, conventions de nommage).
  - [ ] Étudier l’usage combiné fundus + métriques Hood Report pour un modèle multimodal glaucome.

---

### 2. Topcon NW500

#### 2.1. Description générale

- **Nom complet** : Topcon NW500
- **Type d’appareil** : Rétinographe non-mydriatique robotisé
- **Génération** : Récente (≈ 2024+)
- **Lieu / centre** : _(à compléter)_
- **Positionnement clinique** : Screening rapide, flux élevé, délégable à orthoptiste/infirmier·e.

#### 2.2. Modalités de photographie fundus

- **Types de photographie** :
  - Couleur vraie (true color) pour fundus 2D
  - Vue stéréo / périphérie possible selon protocole
- **Technologie d’acquisition** :
  - Slit-Scan + Rolling Shutter (optimisée petites pupilles et lumière ambiante)
  - Conçue pour réduire flare et ombres

#### 2.3. Champ et résolution

- **Champ visuel** : 50°
- **Résolution capteur** : 12 MP
- **Résolution sur fundus** :
  - Centre ≥ 60 lp/mm
- **Taille d’image (pixels)** : _(à préciser si connue)_

#### 2.4. Conditions d’acquisition

- **Diamètre pupillaire minimal** :
  - ≥ 2.0 mm (optimisé petites pupilles)
- **Mode de capture** :
  - Full robotic, 1-touch (alignement + focus automatiques)
  - Bilatéral 2 champs en ≈ 30 secondes
- **Conditions lumineuses** :
  - Utilisable en lumière ambiante
- **Impact clinique** :
  - Très adapté pour dépistage de masse / consultations chargées.

#### 2.5. Formats, export et métadonnées

- **Logiciel de gestion** : `IMAGEnet6`
- **Formats d’export image fundus** :
  - JPEG / TIFF haute résolution (via IMAGEnet6)
- **Métadonnées** :
  - Standard IMAGEnet6 (identifiants patient/examen, œil, date, etc.)
  - _(à compléter si présence de tags additionnels spécifiques NW500)_

#### 2.6. Compatibilité avec nos algorithmes

- **Couleur / dynamique** :
  - True color 12 MP → **très compatible** avec nos modèles CNN (entrée RGB).
  - Lumière ambiante possible → variabilité de luminosité à gérer par prétraitement.
- **Champ de vision (50°)** :
  - Proche standard 45–50° → **intégrable** dans nos pipelines actuels.
- **Résolution (12 MP, ≥60 lp/mm)** :
  - Résolution élevée → marge confortable pour downsampling (512–1024 px) sans perte de détails glaucomateux.
- **Technologie Slit-Scan / Rolling Shutter** :
  - Réduction des artefacts (flare, ombres) → a priori **bénéfique** pour stabilité du modèle.
  - Mais profil d’intensité potentiellement différent des caméras « classiques » → risque de **domain shift léger**.

##### Conclusion compatibilité (résumé)

- **Compatibilité globale** : ✅ **Très compatible** pour nos modèles fundus couleur.
- **Prétraitements recommandés** :
  - Normalisation de la taille (resize) et de la luminance (CLAHE / normalisation gamma à tester).
  - Vérifier l’impact de la lumière ambiante sur le contraste (éventuel rehaussement local).
- **Spécificités vs autres appareils** :
  - Petites pupilles et lumière ambiante → distribution d’illumination différente, à prendre en compte dans les splits de validation.

#### 2.7. Risques / points d’attention

- **Variabilité illumination** :
  - Lumière ambiante → fluctuations d’éclairage entre centres / salles.
- **Domain shift vs caméras à flash classique** :
  - Couleurs et contraste peuvent différer de bases historiques (TRC-50DX, Maestro2…).

#### 2.8. Actions à prévoir

- **Court terme** :
  - [ ] Récupérer un batch pilote NW500 (ex. 100–200 images) pour analyse statistique (histogrammes, couleurs, sharpness).
  - [ ] Tester l’inférence de nos modèles actuels sur ce batch, sans ré-entraînement.
- **Moyen terme** :
  - [ ] Si écart de perf significatif → prévoir fine-tuning/domain adaptation spécifique NW500.
  - [ ] Documenter le pipeline d’export IMAGEnet6 → dataset (scripts, naming).

---

### 3. Topcon TRC-50DX

#### 3.1. Description générale

- **Nom complet** : Topcon TRC-50DX
- **Type d’appareil** : Rétinographe (principalement non-mydriatique, génération plus ancienne)
- **Génération** : Plus ancienne que NW500 (≈ TRC-NW400 / CCD standard)
- **Lieu / centre** : _(à compléter)_
- **Positionnement clinique** : Appareil de référence historique pour imagerie fundus couleur.

#### 3.2. Modalités de photographie fundus

- **Types de photographie** :
  - Fundus couleur standard (flash xenon + CCD)
  - Autres modalités possibles selon configuration (red-free, etc.) _(à préciser)_
- **Technologie d’acquisition** :
  - Flash conventionnel
  - CCD standard

#### 3.3. Champ et résolution

- **Champ visuel** : 50°
- **Résolution capteur** :
  - Inférieure au NW500 (pas de 12 MP spécifié, CCD standard)
- **Résolution sur fundus** :
  - Non spécifiée, mais historiquement suffisante pour interprétation clinique.

#### 3.4. Conditions d’acquisition

- **Diamètre pupillaire typique** :
  - ≥ 2.5–3.0 mm (plus sensible au myosis que NW500)
- **Mode de capture** :
  - Semi-automatique
  - Plus lent et opérateur-dépendant vs robotisé NW500
- **Conditions lumineuses** :
  - Environnement plus contrôlé / obscurci généralement recommandé.

#### 3.5. Formats, export et métadonnées

- **Formats d’export** :
  - Fundus couleur classiques (souvent JPEG ou TIFF, selon setup)
- **Métadonnées** :
  - Standard (identifiants, œil, date, etc.) — à documenter précisément selon le centre.

#### 3.6. Compatibilité avec nos algorithmes

- **Couleur / dynamique** :
  - Imagerie couleur « traditionnelle » (flash xenon) → **proche** de nombreuses bases publiques et cliniques historiques.
  - A priori **très compatible** avec les modèles CNN pré-entraînés sur fundus classiques.
- **Champ de vision (50°)** :
  - Aligné avec NW500 → facilite mélange de données si les prétraitements sont bien normalisés.
- **Résolution** :
  - Inférieure au NW500 mais suffisante pour usage clinique → a priori **suffisante** pour nos modèles, sous réserve de qualité d’acquisition.
- **Comparaison NW500 vs TRC-50DX** :
  - TRC-50DX plus sensible au myosis et aux conditions de lumière → risque d’images plus bruitées / sous-exposées.

##### Conclusion compatibilité (résumé)

- **Compatibilité globale** : ✅ **Compatible** avec nos modèles CNN actuels (fondus couleur standard).
- **Prétraitements recommandés** :
  - Normalisation couleur et contraste (surtout pour images sous/sur-exposées).
  - Vérifier homogénéité de résolution si mélange avec NW500 dans un même dataset.
- **Rôle dans le projet** :
  - Bon candidat comme « caméra de référence historique » pour entraîner ou valider les modèles sur des conditions plus classiques.

#### 3.7. Risques / points d’attention

- **Variabilité opérateur** :
  - Semi-manuelle → grande hétérogénéité de centrage, focus et exposition.
- **Sensibilité à la pupille et lumière** :
  - Myosis et mauvaise obscurité → risque de qualité dégradée, nécessitant filtrage qualité avant entraînement.

#### 3.8. Actions à prévoir

- **Court terme** :
  - [ ] Collecter un échantillon d’images TRC-50DX issues de la pratique réelle (qualité variable).
  - [ ] Mettre en place un score qualité automatique (flou, centrage, exposition) pour filtrer les pires images avant entraînement.
- **Moyen terme** :
  - [ ] Étudier performances comparées NW500 vs TRC-50DX (AUC, sensibilité, spécificité par appareil).
  - [ ] Envisager des augmentations spécifiques (blur, variations d’exposition) pour mieux couvrir la variabilité TRC-50DX.

---

### 4. Autres rétinographes (à compléter)

Pour chaque nouvel appareil, copier/coller la structure ci-dessus et adapter :

- **Nom complet** : _(à compléter)_
- **Type d’appareil** : _(rétinographe non-mydriatique / mydriatique, OCT, widefield, etc.)_
- **Lieu / centre** : _(à compléter)_
- **Spécifications image** : _(champ, résolution, profondeur de couleur…)_
- **Modalités disponibles** : _(couleur, red-free, IR, auto-fluo, etc.)_
- **Formats & métadonnées** : _(formats d’export, tags DICOM, rapports intégrés…)_
- **Compatibilité avec nos algorithmes** : _(OK / à adapter / non compatible en l’état)_
- **Risques / points d’attention** : _(domain shift, qualité, artefacts…)_
- **Actions à prévoir** : _(tests d’inférence, fine-tuning, scripts de conversion…)_

