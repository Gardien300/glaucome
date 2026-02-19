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

### 2. Autres rétinographes (à compléter)

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

