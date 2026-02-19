"""
Features spatiales spécifiques au glaucome.
Détection du disque optique, ratio cup/disc, règle ISNT, features par zones et quadrants.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Détection du disque optique
# ---------------------------------------------------------------------------

def estimate_optic_disc(img: np.ndarray) -> tuple:
    """
    Localise le disque optique (zone la plus brillante du fundus).
    Utilise le canal vert (ou rouge selon le contraste) avec flou gaussien + seuillage.
    Retourne (center_x, center_y, radius).
    """
    if cv2 is None:
        h, w = img.shape[:2]
        return w // 2, h // 2, min(h, w) // 8

    if len(img.shape) == 3:
        # Le canal rouge est souvent le plus brillant pour le disque optique
        red = img[:, :, 0]
        green = img[:, :, 1]
        # Combiner rouge et vert pour une meilleure détection
        bright = cv2.addWeighted(red.astype(np.float32), 0.6, green.astype(np.float32), 0.4, 0)
        bright = bright.astype(np.uint8)
    else:
        bright = img

    h, w = bright.shape[:2]

    # Flou gaussien large pour lisser et trouver le centre du disque
    ksize = max(h, w) // 4
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 3)
    blurred = cv2.GaussianBlur(bright, (ksize, ksize), 0)

    # Position du max d'intensité = centre approximatif du disque
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx, cy = max_loc

    # Raffiner : seuiller la zone brillante autour du centre
    # Seuil adaptatif basé sur le percentile d'intensité
    threshold_val = np.percentile(bright, 95)
    _, binary = cv2.threshold(bright, int(threshold_val), 255, cv2.THRESH_BINARY)

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Trouver le contour le plus proche du centre estimé
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sélectionner le contour dont le centre est le plus proche du max d'intensité
        best_contour = None
        best_dist = float("inf")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Ignorer les très petits contours
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cnt_cx = int(M["m10"] / M["m00"])
            cnt_cy = int(M["m01"] / M["m00"])
            dist = np.sqrt((cnt_cx - cx) ** 2 + (cnt_cy - cy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_contour = cnt

        if best_contour is not None:
            (cx_ref, cy_ref), radius = cv2.minEnclosingCircle(best_contour)
            return int(cx_ref), int(cy_ref), max(int(radius), 10)

    # Fallback : utiliser le max d'intensité et un rayon par défaut
    default_radius = min(h, w) // 12
    return cx, cy, max(default_radius, 10)


# ---------------------------------------------------------------------------
# Features par zones concentriques
# ---------------------------------------------------------------------------

def extract_zone_features(img: np.ndarray, od_cx: int, od_cy: int, od_radius: int) -> np.ndarray:
    """
    Features par zones concentriques autour du disque optique.
    Zone 1 : intérieur du disque (cup potentielle)
    Zone 2 : bord du disque (neuroretinal rim)
    Zone 3 : péripapillaire (couche de fibres nerveuses - RNFL)
    Zone 4 : rétine périphérique

    Pour chaque zone : mean_green, std_green, mean_red, ratio_green/red, entropie.
    4 zones × 5 features = 20 features.
    """
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - od_cx) ** 2 + (Y - od_cy) ** 2)

    r = od_radius
    zones = [
        dist <= r,                           # Zone 1 : intérieur disque
        (dist > r) & (dist <= 1.5 * r),      # Zone 2 : rim
        (dist > 1.5 * r) & (dist <= 3.0 * r),  # Zone 3 : péripapillaire
        dist > 3.0 * r,                      # Zone 4 : périphérie
    ]

    if len(img.shape) == 3:
        green = img[:, :, 1].astype(np.float64)
        red = img[:, :, 0].astype(np.float64)
    else:
        green = img.astype(np.float64)
        red = green

    features = []
    for zone_mask in zones:
        g_vals = green[zone_mask]
        r_vals = red[zone_mask]

        if len(g_vals) == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        mean_g = float(g_vals.mean())
        std_g = float(g_vals.std())
        mean_r = float(r_vals.mean())
        ratio_gr = mean_g / (mean_r + 1e-8)

        # Entropie de l'histogramme du canal vert dans la zone
        hist, _ = np.histogram(g_vals, bins=32, range=(0, 256))
        hist = hist.astype(np.float64) / (hist.sum() + 1e-8)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

        features.extend([mean_g, std_g, mean_r, ratio_gr, entropy])

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Features par quadrants ISNT
# ---------------------------------------------------------------------------

def extract_quadrant_features(img: np.ndarray, od_cx: int, od_cy: int, od_radius: int) -> np.ndarray:
    """
    Features par quadrants ISNT autour du disque optique.
    La règle ISNT : en œil sain, l'épaisseur du rim suit I > S > N > T.
    Le glaucome perturbe cet ordre.

    Quadrants dans la zone péripapillaire (1.0r à 2.5r) :
    - I (Inférieur) : en bas
    - S (Supérieur) : en haut
    - N (Nasal) : côté nasal (gauche pour œil droit, droite pour œil gauche — on prend les deux)
    - T (Temporal) : côté temporal

    Pour chaque quadrant : mean_green, std_green, vessel_density_proxy.
    4 quadrants × 3 = 12 features.
    + 6 ratios inter-quadrants (I/S, I/N, I/T, S/N, S/T, N/T).
    + 1 ISNT violation score.
    Total : 19 features.
    """
    h, w = img.shape[:2]
    Y, X = np.meshgrid(np.arange(w), np.arange(h))  # X = colonnes, Y = lignes

    # Zone péripapillaire
    dist = np.sqrt((Y - od_cx) ** 2 + (X - od_cy) ** 2)
    peripap = (dist > od_radius) & (dist <= 2.5 * od_radius)

    # Angles par rapport au centre du disque
    angles = np.arctan2(X - od_cy, Y - od_cx)  # radians, -pi à pi

    # Quadrants (conventions : I=bas, S=haut, N=gauche, T=droite)
    quadrant_masks = {
        "I": (angles >= np.pi / 4) & (angles < 3 * np.pi / 4),      # Inférieur (bas)
        "S": (angles >= -3 * np.pi / 4) & (angles < -np.pi / 4),    # Supérieur (haut)
        "N": (angles >= 3 * np.pi / 4) | (angles < -3 * np.pi / 4), # Nasal (gauche)
        "T": (angles >= -np.pi / 4) & (angles < np.pi / 4),          # Temporal (droite)
    }

    if len(img.shape) == 3:
        green = img[:, :, 1].astype(np.float64)
    else:
        green = img.astype(np.float64)

    quad_means = {}
    features = []

    for name in ["I", "S", "N", "T"]:
        mask = peripap & quadrant_masks[name]
        vals = green[mask]

        if len(vals) == 0:
            features.extend([0.0, 0.0, 0.0])
            quad_means[name] = 0.0
            continue

        mean_g = float(vals.mean())
        std_g = float(vals.std())

        # Proxy de densité vasculaire : proportion de pixels sombres
        # Les vaisseaux sont sombres sur le canal vert
        dark_threshold = mean_g - 1.5 * std_g
        vessel_density = float(np.sum(vals < dark_threshold) / (len(vals) + 1e-8))

        features.extend([mean_g, std_g, vessel_density])
        quad_means[name] = mean_g

    # Ratios inter-quadrants
    pairs = [("I", "S"), ("I", "N"), ("I", "T"), ("S", "N"), ("S", "T"), ("N", "T")]
    for a, b in pairs:
        ratio = quad_means[a] / (quad_means[b] + 1e-8)
        features.append(float(ratio))

    # Score de violation ISNT
    # Règle ISNT : I > S > N > T. Score = nombre de violations / 3
    order = [quad_means["I"], quad_means["S"], quad_means["N"], quad_means["T"]]
    violations = 0
    for i in range(len(order) - 1):
        if order[i] < order[i + 1]:
            violations += 1
    isnt_score = violations / 3.0
    features.append(float(isnt_score))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ratio cup/disc (CDR)
# ---------------------------------------------------------------------------

def estimate_cup_to_disc_ratio(img: np.ndarray, od_cx: int, od_cy: int, od_radius: int) -> np.ndarray:
    """
    Estime le ratio cup/disc (CDR) et le CDR vertical (VCDR).
    Le CDR est le rapport entre l'aire de la cup (zone brillante centrale du disque)
    et l'aire du disque optique.
    CDR > 0.6 est un signe de glaucome.
    Retourne [CDR, VCDR].
    """
    if len(img.shape) == 3:
        green = img[:, :, 1]
    else:
        green = img

    h, w = green.shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - od_cx) ** 2 + (Y - od_cy) ** 2)

    # Zone du disque optique
    disc_mask = dist <= od_radius
    disc_pixels = green[disc_mask]

    if len(disc_pixels) == 0:
        return np.array([0.5, 0.5], dtype=np.float32)

    # La cup est la zone la plus brillante à l'intérieur du disque
    # Seuil adaptatif : percentile élevé des intensités du disque
    cup_threshold = np.percentile(disc_pixels, 70)

    cup_mask = disc_mask & (green >= cup_threshold)
    cup_area = float(cup_mask.sum())
    disc_area = float(disc_mask.sum())

    cdr = cup_area / (disc_area + 1e-8)

    # CDR vertical : ratio sur l'axe vertical passant par le centre du disque
    col_range = max(1, od_radius // 10)
    col_start = max(0, od_cx - col_range)
    col_end = min(w, od_cx + col_range + 1)

    vertical_strip = green[:, col_start:col_end]
    dist_strip = dist[:, col_start:col_end]

    disc_v = dist_strip <= od_radius
    if disc_v.sum() == 0:
        vcdr = cdr
    else:
        disc_v_pixels = vertical_strip[disc_v]
        cup_v_threshold = np.percentile(disc_v_pixels, 70)
        cup_v = disc_v & (vertical_strip >= cup_v_threshold)
        vcdr = float(cup_v.sum()) / (float(disc_v.sum()) + 1e-8)

    return np.array([cdr, vcdr], dtype=np.float32)


# ---------------------------------------------------------------------------
# Features vasculaires
# ---------------------------------------------------------------------------

def extract_vessel_features(img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Features vasculaires basées sur le top-hat morphologique.
    Détecte les vaisseaux (structures sombres fines sur le canal vert).
    Features : densité globale, densité par quadrant (4), stats de contraste.
    Total : 8 features.
    """
    if cv2 is None:
        return np.zeros(8, dtype=np.float32)

    if len(img.shape) == 3:
        green = img[:, :, 1]
    else:
        green = img

    # Top-hat noir (black-hat) : détecte les structures sombres (vaisseaux)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)

    # Seuillage pour binariser les vaisseaux
    _, vessel_binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if mask is not None:
        vessel_binary = vessel_binary & mask

    h, w = green.shape
    total_pixels = float(h * w)
    if mask is not None:
        total_pixels = float(mask.sum() / 255.0) + 1e-8

    # Densité globale
    vessel_density = float(vessel_binary.sum() / 255.0) / total_pixels

    # Densité par quadrant
    mid_h, mid_w = h // 2, w // 2
    quadrants = [
        vessel_binary[:mid_h, :mid_w],    # Haut-gauche
        vessel_binary[:mid_h, mid_w:],     # Haut-droite
        vessel_binary[mid_h:, :mid_w],     # Bas-gauche
        vessel_binary[mid_h:, mid_w:],     # Bas-droite
    ]
    quad_densities = []
    for q in quadrants:
        q_pixels = float(q.size)
        q_density = float(q.sum() / 255.0) / (q_pixels + 1e-8)
        quad_densities.append(q_density)

    # Contraste vasculaire (intensité moyenne des vaisseaux vs fond)
    vessel_pixels = green[vessel_binary > 0]
    non_vessel_pixels = green[vessel_binary == 0]
    if len(vessel_pixels) > 0 and len(non_vessel_pixels) > 0:
        vessel_contrast = float(non_vessel_pixels.mean()) - float(vessel_pixels.mean())
    else:
        vessel_contrast = 0.0

    # Nombre de composantes connexes (proxy de complexité vasculaire)
    n_components, _ = cv2.connectedComponents(vessel_binary)
    branching = float(n_components) / 100.0  # Normaliser

    features = [vessel_density] + quad_densities + [vessel_contrast, branching]
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Extraction complète des features spatiales
# ---------------------------------------------------------------------------

def extract_spatial_features(img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Extrait toutes les features spatiales pour une image fundus.
    1. Détection du disque optique
    2. Features par zones concentriques (20)
    3. Features par quadrants ISNT (19)
    4. Ratio cup/disc (2)
    5. Features vasculaires (8)
    Total : 49 features.
    """
    od_cx, od_cy, od_radius = estimate_optic_disc(img)

    parts = [
        extract_zone_features(img, od_cx, od_cy, od_radius),
        extract_quadrant_features(img, od_cx, od_cy, od_radius),
        estimate_cup_to_disc_ratio(img, od_cx, od_cy, od_radius),
        extract_vessel_features(img, mask=mask),
    ]
    return np.concatenate(parts).astype(np.float32)
