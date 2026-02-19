"""
Prétraitement avancé pour images de fond d'œil (fundus).
CLAHE, extraction canal vert, masque circulaire, détection ROI.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applique CLAHE sur le canal L de l'espace LAB.
    Normalise l'éclairage entre différentes caméras/conditions.
    img: (H, W, 3) RGB uint8.
    """
    if cv2 is None:
        raise ImportError("opencv-python requis pour CLAHE")
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_green_channel(img: np.ndarray) -> np.ndarray:
    """
    Extrait le canal vert (le plus informatif pour les structures rétiniennes).
    Retourne une image grayscale (H, W).
    """
    if len(img.shape) == 3:
        return img[:, :, 1]
    return img


def create_circular_mask(h: int, w: int, margin: float = 0.02) -> np.ndarray:
    """
    Crée un masque circulaire pour exclure les bords noirs du fundus.
    Retourne un masque booléen (H, W).
    """
    center_y, center_x = h // 2, w // 2
    radius = int(min(h, w) // 2 * (1 - margin))
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    return dist <= radius


def detect_fundus_roi(img: np.ndarray, threshold: int = 15) -> tuple:
    """
    Détecte la zone utile (non noire) de l'image fundus.
    Retourne (center_x, center_y, radius) du cercle englobant.
    """
    if cv2 is None:
        h, w = img.shape[:2]
        return w // 2, h // 2, min(h, w) // 2

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Seuiller pour trouver la zone non noire
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Trouver les contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = img.shape[:2]
        return w // 2, h // 2, min(h, w) // 2

    # Plus grand contour = zone du fundus
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return int(cx), int(cy), int(radius)


def create_roi_mask(img: np.ndarray, threshold: int = 15, margin: float = 0.02) -> np.ndarray:
    """
    Crée un masque basé sur la détection du ROI fundus.
    Plus robuste que le masque circulaire fixe pour des images de tailles variées.
    Retourne un masque uint8 (H, W) avec 255 pour les pixels valides.
    """
    cx, cy, radius = detect_fundus_roi(img, threshold)
    radius = int(radius * (1 - margin))
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask_bool = dist <= radius
    return mask_bool.astype(np.uint8) * 255


def preprocess_fundus(
    img: np.ndarray,
    apply_clahe_flag: bool = True,
    clip_limit: float = 2.0,
    grid_size: tuple = (8, 8),
) -> tuple:
    """
    Pipeline de prétraitement complet pour une image fundus.
    1. Détection du ROI (masque circulaire adaptatif)
    2. Application de CLAHE pour normaliser l'éclairage

    Retourne (img_preprocessed, mask_uint8).
    img_preprocessed: (H, W, 3) RGB uint8 avec CLAHE appliqué.
    mask_uint8: (H, W) uint8 avec 255 pour les pixels du fundus.
    """
    # Créer le masque ROI
    mask = create_roi_mask(img)

    # Appliquer CLAHE
    if apply_clahe_flag:
        img = apply_clahe(img, clip_limit=clip_limit, grid_size=grid_size)

    return img, mask
