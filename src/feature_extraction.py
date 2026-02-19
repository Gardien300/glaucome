"""
Extraction de features pour le ML classique — projet glaucome (AIROGS).
Options : simple, hog, lbp, lbp_multiscale, haralick, color_spaces, gabor, combined, full.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

# skimage importé à la demande (lazy) pour éviter ~1 min au chargement du module
_local_binary_pattern = None
_greycomatrix = _greycoprops = None

from .preprocessing import load_image
from .preprocessing_advanced import preprocess_fundus, extract_green_channel
from .features_spatial import extract_spatial_features
from .config import PROJECT_ROOT, RANDOM_STATE


# ---------------------------------------------------------------------------
# Features simples (histogrammes RGB + stats)
# ---------------------------------------------------------------------------

def extract_features_simple(img: np.ndarray, bins: int = 32, mask: np.ndarray = None) -> np.ndarray:
    """
    Features simples : histogrammes RGB (concaténés) + moyenne/écart-type par canal.
    img: (H, W, 3) RGB, uint8.
    mask: (H, W) uint8 optionnel, 255 = pixel valide.
    """
    if cv2 is None:
        raise ImportError("opencv-python requis pour feature_extraction")
    features = []
    for c in range(3):
        hist = cv2.calcHist([img], [c], mask, [bins], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        features.extend(hist)
    for c in range(3):
        channel = img[:, :, c]
        if mask is not None:
            channel = channel[mask > 0]
        features.append(float(channel.mean()))
        features.append(float(channel.std()))
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------

def extract_features_hog(img: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    """HOG sur l'image (grise si couleur)."""
    if cv2 is None:
        raise ImportError("opencv-python requis")
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    win_size = size
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    feat = hog.compute(gray)
    return feat.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# LBP
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return img
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _get_lbp():
    global _local_binary_pattern
    if _local_binary_pattern is None:
        from skimage.feature import local_binary_pattern
        _local_binary_pattern = local_binary_pattern
    return _local_binary_pattern


def extract_features_lbp(img: np.ndarray, P: int = 8, R: int = 1, n_bins: int = None) -> np.ndarray:
    """LBP (Local Binary Pattern) : histogramme normalisé (method=uniform)."""
    lbp_fn = _get_lbp()
    gray = _to_gray(img)
    lbp = lbp_fn(gray, P=P, R=R, method="uniform")
    if n_bins is None:
        n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    return hist


def extract_features_lbp_multiscale(img: np.ndarray) -> np.ndarray:
    """LBP multi-échelles : P=8/R=1, P=16/R=2, P=24/R=3. Capture les textures à différentes résolutions."""
    configs = [(8, 1), (16, 2), (24, 3)]
    parts = []
    for P, R in configs:
        parts.append(extract_features_lbp(img, P=P, R=R))
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Haralick (GLCM)
# ---------------------------------------------------------------------------

def _get_haralick():
    global _greycomatrix, _greycoprops
    if _greycomatrix is None:
        try:
            from skimage.feature import greycomatrix, greycoprops
        except ImportError:
            from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops
        _greycomatrix, _greycoprops = greycomatrix, greycoprops
    return _greycomatrix, _greycoprops


def extract_features_haralick(img: np.ndarray, distances=None, angles=None) -> np.ndarray:
    """Haralick (texture) : contrast, correlation, energy, homogeneity sur GLCM."""
    greycomatrix, greycoprops = _get_haralick()
    gray = _to_gray(img)
    if distances is None:
        distances = [1, 2]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = greycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ["contrast", "correlation", "energy", "homogeneity"]
    feats = []
    for prop in props:
        p = greycoprops(glcm, prop).ravel()
        feats.extend(p)
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Color spaces (HSV, LAB, canal vert, moments statistiques)
# ---------------------------------------------------------------------------

def extract_features_color_spaces(img: np.ndarray, mask: np.ndarray = None, bins: int = 32) -> np.ndarray:
    """
    Features multi-espaces couleur :
    - Histogrammes HSV (3×bins), LAB (3×bins), canal vert (bins)
    - Moments statistiques (mean, std, skewness, kurtosis) par canal (RGB+HSV+LAB+green = 10 canaux)
    Total : 7×bins + 10×4 = ~264 features (bins=32).
    """
    if cv2 is None:
        raise ImportError("opencv-python requis")
    from scipy.stats import skew, kurtosis

    features = []

    # Conversions
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    green = img[:, :, 1]

    # Histogrammes HSV
    for c in range(3):
        max_val = 180 if c == 0 else 256
        hist = cv2.calcHist([hsv], [c], mask, [bins], [0, max_val])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        features.extend(hist)

    # Histogrammes LAB
    for c in range(3):
        hist = cv2.calcHist([lab], [c], mask, [bins], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        features.extend(hist)

    # Histogramme canal vert
    hist = cv2.calcHist([img], [1], mask, [bins], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-8)
    features.extend(hist)

    # Moments statistiques pour chaque canal
    all_channels = []
    for c in range(3):
        all_channels.append(img[:, :, c])   # RGB
    for c in range(3):
        all_channels.append(hsv[:, :, c])   # HSV
    for c in range(3):
        all_channels.append(lab[:, :, c])   # LAB
    all_channels.append(green)               # Green

    for ch in all_channels:
        if mask is not None:
            vals = ch[mask > 0].astype(np.float64)
        else:
            vals = ch.ravel().astype(np.float64)
        if len(vals) == 0:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.append(float(vals.mean()))
            features.append(float(vals.std()))
            features.append(float(skew(vals)))
            features.append(float(kurtosis(vals)))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Gabor filters
# ---------------------------------------------------------------------------

def extract_features_gabor(img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Filtres de Gabor multi-échelles et multi-orientations.
    4 fréquences × 6 orientations = 24 filtres, 2 stats chacun = 48 features.
    Capture les textures orientées (vaisseaux, fibres nerveuses).
    """
    if cv2 is None:
        raise ImportError("opencv-python requis")

    gray = _to_gray(img).astype(np.float64)
    frequencies = [0.05, 0.1, 0.2, 0.4]
    orientations = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]

    features = []
    for freq in frequencies:
        for theta in orientations:
            # sigma proportionnel à 1/fréquence
            sigma = 1.0 / (freq * np.pi) * 0.5
            lambd = 1.0 / freq
            kernel = cv2.getGaborKernel(
                ksize=(0, 0), sigma=sigma, theta=theta,
                lambd=lambd, gamma=0.5, psi=0,
            )
            response = cv2.filter2D(gray, cv2.CV_64F, kernel)
            if mask is not None:
                vals = response[mask > 0]
            else:
                vals = response.ravel()
            features.append(float(vals.mean()))
            features.append(float(vals.std()))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined (simple + HOG + LBP + Haralick) — rétrocompatible
# ---------------------------------------------------------------------------

def extract_features_combined(img: np.ndarray, size: tuple = (224, 224), bins: int = 32) -> np.ndarray:
    """Concatène simple + HOG + LBP + Haralick (rétrocompatible)."""
    parts = [
        extract_features_simple(img, bins=bins),
        extract_features_hog(img, size=size),
        extract_features_lbp(img, n_bins=10),
        extract_features_haralick(img),
    ]
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Construction de la matrice de features
# ---------------------------------------------------------------------------

def _compute_cache_key(df: pd.DataFrame, feature_type: str, size: tuple, preprocess: bool) -> str:
    """Calcule un hash unique pour cette combinaison de données et paramètres."""
    paths = sorted(df["path"].tolist()) if "path" in df.columns else []
    key_str = f"{feature_type}_{size}_{preprocess}_{'|'.join(paths[:10])}_{len(paths)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def build_feature_matrix(
    df: pd.DataFrame,
    path_column: str = "path",
    label_column: str = "class",
    root: Path = None,
    size: tuple = (224, 224),
    feature_type: str = "simple",
    max_samples: int = None,
    preprocess: bool = True,
    cache_dir: str = None,
):
    """
    Construit la matrice de features et le vecteur de labels.
    feature_type: "simple", "hog", "lbp", "lbp_multiscale", "haralick",
                  "color_spaces", "gabor", "combined", "full".
    preprocess: si True, applique CLAHE + masque ROI avant l'extraction.
    cache_dir: si fourni, sauvegarde/charge les features en cache .npz.
    """
    from tqdm import tqdm

    root = root or PROJECT_ROOT
    df = df.copy()
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=RANDOM_STATE)

    valid_types = ("simple", "hog", "lbp", "lbp_multiscale", "haralick",
                   "color_spaces", "gabor", "combined", "full")
    if feature_type not in valid_types:
        raise ValueError(f"feature_type doit être parmi {valid_types}")

    # Cache
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_key = _compute_cache_key(df, feature_type, size, preprocess)
        cache_file = cache_path / f"{cache_key}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            return data["X"], data["y"]

    X_list = []
    y_list = []
    first_error = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Features ({feature_type})"):
        path = root / row[path_column]
        try:
            img = load_image(path, size=size)

            # Prétraitement ophtalmo
            mask = None
            if preprocess:
                img, mask = preprocess_fundus(img)

            feat = _extract_single(img, feature_type, size, mask)
            X_list.append(feat)
            y_list.append(1 if row[label_column] == "RG" else 0)
        except ImportError:
            raise
        except Exception as e:
            if first_error is None:
                first_error = e
            continue

    if len(X_list) == 0:
        msg = "Aucune feature extraite. Vérifier les chemins d'images."
        if first_error is not None:
            msg += f" Première erreur : {first_error}"
        raise ValueError(msg) from first_error

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Sauvegarder en cache
    if cache_dir:
        np.savez_compressed(cache_file, X=X, y=y)

    return X, y


def _extract_single(img: np.ndarray, feature_type: str, size: tuple, mask: np.ndarray = None) -> np.ndarray:
    """Extrait les features d'une seule image selon le type demandé."""
    if feature_type == "simple":
        return extract_features_simple(img, bins=32, mask=mask)
    elif feature_type == "hog":
        return extract_features_hog(img, size=size)
    elif feature_type == "lbp":
        return extract_features_lbp(img)
    elif feature_type == "lbp_multiscale":
        return extract_features_lbp_multiscale(img)
    elif feature_type == "haralick":
        return extract_features_haralick(img)
    elif feature_type == "color_spaces":
        return extract_features_color_spaces(img, mask=mask)
    elif feature_type == "gabor":
        return extract_features_gabor(img, mask=mask)
    elif feature_type == "combined":
        return extract_features_combined(img, size=size, bins=32)
    elif feature_type == "full":
        return _extract_full(img, size, mask)
    else:
        raise ValueError(f"feature_type inconnu : {feature_type}")


def _extract_full(img: np.ndarray, size: tuple, mask: np.ndarray = None) -> np.ndarray:
    """
    Feature type 'full' : concatène toutes les features disponibles.
    color_spaces + gabor + lbp_multiscale + haralick + simple + spatial.
    """
    parts = [
        extract_features_color_spaces(img, mask=mask),
        extract_features_gabor(img, mask=mask),
        extract_features_lbp_multiscale(img),
        extract_features_haralick(img),
        extract_features_simple(img, bins=32, mask=mask),
        extract_spatial_features(img, mask=mask),
    ]
    return np.concatenate(parts).astype(np.float32)
