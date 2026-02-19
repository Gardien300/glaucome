"""
Prétraitement des images fond d'œil pour la phase ML.
"""

from pathlib import Path
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


def load_image(path: Path, size: tuple = (224, 224), use_cv2: bool = True):
    """
    Charge une image, la redimensionne et retourne un tableau numpy (H, W, C) RGB.
    size: (width, height) ou (height, width) selon la convention utilisée.
    """
    path = Path(path)
    if use_cv2 and cv2 is not None:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Impossible de charger: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif Image is not None:
        img = np.array(Image.open(path).convert("RGB"))
    else:
        raise ImportError("Installer opencv-python ou Pillow")
    # resize: cv2 utilise (width, height)
    if size and (img.shape[0] != size[1] or img.shape[1] != size[0]):
        if cv2 is not None:
            img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
        else:
            img = _resize_pil(img, size)
    return img


def _resize_pil(img: np.ndarray, size: tuple) -> np.ndarray:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img)
    pil = pil.resize((size[0], size[1]), PILImage.Resampling.LANCZOS)
    return np.array(pil)
