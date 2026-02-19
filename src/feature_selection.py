"""
Sélection de features pour réduire la dimensionnalité et améliorer la généralisation.
Pipeline : variance threshold → information mutuelle → PCA optionnel.
"""

import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    SelectKBest,
)
from sklearn.decomposition import PCA

from .config import RANDOM_STATE


def remove_low_variance(X: np.ndarray, threshold: float = 0.01) -> tuple:
    """
    Supprime les features quasi-constantes (variance < threshold).
    Retourne (X_filtered, selector) pour pouvoir transformer le test set.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    n_removed = X.shape[1] - X_filtered.shape[1]
    if n_removed > 0:
        print(f"  VarianceThreshold: {X.shape[1]} → {X_filtered.shape[1]} features ({n_removed} supprimées)")
    return X_filtered, selector


def select_by_mutual_info(X: np.ndarray, y: np.ndarray, k: int = 100) -> tuple:
    """
    Sélectionne les k meilleures features par information mutuelle.
    Retourne (X_selected, selector).
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE),
        k=k,
    )
    X_selected = selector.fit_transform(X, y)
    print(f"  Mutual Info: {X.shape[1]} → {X_selected.shape[1]} features (top-{k})")
    return X_selected, selector


def reduce_with_pca(X: np.ndarray, variance_ratio: float = 0.95) -> tuple:
    """
    Réduit la dimensionnalité par PCA en conservant un pourcentage de variance.
    Retourne (X_reduced, pca_model).
    """
    pca = PCA(n_components=variance_ratio, random_state=RANDOM_STATE)
    X_reduced = pca.fit_transform(X)
    print(f"  PCA ({variance_ratio*100:.0f}% variance): {X.shape[1]} → {X_reduced.shape[1]} composantes")
    return X_reduced, pca


def get_feature_importance_ranking(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcule le score d'information mutuelle de chaque feature.
    Retourne un array de scores (même taille que X.shape[1]).
    Utile pour le reporting et la visualisation.
    """
    scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return scores


class FeatureSelector:
    """
    Pipeline de sélection de features configurable.
    Enchaîne : variance threshold → mutual info → PCA optionnel.
    S'utilise comme un transformer scikit-learn (fit/transform).
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        k_best: int = 150,
        use_pca: bool = False,
        pca_variance: float = 0.95,
    ):
        self.variance_threshold = variance_threshold
        self.k_best = k_best
        self.use_pca = use_pca
        self.pca_variance = pca_variance

        self._var_selector = None
        self._mi_selector = None
        self._pca = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        """Ajuste le pipeline de sélection."""
        print(f"Feature selection ({X.shape[1]} features initiales):")

        # 1. Variance threshold
        X_step, self._var_selector = remove_low_variance(X, self.variance_threshold)

        # 2. Mutual info
        k = min(self.k_best, X_step.shape[1])
        X_step, self._mi_selector = select_by_mutual_info(X_step, y, k=k)

        # 3. PCA optionnel
        if self.use_pca:
            X_step, self._pca = reduce_with_pca(X_step, self.pca_variance)

        print(f"  Résultat final: {X.shape[1]} → {X_step.shape[1]} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applique la sélection ajustée sur de nouvelles données."""
        X_step = self._var_selector.transform(X)
        X_step = self._mi_selector.transform(X_step)
        if self.use_pca and self._pca is not None:
            X_step = self._pca.transform(X_step)
        return X_step

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ajuste et transforme en une seule étape."""
        self.fit(X, y)
        return self.transform(X)
