"""
Pipeline ML : SMOTE (dans le CV), ensembles, calibration, évaluation cross-validée.
Tout en ML classique (pas de deep learning).
Imports lourds chargés à la demande pour éviter ~1–2 min au démarrage.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer

from .config import RANDOM_STATE

_xgb = None


# ---------------------------------------------------------------------------
# Imports lazy
# ---------------------------------------------------------------------------

def _get_xgb():
    global _xgb
    if _xgb is None:
        try:
            import xgboost as xgb
            _xgb = xgb
        except ImportError:
            _xgb = False
    return _xgb


def _get_imblearn_pipeline():
    """Retourne la classe Pipeline d'imblearn (SMOTE-safe)."""
    try:
        from imblearn.pipeline import Pipeline
        return Pipeline
    except ImportError:
        raise ImportError("imbalanced-learn requis (pip install imbalanced-learn)")


def _get_smote_variants():
    """Retourne les classes de resampling disponibles."""
    variants = {}
    try:
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.combine import SMOTETomek
        variants["smote"] = SMOTE
        variants["adasyn"] = ADASYN
        variants["borderline"] = BorderlineSMOTE
        variants["smote_tomek"] = SMOTETomek
    except ImportError:
        pass
    return variants


# ---------------------------------------------------------------------------
# Classifieurs
# ---------------------------------------------------------------------------

def get_classifier(name: str, class_weight: str = "balanced", scale_pos_weight: int = None):
    """
    Retourne un classifieur non entraîné.
    name: 'rf', 'lr', 'xgb'.
    scale_pos_weight: pour XGBoost sans SMOTE, mettre ~29 (ratio NRG/RG).
    """
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=15,
            class_weight=class_weight, random_state=RANDOM_STATE,
        )
    if name == "lr":
        return LogisticRegression(
            max_iter=2000, class_weight=class_weight,
            solver="saga", random_state=RANDOM_STATE,
        )
    if name == "xgb":
        xgb = _get_xgb()
        if xgb is not None and xgb is not False:
            spw = scale_pos_weight if scale_pos_weight else 1
            return xgb.XGBClassifier(
                n_estimators=100, max_depth=6,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )
        raise ImportError("xgboost requis (pip install xgboost)")
    raise ValueError(f"name doit être 'rf', 'lr' ou 'xgb', reçu: '{name}'")


# ---------------------------------------------------------------------------
# SMOTE helpers (standalone, rétrocompatible)
# ---------------------------------------------------------------------------

def get_smote(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5):
    """
    Applique SMOTE sur (X, y). Rétrocompatible.
    ATTENTION : à utiliser UNIQUEMENT hors cross-validation.
    Pour le CV, utiliser create_pipeline() qui encapsule SMOTE dans les folds.
    """
    variants = _get_smote_variants()
    SMOTE = variants.get("smote")
    if SMOTE is None:
        return X, y
    n_min = int(np.sum(y == 1))
    if n_min < 2:
        return X, y
    k = min(k_neighbors, n_min - 1)
    if k < 1:
        return X, y
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    return smote.fit_resample(X, y)


# ---------------------------------------------------------------------------
# Pipeline imblearn (SMOTE dans le CV — pas de fuite de données)
# ---------------------------------------------------------------------------

def create_pipeline(
    model_name: str = "xgb",
    resample_strategy: str = "smote",
    use_scaler: bool = True,
    class_weight: str = "balanced",
    scale_pos_weight: int = None,
    clf_params: dict = None,
):
    """
    Crée un pipeline imblearn : resampling → scaler → classifieur.
    Le resampling est appliqué UNIQUEMENT sur le train de chaque fold CV.

    resample_strategy: 'smote', 'adasyn', 'borderline', 'smote_tomek', ou None.
    clf_params: paramètres supplémentaires pour le classifieur (ex: tuning).
    """
    Pipeline = _get_imblearn_pipeline()

    steps = []

    # 1. Resampling (optionnel)
    if resample_strategy:
        variants = _get_smote_variants()
        ResamplerClass = variants.get(resample_strategy)
        if ResamplerClass is not None:
            if resample_strategy in ("smote", "borderline"):
                resampler = ResamplerClass(random_state=RANDOM_STATE, k_neighbors=5)
            elif resample_strategy == "adasyn":
                resampler = ResamplerClass(random_state=RANDOM_STATE, n_neighbors=5)
            else:
                resampler = ResamplerClass(random_state=RANDOM_STATE)
            steps.append(("resampler", resampler))

    # 2. Scaler
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    # 3. Classifieur
    clf = get_classifier(model_name, class_weight=class_weight, scale_pos_weight=scale_pos_weight)
    if clf_params:
        clf.set_params(**clf_params)
    steps.append(("classifier", clf))

    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Évaluation cross-validée
# ---------------------------------------------------------------------------

def _sensitivity_at_95spec(y_true, y_score):
    """Scorer pour la sensibilité à 95% spécificité."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    idx = np.where(fpr <= 0.05)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


sens_at_95spec_scorer = make_scorer(
    _sensitivity_at_95spec, needs_proba=True, response_method="predict_proba",
)


def evaluate_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    pipeline,
    cv: int = 3,
    scoring: dict = None,
) -> dict:
    """
    Évaluation cross-validée avec le pipeline imblearn.
    SMOTE est appliqué à l'intérieur de chaque fold (pas de fuite de données).

    Retourne un dict avec mean et std de chaque métrique.
    """
    if scoring is None:
        scoring = {
            "roc_auc": "roc_auc",
            "sens_at_95spec": sens_at_95spec_scorer,
        }

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    print(f"Cross-validation {cv}-fold...")
    results = cross_validate(
        pipeline, X, y,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    summary = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        if key in results:
            scores = results[key]
            summary[metric_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores.tolist(),
            }
            print(f"  {metric_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return summary


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def create_voting_ensemble(models: dict = None, voting: str = "soft") -> VotingClassifier:
    """
    Crée un VotingClassifier à partir de modèles entraînés ou par défaut.
    models: dict {name: estimator}. Si None, crée RF + XGB + LR par défaut.
    """
    if models is None:
        models = {
            "rf": get_classifier("rf"),
            "xgb": get_classifier("xgb", scale_pos_weight=29),
            "lr": get_classifier("lr"),
        }

    estimators = [(name, clf) for name, clf in models.items()]
    return VotingClassifier(estimators=estimators, voting=voting)


def create_stacking_ensemble(
    base_models: dict = None,
    final_estimator=None,
) -> StackingClassifier:
    """
    Crée un StackingClassifier.
    base_models: dict {name: estimator} pour les modèles de base.
    final_estimator: modèle final (par défaut: LogisticRegression balanced).
    """
    if base_models is None:
        base_models = {
            "rf": get_classifier("rf"),
            "xgb": get_classifier("xgb", scale_pos_weight=29),
        }

    if final_estimator is None:
        final_estimator = LogisticRegression(
            class_weight="balanced", max_iter=2000,
            solver="saga", random_state=RANDOM_STATE,
        )

    estimators = [(name, clf) for name, clf in base_models.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        passthrough=False,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model(clf, X_val: np.ndarray, y_val: np.ndarray, method: str = "isotonic"):
    """
    Calibre les probabilités d'un modèle déjà entraîné.
    Améliore la fiabilité du seuillage (sensibilité@95%sp).
    clf: modèle déjà fit.
    """
    calibrated = CalibratedClassifierCV(clf, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


# ---------------------------------------------------------------------------
# Seuillage (rétrocompatible)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def threshold_predict(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Prédiction binaire à partir des probas (classe 1 si proba[:, 1] >= threshold)."""
    return (proba[:, 1] >= threshold).astype(int)


def find_threshold_for_specificity(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.95) -> float:
    """Retourne le seuil tel que la spécificité >= target_spec."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return 0.5
    best_i = idx[np.argmax(tpr[idx])]
    return float(thresholds[best_i])
