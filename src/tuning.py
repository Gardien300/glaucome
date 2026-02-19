"""
Hyperparameter tuning pour les classifieurs ML.
Espaces de recherche prédéfinis pour RF, XGBoost, LR.
Utilise RandomizedSearchCV avec StratifiedKFold.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Espaces de recherche
# ---------------------------------------------------------------------------

PARAM_SPACES = {
    "rf": {
        "n_estimators": [100, 200, 300, 500, 800, 1000],
        "max_depth": [10, 15, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    },
    "xgb": {
        "n_estimators": [100, 200, 300, 500, 800, 1000],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "scale_pos_weight": [1, 10, 29],
    },
    "lr": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["saga"],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "class_weight": ["balanced"],
        "max_iter": [2000],
    },
}


# ---------------------------------------------------------------------------
# Fonctions de tuning
# ---------------------------------------------------------------------------

def get_base_classifier(name: str):
    """Retourne un classifieur de base non configuré pour le tuning."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if name == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE)
    elif name == "lr":
        return LogisticRegression(random_state=RANDOM_STATE)
    elif name == "xgb":
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        except ImportError:
            raise ImportError("xgboost requis pour le tuning XGB")
    else:
        raise ValueError(f"Classifieur inconnu : {name}")


def tune_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "xgb",
    cv: int = 5,
    n_iter: int = 50,
    scoring: str = "roc_auc",
    verbose: int = 1,
) -> dict:
    """
    Hyperparameter tuning via RandomizedSearchCV.

    Retourne un dict avec :
    - 'best_model': le meilleur modèle entraîné
    - 'best_params': les meilleurs hyperparamètres
    - 'best_score': le meilleur score CV
    - 'cv_results': résultats détaillés du CV
    """
    clf = get_base_classifier(model_name)
    param_space = PARAM_SPACES.get(model_name)
    if param_space is None:
        raise ValueError(f"Pas d'espace de recherche pour {model_name}. Dispo: {list(PARAM_SPACES.keys())}")

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_strategy,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )

    print(f"Tuning {model_name.upper()} ({n_iter} itérations, {cv}-fold CV, scoring={scoring})...")
    search.fit(X, y)

    print(f"  Meilleur score: {search.best_score_:.4f}")
    print(f"  Meilleurs params: {search.best_params_}")

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": search.cv_results_,
    }


def tune_all_models(
    X: np.ndarray,
    y: np.ndarray,
    models: list = None,
    cv: int = 5,
    n_iter: int = 50,
    scoring: str = "roc_auc",
) -> dict:
    """
    Tune tous les modèles et retourne le meilleur.
    Retourne un dict {model_name: tune_result} + 'best_overall'.
    """
    if models is None:
        models = ["rf", "xgb", "lr"]

    results = {}
    best_name = None
    best_score = -1

    for name in models:
        print(f"\n{'='*60}")
        try:
            result = tune_model(X, y, name, cv=cv, n_iter=n_iter, scoring=scoring)
            results[name] = result
            if result["best_score"] > best_score:
                best_score = result["best_score"]
                best_name = name
        except Exception as e:
            print(f"  Erreur tuning {name}: {e}")
            continue

    if best_name:
        print(f"\n{'='*60}")
        print(f"Meilleur modèle global: {best_name.upper()} (score={best_score:.4f})")
        results["best_overall"] = best_name

    return results
