"""
Métriques d'évaluation alignées sur le challenge AIROGS.
Fix pAUC, ajout bootstrap CI, courbes ROC comparatives.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour éviter les erreurs
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score


# ---------------------------------------------------------------------------
# Métriques de base
# ---------------------------------------------------------------------------

def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.95) -> float:
    """Sensibilité à une spécificité cible (ex: 95%)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    idx = np.where(fpr <= 1 - target_spec)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def partial_auc(y_true: np.ndarray, y_score: np.ndarray, spec_min: float = 0.90, spec_max: float = 1.0) -> float:
    """
    pAUC : aire partielle sous la courbe ROC entre deux spécificités.
    Utilise sklearn.metrics.roc_auc_score(max_fpr=...) pour un calcul correct.
    FIX : l'ancienne implémentation avec np.trapz retournait des valeurs négatives.
    """
    max_fpr = 1 - spec_min  # spec_min=0.9 => max_fpr=0.1
    try:
        return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))
    except ValueError:
        return 0.0


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC standard."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    return float(auc(fpr, tpr))


# ---------------------------------------------------------------------------
# Métriques complètes
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Calcule toutes les métriques AIROGS.
    Retourne un dict avec AUC-ROC, pAUC, sensibilité@95%sp, sensibilité@90%sp.
    """
    return {
        "auc_roc": roc_auc(y_true, y_score),
        "pauc_90_100": partial_auc(y_true, y_score, spec_min=0.90),
        "sens_at_95spec": sensitivity_at_specificity(y_true, y_score, target_spec=0.95),
        "sens_at_90spec": sensitivity_at_specificity(y_true, y_score, target_spec=0.90),
    }


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None):
    """Affiche les métriques principales."""
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    if y_score is not None:
        metrics = compute_all_metrics(y_true, y_score)
        print(f"AUC-ROC:                    {metrics['auc_roc']:.4f}")
        print(f"pAUC (90-100% spec):        {metrics['pauc_90_100']:.4f}")
        print(f"Sensibilité @ 95% spec:     {metrics['sens_at_95spec']:.4f}")
        print(f"Sensibilité @ 90% spec:     {metrics['sens_at_90spec']:.4f}")


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple:
    """
    Calcule l'intervalle de confiance par bootstrap pour une métrique.
    metric_fn: callable(y_true, y_score) -> float.
    Retourne (mean, lower_bound, upper_bound).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_s = y_score[idx]
        # Vérifier qu'on a les deux classes
        if len(np.unique(y_t)) < 2:
            continue
        try:
            scores.append(metric_fn(y_t, y_s))
        except Exception:
            continue

    if not scores:
        return 0.0, 0.0, 0.0

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    mean = float(np.mean(scores))
    return mean, lower, upper


def compute_all_metrics_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Calcule toutes les métriques avec intervalles de confiance à 95%.
    Retourne un dict {metric_name: {"value": ..., "ci_lower": ..., "ci_upper": ...}}.
    """
    metrics_fns = {
        "auc_roc": roc_auc,
        "pauc_90_100": lambda yt, ys: partial_auc(yt, ys, spec_min=0.90),
        "sens_at_95spec": lambda yt, ys: sensitivity_at_specificity(yt, ys, 0.95),
        "sens_at_90spec": lambda yt, ys: sensitivity_at_specificity(yt, ys, 0.90),
    }

    results = {}
    for name, fn in metrics_fns.items():
        value = fn(y_true, y_score)
        mean, lower, upper = bootstrap_confidence_interval(y_true, y_score, fn, n_bootstrap)
        results[name] = {
            "value": value,
            "ci_lower": lower,
            "ci_upper": upper,
        }
        print(f"  {name}: {value:.4f} [{lower:.4f}, {upper:.4f}]")

    return results


# ---------------------------------------------------------------------------
# Courbes ROC comparatives
# ---------------------------------------------------------------------------

def plot_roc_comparison(
    results: dict,
    save_path: str = None,
    title: str = "Courbes ROC — Comparaison des modèles",
    figsize: tuple = (10, 8),
):
    """
    Trace les courbes ROC de plusieurs modèles sur le même graphe.
    results: dict {model_name: {"y_true": array, "y_score": array}}.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        y_true = data["y_true"]
        y_score = data["y_score"]
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")

    # Ligne diagonale (modèle aléatoire)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aléatoire")

    # Ligne verticale à 95% spécificité (FPR=0.05)
    ax.axvline(x=0.05, color="red", linestyle=":", alpha=0.5, label="95% spécificité")

    ax.set_xlabel("Taux de faux positifs (1 - Spécificité)", fontsize=12)
    ax.set_ylabel("Taux de vrais positifs (Sensibilité)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC sauvegardée: {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None,
    save_path: str = None,
    title: str = "Matrice de confusion",
):
    """Affiche la matrice de confusion avec annotations."""
    if labels is None:
        labels = ["NRG", "RG"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        ylabel="Vrai label", xlabel="Prédiction",
        title=title,
    )

    # Annoter les cellules
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
