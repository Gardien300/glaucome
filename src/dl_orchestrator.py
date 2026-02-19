"""
Orchestrateur d'expériences deep learning pour le projet glaucome.

Permet de lancer plusieurs expériences DL à la suite sur un pod GPU,
sans intervention manuelle entre les runs.

Usage typique (depuis la racine du projet) :

    # Exécuter le plan par défaut (baseline + highperf)
    python -m src.dl_orchestrator

    # Ou passer explicitement une liste de configs
    python -m src.dl_orchestrator --configs configs/dl_baseline.yaml configs/dl_highperf.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from .config import PROJECT_ROOT
from .dl_train import load_config, train_from_config


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_experiments(config_paths: List[Path]) -> None:
    """
    Lance séquentiellement une série de fichiers de config DL.

    Args:
        config_paths: liste de chemins vers des fichiers YAML (relatifs ou absolus).
    """
    print(f"[{_now_str()}] Début de l'orchestration DL.")

    for cfg_path in config_paths:
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path

        if not cfg_path.exists():
            print(f"[{_now_str()}] ⚠️  Config introuvable : {cfg_path} — on saute ce run.")
            continue

        print(f"\n[{_now_str()}] === Lancement expérience DL avec config : {cfg_path} ===")
        cfg = load_config(cfg_path)
        try:
            train_from_config(cfg)
        except Exception as e:
            # On loggue l'erreur mais on continue les expériences suivantes
            print(f"[{_now_str()}] ❌ Erreur pendant l'entraînement avec {cfg_path}: {e}")
        else:
            print(f"[{_now_str()}] ✅ Fin de l'expérience DL pour {cfg_path}")

    print(f"\n[{_now_str()}] Orchestration DL terminée.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrateur d'expériences deep learning (glaucome)."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=str,
        help=(
            "Liste de fichiers de config YAML à lancer séquentiellement "
            "(ex: configs/dl_baseline.yaml configs/dl_highperf.yaml). "
            "Si non fourni, un plan par défaut sera utilisé."
        ),
    )
    args = parser.parse_args()

    if args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        # Plan par défaut : baseline DL puis modèle produit high-perf
        config_paths = [
            Path("configs/dl_baseline.yaml"),
            Path("configs/dl_highperf.yaml"),
        ]

    run_experiments(config_paths)


if __name__ == "__main__":
    main()

