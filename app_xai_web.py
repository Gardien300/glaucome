"""
Application web locale (Gradio) pour l'inférence DL + XAI.

Usage :
    1) Entraîner le modèle DL (par ex. efficientnet_v2_m) :
           python -m src.dl_train --config configs/dl_highperf.yaml
    2) Lancer l'interface :
           python app_xai_web.py
    3) Ouvrir le lien local dans le navigateur, déposer une image de fond d'œil.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from src.dl_inference import predict_with_explanations


PROJECT_ROOT = Path(__file__).resolve().parent


def _infer(image: Image.Image):
    if image is None:
        return "Aucune image fournie.", None

    prob_rg, label_str, overlay = predict_with_explanations(image)
    prob_percent = prob_rg * 100.0
    text = f"{label_str} — probabilité RG = {prob_percent:.1f}%"
    return text, overlay


title = "Dépistage glaucome référable — Modèle DL + XAI"
description = (
    "Déposez une image de fond d'œil (AIROGS-like). "
    "Le modèle deep learning prédit la probabilité de glaucome référable (RG) "
    "et affiche une carte Grad-CAM indiquant les zones qui ont le plus contribué à la décision."
)

demo = gr.Interface(
    fn=_infer,
    inputs=gr.Image(type="pil", label="Image de fond d'œil (JPEG/PNG)"),
    outputs=[
        gr.Textbox(label="Interprétation du modèle"),
        gr.Image(label="Heatmap Grad-CAM (XAI)"),
    ],
    title=title,
    description=description,
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()

