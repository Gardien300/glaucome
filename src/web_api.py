from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from PIL import Image, UnidentifiedImageError

from .config import PROJECT_ROOT
from .web_inference import InferenceResult, get_inference_service


class DoctorLabel(str, Enum):
    NRG = "NRG"
    RG = "RG"
    UNCERTAIN = "UNCERTAIN"


class EyeSide(str, Enum):
    OD = "OD"  # œil droit
    OG = "OG"  # œil gauche
    OU = "OU"  # les deux / non précisé


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    timestamp_utc: str
    doctor_id: str
    retinograph: Optional[str] = None
    eye_side: Optional[EyeSide] = None
    doctor_label: DoctorLabel
    doctor_comment: Optional[str] = None
    image_path: str
    heatmap_path: Optional[str] = None
    probability_rg: float
    predicted_class: str
    threshold: float
    model_version: str


def _ensure_access_token(x_access_token: Optional[str] = Header(default=None)) -> None:
    """
    Garde-fou simple : si ACCESS_TOKEN est défini, il doit être présent dans l'en-tête.
    """
    required = os.environ.get("ACCESS_TOKEN")
    if required and x_access_token != required:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token")


def _get_uploads_dir() -> Path:
    base = os.environ.get("UPLOADS_DIR")
    if base:
        return Path(base)
    return PROJECT_ROOT / "web_uploads"


def _get_logs_dir() -> Path:
    base = os.environ.get("LOGS_DIR")
    if base:
        return Path(base)
    return PROJECT_ROOT / "data" / "web_logs"


def _append_to_log_csv(row: dict) -> Path:
    logs_dir = _get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    date_str = row["timestamp_utc"][:10].replace("-", "")
    log_path = logs_dir / f"predictions_{date_str}.csv"

    file_exists = log_path.exists()
    fieldnames = list(row.keys())
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return log_path


app = FastAPI(
    title="Plateforme web — dépistage du glaucome référable",
    description="API FastAPI pour l'upload de fond d'œil, l'inférence DL et le logging des prédictions.",
)

# CORS large pour le développement (React en local).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """
    Précharge le modèle au démarrage du serveur.
    """
    _ = get_inference_service()


@app.get("/health")
def health_check():
    service = get_inference_service()
    return {
        "status": "ok",
        "model_version": service.model_version,
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(_ensure_access_token)])
async def predict_endpoint(
    doctor_id: str = Form(...),
    doctor_label: DoctorLabel = Form(...),
    eye_side: Optional[EyeSide] = Form(None),
    retinograph: Optional[str] = Form(None),
    doctor_comment: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    # Vérification basique du type de fichier
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier doit être une image (JPEG/PNG).",
        )

    uploads_dir = _get_uploads_dir()
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    dest_dir = uploads_dir / today
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Nom de fichier anonymisé
    suffix = Path(file.filename or "").suffix or ".jpg"
    image_id = f"{uuid4().hex}{suffix}"
    image_path = dest_dir / image_id

    # Sauvegarde du fichier brut
    content = await file.read()
    with image_path.open("wb") as f:
        f.write(content)

    # Chargement PIL pour l'inférence
    try:
        with Image.open(image_path) as img:
            pil_img = img.convert("RGB")
    except UnidentifiedImageError:
        image_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de lire l'image fournie.",
        )

    service = get_inference_service()
    result: InferenceResult
    overlay = None
    try:
        result, overlay = service.predict_with_xai(pil_img)  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback sans XAI si non disponible
        result = service.predict(pil_img)

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    rel_image_path = image_path.relative_to(PROJECT_ROOT).as_posix()

    # Sauvegarde éventuelle de l'overlay Grad-CAM
    rel_heatmap_path: Optional[str] = None
    if overlay is not None:
        heatmap_name = f"{image_id.rsplit('.', 1)[0]}_cam.png"
        heatmap_path = dest_dir / heatmap_name
        from PIL import Image as PilImage

        PilImage.fromarray(overlay).save(heatmap_path)
        rel_heatmap_path = heatmap_path.relative_to(PROJECT_ROOT).as_posix()

    log_row = {
        "timestamp_utc": timestamp_utc,
        "doctor_id": doctor_id,
        "retinograph": (retinograph or "").strip(),
        "image_path": rel_image_path,
        "eye_side": eye_side.value if eye_side is not None else "",
        "doctor_label": doctor_label.value,
        "doctor_comment": (doctor_comment or "").strip(),
        "model_probability_rg": result.probability_rg,
        "model_predicted_class": result.predicted_class,
        "model_threshold": result.threshold,
        "model_version": result.model_version,
        "heatmap_path": rel_heatmap_path or "",
    }
    _append_to_log_csv(log_row)

    response = PredictionResponse(
        timestamp_utc=timestamp_utc,
        doctor_id=doctor_id,
        retinograph=retinograph,
        eye_side=eye_side,
        doctor_label=doctor_label,
        doctor_comment=doctor_comment,
        image_path=rel_image_path,
        heatmap_path=rel_heatmap_path,
        probability_rg=result.probability_rg,
        predicted_class=result.predicted_class,
        threshold=result.threshold,
        model_version=result.model_version,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())


# Servir les images uploadées (web_uploads/...) pour la heatmap Grad-CAM
uploads_root = _get_uploads_dir()
uploads_root.mkdir(parents=True, exist_ok=True)
app.mount(
    "/web_uploads",
    StaticFiles(directory=uploads_root, html=False),
    name="web_uploads",
)

# Optionnel : servir le build React si présent (web-frontend/dist)
frontend_dist = PROJECT_ROOT / "web-frontend" / "dist"
if frontend_dist.exists():
    app.mount(
        "/",
        StaticFiles(directory=frontend_dist, html=True),
        name="frontend",
    )

