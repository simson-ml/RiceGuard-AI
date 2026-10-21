import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Set

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from model import classifier, predict_image, class_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("riceguard.api")

# Allowed MIME types
SUPPORTED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/pjpeg",
    "image/x-png",
}

# Supported PIL format names
SUPPORTED_IMAGE_FORMATS: Set[str] = {"JPEG", "PNG", "WEBP"}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to ensure model is verified on application startup."""
    logger.info("Initializing RiceGuard AI API...")
    if not classifier.is_loaded:
        try:
            classifier.load()
            logger.info("Model loaded successfully on startup.")
        except Exception as exc:
            logger.error("Failed to load model on startup: %s", exc)
    else:
        logger.info("Model was already loaded.")
    yield
    logger.info("Shutting down RiceGuard AI API...")


app = FastAPI(
    title="RiceGuard AI API",
    description="Automated early detection of rice leaf diseases using deep learning (MobileNetV3-Small).",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS origins
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "https://riceguard-ai.netlify.app",
]

env_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

allowed_origins = list(dict.fromkeys(DEFAULT_ORIGINS + env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https:\/\/.*--riceguard-ai\.netlify\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    """Returns general service information."""
    return {
        "service": "RiceGuard AI API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "classes": "/classes",
    }


@app.get("/health", tags=["Health"])
def health(response: Response):
    """Reports genuine service and deep learning model health."""
    if not classifier.is_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": classifier.load_error or "Model is not loaded",
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "classes_count": len(classifier.class_names),
        "architecture": classifier.architecture,
        "device": str(classifier.device),
    }


@app.get("/classes", tags=["Model"])
def get_classes():
    """Returns available crop disease and healthy class names."""
    return {
        "classes": classifier.class_names or class_names,
        "count": len(classifier.class_names or class_names),
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """Accepts an uploaded rice leaf image and returns disease predictions with confidence scores."""
    raw_content_type = file.content_type or ""
    clean_content_type = raw_content_type.split(";")[0].strip().lower()

    if clean_content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPG, PNG, or WEBP image",
        )

    # Read up to MAX_FILE_SIZE + 1 to detect oversized files without unbounded memory consumption
    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Image must be 10 MB or smaller",
        )

    try:
        # Step 1: verify file integrity
        image_stream = BytesIO(contents)
        verify_img = Image.open(image_stream)
        img_format = verify_img.format
        verify_img.verify()

        if img_format not in SUPPORTED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported image format: {img_format}. Please use JPG, PNG, or WEBP.",
            )

        # Step 2: reopen to load raw image data after verification
        image = Image.open(BytesIO(contents))
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Invalid or corrupted image uploaded: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted image",
        )

    if not classifier.is_loaded:
        logger.error("Inference requested while model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable",
        )

    try:
        return classifier.predict(image)
    except Exception as exc:
        logger.exception("Inference failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        )
