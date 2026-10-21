import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger("riceguard.model")

MODEL_PATH = Path(__file__).parent / "riceguard_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_image(image: Image.Image) -> Image.Image:
    """Ensure image is in RGB format, cleanly handling alpha and palette transparency."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        alpha = rgba.split()[3]
        background.paste(rgba, mask=alpha)
        return background
    return image.convert("RGB")


class RiceGuardClassifier:
    """Manages model loading, state, preprocessing, and inference."""

    def __init__(self, model_path: Path = MODEL_PATH, device: torch.device = DEVICE):
        self.model_path = model_path
        self.device = device
        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = []
        self.image_size: int = 224
        self.architecture: str = "mobilenet_v3_small"
        self.preprocess: Optional[transforms.Compose] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

    def load(self) -> Tuple[nn.Module, List[str], transforms.Compose]:
        """Loads model checkpoint from disk and prepares inference pipeline."""
        if not self.model_path.exists():
            error_msg = f"Model checkpoint not found: {self.model_path}"
            self.load_error = error_msg
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # First attempt secure weights_only loading
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception as exc:
            logger.warning(
                "Secure weights_only=True loading failed (%s). Retrying with weights_only=False...",
                exc,
            )
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.class_names = checkpoint.get("class_names", [])
        self.image_size = checkpoint.get("image_size", 224)
        self.architecture = checkpoint.get("architecture", "mobilenet_v3_small")

        if not self.class_names:
            raise ValueError("Checkpoint does not contain valid 'class_names'.")

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(self.class_names))

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.model = model
        self.is_loaded = True
        self.load_error = None
        logger.info(
            "RiceGuard AI model successfully loaded on %s (%d classes, size %d).",
            self.device,
            len(self.class_names),
            self.image_size,
        )
        return self.model, self.class_names, self.preprocess

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Runs inference on a PIL Image and returns predictions."""
        if not self.is_loaded or self.model is None or self.preprocess is None:
            raise RuntimeError("Model is not loaded. Call load() before predicting.")

        rgb_image = prepare_image(image)
        tensor = self.preprocess(rgb_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_k = min(3, len(self.class_names))
        scores, indices = torch.topk(probabilities, k=top_k, largest=True, sorted=True)

        predictions = [
            {
                "label": self.class_names[index.item()],
                "confidence": round(float(score.item()), 4),
            }
            for score, index in zip(scores, indices)
        ]

        return {
            "predicted_class": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions,
        }


# Singleton classifier instance
classifier = RiceGuardClassifier()

try:
    classifier.load()
except Exception as e:
    logger.warning("Initial model loading failed: %s", e)

# Module-level exports for backwards compatibility
model = classifier.model
class_names = classifier.class_names
preprocess = classifier.preprocess


def load_model(model_path: Path = MODEL_PATH):
    """Backwards-compatible load_model function."""
    classifier.model_path = model_path
    return classifier.load()


def predict_image(image: Image.Image) -> Dict[str, Any]:
    """Backwards-compatible predict_image function."""
    return classifier.predict(image)
