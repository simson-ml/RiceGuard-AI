from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path(__file__).parent / "riceguard_model.pth"

DEVICE = torch.device("cpu")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    # Load only a checkpoint that you trust.
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return model, class_names, preprocess


model, class_names, preprocess = load_model()


def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_k = min(3, len(class_names))
    scores, indices = torch.topk(probabilities, k=top_k)

    predictions = [
        {"label": class_names[index.item()], "confidence": round(score.item(), 4)}
        for score, index in zip(scores, indices)
    ]

    return {
        "predicted_class": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
    }
