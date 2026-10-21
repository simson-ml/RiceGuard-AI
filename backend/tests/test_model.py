import io
from pathlib import Path
import pytest
from PIL import Image

from model import RiceGuardClassifier, prepare_image, classifier, MODEL_PATH


def test_classifier_is_loaded():
    assert classifier.is_loaded is True
    assert len(classifier.class_names) == 10
    assert classifier.image_size == 224
    assert classifier.architecture == "mobilenet_v3_small"
    assert "normal" in classifier.class_names


def test_prepare_image_rgb():
    img = Image.new("RGB", (100, 100), color=(34, 139, 34))
    prepared = prepare_image(img)
    assert prepared.mode == "RGB"
    assert prepared.size == (100, 100)


def test_prepare_image_rgba_transparency():
    # RGBA image with transparent background
    img = Image.new("RGBA", (100, 100), color=(0, 0, 0, 0))
    prepared = prepare_image(img)
    assert prepared.mode == "RGB"
    # Transparent area should become white
    assert prepared.getpixel((0, 0)) == (255, 255, 255)


def test_prediction_output_structure():
    test_img = Image.new("RGB", (224, 224), color=(50, 150, 50))
    result = classifier.predict(test_img)

    assert "predicted_class" in result
    assert "confidence" in result
    assert "top_predictions" in result

    assert isinstance(result["predicted_class"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["top_predictions"], list)

    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 3

    # Check top prediction matches
    assert result["predicted_class"] == result["top_predictions"][0]["label"]
    assert result["confidence"] == result["top_predictions"][0]["confidence"]

    # Check descending order of confidence
    confs = [p["confidence"] for p in result["top_predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_prediction_on_real_asset():
    hero_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "src" / "assets" / "hero.png"
    if hero_path.exists():
        im = Image.open(hero_path)
        res = classifier.predict(im)
        assert res["predicted_class"] in classifier.class_names
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["top_predictions"]) == 3


def test_unloaded_model_raises_runtime_error():
    empty_clf = RiceGuardClassifier(model_path=Path("nonexistent.pth"))
    assert empty_clf.is_loaded is False
    with pytest.raises(RuntimeError, match="Model is not loaded"):
        test_img = Image.new("RGB", (224, 224))
        empty_clf.predict(test_img)


def test_nonexistent_checkpoint_raises_file_not_found():
    bad_clf = RiceGuardClassifier(model_path=Path("missing_file.pth"))
    with pytest.raises(FileNotFoundError):
        bad_clf.load()
