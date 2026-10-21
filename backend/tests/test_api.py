import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from main import app, MAX_FILE_SIZE


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def create_test_image(format="JPEG", size=(100, 100), color=(34, 139, 34)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "RiceGuard AI API"
    assert data["status"] == "online"
    assert "/docs" in data["docs"]


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["classes_count"] == 10
    assert data["architecture"] == "mobilenet_v3_small"


def test_classes_endpoint(client):
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 10
    assert "bacterial_leaf_blight" in data["classes"]
    assert "normal" in data["classes"]


def test_predict_valid_jpeg(client):
    image_bytes = create_test_image("JPEG")
    response = client.post(
        "/predict",
        files={"file": ("leaf.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "top_predictions" in data
    assert isinstance(data["confidence"], float)
    assert len(data["top_predictions"]) == 3


def test_predict_valid_png(client):
    image_bytes = create_test_image("PNG")
    response = client.post(
        "/predict",
        files={"file": ("leaf.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert len(data["top_predictions"]) == 3


def test_predict_valid_webp(client):
    image_bytes = create_test_image("WEBP")
    response = client.post(
        "/predict",
        files={"file": ("leaf.webp", image_bytes, "image/webp")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data


def test_predict_empty_file(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_invalid_image_bytes(client):
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", b"This is not a real image file", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower()


def test_predict_unsupported_media_type(client):
    response = client.post(
        "/predict",
        files={"file": ("data.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 415
    assert "upload a jpg, png, or webp" in response.json()["detail"].lower()


def test_predict_oversized_file(client):
    # Create fake oversized bytes (10MB + 1KB)
    large_bytes = b"0" * (MAX_FILE_SIZE + 1024)
    response = client.post(
        "/predict",
        files={"file": ("large.jpg", large_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "10 mb" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://riceguard-ai.netlify.app",
        "https://deploy-preview-123--riceguard-ai.netlify.app",
    ],
)
def test_cors_preflight_and_headers(client, origin):
    # Test preflight OPTIONS request
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/predict", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin

    # Test GET request with Origin
    get_res = client.get("/health", headers={"Origin": origin})
    assert get_res.status_code == 200
    assert get_res.headers.get("access-control-allow-origin") == origin
