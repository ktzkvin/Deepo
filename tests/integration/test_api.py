"""
Integration tests for the FastAPI application endpoints.
Ensures routing and metadata consistency.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_api_health_structure():
    """Integration test: Verifies the health check and registry reporting."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["loaded_models"], list)

def test_supported_languages_consistency():
    """Integration test: Ensures all 5 required languages are listed in metadata."""
    response = client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()
    expected = ["en", "fr", "es", "zh", "pt"]
    assert all(lang in data["source"] for lang in expected)
    assert data["defaults"]["source"] == "fr"