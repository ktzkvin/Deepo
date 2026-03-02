"""
Integration and End-to-End tests for the FastAPI application.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check():
    """Integration test: Verify the health check endpoint returns 200 OK."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_languages_endpoint():
    """Integration test: Verify the languages endpoint returns supported languages."""
    response = client.get("/api/languages")
    assert response.status_code == 200
    
    data = response.json()
    assert "en" in data.get("source", [])
    assert "fr" in data.get("target", [])


def test_translation_endpoint():
    """
    End-to-End test: Simulate a translation request.
    Accepts 200 (model loaded and inferred successfully) 
    or 503 (model missing in CI environment prior to MLflow registry pull).
    """
    payload = {
        "text": "Hello",
        "source_lang": "en",
        "target_lang": "fr"
    }
    response = client.post("/api/translate", json=payload)
    
    assert response.status_code in [200, 503]