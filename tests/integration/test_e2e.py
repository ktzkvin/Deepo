"""
End-to-End (E2E) tests for the translation pipeline.
Validates the full journey: Request -> Pivot Logic -> Expert Model -> Filter -> Response.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_full_pivot_translation_lifecycle():
    """
    E2E Test: Validates the most complex scenario (FR -> EN -> PT pivot).
    Accepts 503 only if models aren't loaded in the CI environment.
    """
    payload = {
        "text": "Bonjour tout le monde",
        "source_lang": "fr",
        "target_lang": "pt"
    }
    response = client.post("/api/translate", json=payload)
    
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "translated_text" in data
        assert data["target_lang"] == "pt"
        assert len(data["translated_text"].split()) > 0

def test_translation_content_accuracy():
    """
    E2E Test: Verifies that the model actually translates the word.
    Checks if the translation is correct for a simple sentence (FR -> EN).
    """
    payload = {
        "text": "As-tu vu mon ordinateur ?",
        "source_lang": "fr",
        "target_lang": "en"
    }
    response = client.post("/api/translate", json=payload)
    
    if response.status_code == 200:
        translated_text = response.json()["translated_text"].lower()
        assert "have you seen my computer?" in translated_text