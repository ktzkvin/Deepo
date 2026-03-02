"""
Deepo API Main Application.
Handles multi-model routing, pivot translation, and FastAPI endpoints.
"""
from __future__ import annotations

import time
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import mlflow.pyfunc


class TranslateRequest(BaseModel):
    """Data validation model for incoming translation requests."""
    text: str = Field(min_length=1)
    source_lang: str = Field(default="en", min_length=2, max_length=10)
    target_lang: str = Field(default="fr", min_length=2, max_length=10)


class TranslateResponse(BaseModel):
    """Data validation model for outbound translation responses."""
    translated_text: str
    source_lang: str
    target_lang: str
    elapsed_ms: int


app = FastAPI(title="Deepo API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# --- MLflow Models Registry Initialization ---
STAGE = os.getenv("MODEL_STAGE", "Production")
# Ajout du portugais (pt: por) et restriction stricte aux 4 langues + pivot
SUPPORTED_LANGS = {"fr": "fra", "zh": "cmn", "es": "spa", "pt": "por"} 

models_registry = {}

for lang_code, model_suffix in SUPPORTED_LANGS.items():
    model_name = f"deepo-translator-{model_suffix}"
    print(f"Attempting to load model {model_name} from stage {STAGE}...")
    try:
        model_uri = f"models:/{model_name}/{STAGE}"
        models_registry[model_suffix] = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Warning: Could not load model {model_name}. Error: {e}")
# ---------------------------------------------


@app.get("/api/health")
def health():
    """Health check endpoint to verify API and model registry status."""
    return {"status": "ok", "loaded_models": list(models_registry.keys())}


@app.get("/api/languages")
def languages():
    """Returns the list of supported languages for the frontend."""
    return {
        "source": ["en", "fr", "es", "zh", "pt"], 
        "target": ["en", "fr", "es", "zh", "pt"],
        "defaults": {"source": "fr", "target": "en"},
    }


@app.post("/api/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest):
    """
    Main translation endpoint. 
    Implements multi-model routing and pivot language (English) logic.
    """
    t0 = time.perf_counter()
    
    lang_map = {"fr": "fra", "es": "spa", "zh": "cmn", "pt": "por", "en": "eng"}
    source_code = lang_map.get(payload.source_lang)
    target_code = lang_map.get(payload.target_lang)

    if not source_code or not target_code:
        raise HTTPException(status_code=400, detail="Unsupported language configuration.")

    text_to_translate = payload.text.strip().capitalize()
    translated = ""

    # Case 1: Translation TO English (e.g., FR -> EN)
    if payload.target_lang == "en":
        if source_code not in models_registry:
            raise HTTPException(status_code=503, detail=f"Model for source language '{payload.source_lang}' is currently unavailable.")
        
        translator_model = models_registry[source_code]
        model_input = [{"text": text_to_translate, "target_lang": "eng"}]
        translated = translator_model.predict(model_input)[0]

    # Case 2: Translation FROM English (e.g., EN -> ZH)
    elif payload.source_lang == "en":
        if target_code not in models_registry:
            raise HTTPException(status_code=503, detail=f"Model for target language '{payload.target_lang}' is currently unavailable.")
        
        translator_model = models_registry[target_code]
        model_input = [{"text": text_to_translate, "target_lang": target_code}]
        translated = translator_model.predict(model_input)[0]

    # Case 3: Pivot Translation via English (e.g., FR -> EN -> ZH)
    else:
        if source_code not in models_registry or target_code not in models_registry:
            raise HTTPException(status_code=503, detail="Both source and target expert models are required for pivot translation.")
        
        # Step 3.1: Source -> English
        model_1 = models_registry[source_code]
        input_1 = [{"text": text_to_translate, "target_lang": "eng"}]
        intermediate_en = model_1.predict(input_1)[0]
        
        # Step 3.2: English -> Target
        model_2 = models_registry[target_code]
        input_2 = [{"text": intermediate_en.capitalize(), "target_lang": target_code}]
        translated = model_2.predict(input_2)[0]

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return TranslateResponse(
        translated_text=translated,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        elapsed_ms=elapsed_ms,
    )


# --- Frontend Static Files Mounting ---
repo_root = Path(__file__).resolve().parents[2]
frontend_dir = repo_root / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")