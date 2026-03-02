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
    text: str = Field(min_length=1)
    source_lang: str = Field(default="en", min_length=2, max_length=10)
    target_lang: str = Field(default="fr", min_length=2, max_length=10)

class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    elapsed_ms: int

app = FastAPI(title="Deepo API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Chargement du modèle MLflow ---
MODEL_NAME = os.getenv("MODEL_NAME", "deepo-translator")
STAGE = os.getenv("MODEL_STAGE", "Production")

print(f"Loading model {MODEL_NAME} from stage {STAGE}...")
try:
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    translator_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model from MLflow. Error: {e}")
    translator_model = None
# -----------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": translator_model is not None}

@app.get("/api/languages")
def languages():
    return {
        # On ajoute en, fr, es, zh (chinois), ar
        "source": ["en", "fr", "es", "zh", "ar"], 
        "target": ["en", "fr", "es", "zh", "ar"],
        "defaults": {"source": "fr", "target": "en"},
    }

@app.post("/api/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest):
    if not translator_model:
        raise HTTPException(status_code=503, detail="Model not loaded from MLflow Registry")
        
    t0 = time.perf_counter()
    
    # LE NOUVEAU MAPPING AVEC ANGLAIS ET CHINOIS
    lang_map = {"fr": "fra", "es": "spa", "zh": "cmn", "ar": "ara", "en": "eng"}
    model_target_lang = lang_map.get(payload.target_lang, "eng")
    
    # On force la majuscule comme dans Colab pour l'aider !
    text_to_translate = payload.text.strip().capitalize()
    
    model_input = [{"text": text_to_translate, "target_lang": model_target_lang}]
    predictions = translator_model.predict(model_input)
    translated = predictions[0]
    
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return TranslateResponse(
        translated_text=translated,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        elapsed_ms=elapsed_ms,
    )

repo_root = Path(__file__).resolve().parents[2]
frontend_dir = repo_root / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")