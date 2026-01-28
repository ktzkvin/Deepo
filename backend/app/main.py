from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    source_lang: str = Field(default="auto", min_length=2, max_length=10)
    target_lang: str = Field(default="fr", min_length=2, max_length=10)
    mode: Literal["lorem"] = "lorem"


class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    mode: str
    elapsed_ms: int


def lorem_from_text(text: str, min_chars: int = 140, max_chars: int = 1400) -> str:
    base = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor.",
        "Cras elementum ultrices diam. Maecenas ligula massa, varius a, semper congue, euismod non, mi. Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, non fermentum diam nisl sit amet erat.",
        "Duis semper. Duis arcu massa, scelerisque vitae, consequat in, pretium a, enim. Pellentesque congue.",
        "Ut in risus volutpat libero pharetra tempor. Cras vestibulum bibendum augue. Praesent egestas leo in pede. Praesent blandit odio eu enim.",
        "Pellentesque sed dui ut augue blandit sodales. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Aliquam nibh.",
        "Mauris ac mauris sed pede pellentesque fermentum. Maecenas adipiscing ante non diam sodales hendrerit.",
    ]
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    start = h % len(base)
    desired = max(min_chars, min(max_chars, max(220, len(text) * 2)))
    out = []
    i = 0
    while len(" ".join(out)) < desired:
        out.append(base[(start + i) % len(base)])
        i += 1
    result = " ".join(out)
    return result[:desired].rstrip()


repo_root = Path(__file__).resolve().parents[2]
frontend_dir = repo_root / "frontend"

app = FastAPI(title="Deepo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "lorem"}


@app.get("/api/languages")
def languages():
    return {
        "source": ["auto", "en", "fr", "es", "de", "it", "ar"],
        "target": ["fr", "en", "es", "de", "it", "ar"],
        "defaults": {"source": "auto", "target": "fr"},
    }


@app.post("/api/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    t0 = time.perf_counter()
    translated = lorem_from_text(text)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return TranslateResponse(
        translated_text=translated,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        mode=payload.mode,
        elapsed_ms=elapsed_ms,
    )


if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")