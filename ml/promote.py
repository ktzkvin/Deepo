from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python promote.py <lang_suffix> (e.g., fra, cmn)")
        sys.exit(1)
        
    lang_suffix = sys.argv[1]
    ver_file = ROOT / "ml" / "artifacts" / f"model_version_{lang_suffix}.txt"

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    model_name = f"deepo-translator-{lang_suffix}"
    target_stage = os.getenv("PROMOTE_TO_STAGE", "Production")

    if not ver_file.exists():
        raise SystemExit(f"{ver_file.name} not found. Register the model first.")

    version = ver_file.read_text(encoding="utf-8").strip()
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=target_stage,
        archive_existing_versions=True,
    )
    print(f"promoted {model_name} v{version} to {target_stage}")

if __name__ == "__main__":
    main()