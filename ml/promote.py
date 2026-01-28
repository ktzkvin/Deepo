from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
VER_FILE = ROOT / "ml" / "artifacts" / "model_version.txt"


def main() -> None:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)

    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    model_name = os.getenv("MODEL_NAME", "deepo-translator")
    target_stage = os.getenv("PROMOTE_TO_STAGE", "Production")

    if not VER_FILE.exists():
        raise SystemExit("model_version.txt not found")

    version = VER_FILE.read_text(encoding="utf-8").strip()
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