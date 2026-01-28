from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "ml" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def get_git_sha() -> str:
    sha = os.getenv("GITHUB_SHA", "").strip()
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


def get_dvc_data_version() -> str:
    dvc_file = ROOT / "data" / "raw.dvc"
    if not dvc_file.exists():
        dvc_file = ROOT / "data" / "raw"  # fallback
        return "unknown"
    try:
        txt = dvc_file.read_text(encoding="utf-8")
        for line in txt.splitlines():
            if line.strip().startswith("md5:"):
                return line.split("md5:")[1].strip()
    except Exception:
        pass
    return "unknown"


class DummyTranslator(PythonModel):
    def predict(self, context: Any, model_input: Any) -> Any:
        out = []
        for row in model_input:
            t = str(row.get("text", ""))
            out.append(t[::-1])
        return out


def main() -> None:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

    mlflow.set_tracking_uri(tracking_uri)
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    model_name = os.getenv("MODEL_NAME", "deepo-translator")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "deepo-training")

    mlflow.set_experiment(exp_name)

    git_sha = get_git_sha()
    dvc_ver = get_dvc_data_version()

    with mlflow.start_run() as run:
        mlflow.log_param("git_sha", git_sha)
        mlflow.log_param("dvc_data_version", dvc_ver)
        mlflow.log_param("model_kind", "dummy_reverse")

        t0 = time.perf_counter()
        dummy = DummyTranslator()
        sample_in = [{"text": "hello", "source_lang": "en", "target_lang": "fr"}]
        pred = dummy.predict(None, sample_in)[0]
        latency_ms = int((time.perf_counter() - t0) * 1000)

        metric_ok = 1.0 if pred == "olleh" else 0.0
        mlflow.log_metric("smoke_ok", metric_ok)
        mlflow.log_metric("latency_ms", latency_ms)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=dummy,
            registered_model_name=model_name,
        )

        ART.joinpath("metrics.json").write_text(
            json.dumps({"smoke_ok": metric_ok, "latency_ms": latency_ms}, indent=2),
            encoding="utf-8",
        )

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        newest = max(int(v.version) for v in versions)
        ART.joinpath("model_version.txt").write_text(str(newest), encoding="utf-8")
        ART.joinpath("run_id.txt").write_text(run.info.run_id, encoding="utf-8")


if __name__ == "__main__":
    main()