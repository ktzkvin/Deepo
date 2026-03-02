from __future__ import annotations
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
import torch

import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "ml" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
CKPT_PATH = ROOT / "models" / "lstm_seq2seq" / "best.pt"

def get_git_sha() -> str:
    sha = os.getenv("GITHUB_SHA", "").strip()
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"

def get_dvc_data_version() -> str:
    # (Garde ta fonction get_dvc_data_version d'origine ici)
    return "unknown"

class DeepoTranslatorWrapper(PythonModel):
    def load_context(self, context):
        import model_def # Importé via code_paths de MLflow
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(context.artifacts["ckpt"], map_location=self.device)
        
        self.src_itos = ckpt['src_itos']
        self.tgt_itos = ckpt['tgt_itos']
        self.src_stoi = {w: i for i, w in enumerate(self.src_itos)}
        self.tgt_stoi = {w: i for i, w in enumerate(self.tgt_itos)}
        
        self.PAD = '<pad>'
        self.SOS = '<s>'
        self.EOS = '</s>'
        self.UNK = '<unk>'
        self.MAX_LEN = 60
        
        emb, hid = ckpt['emb'], ckpt['hid']
        num_layers = ckpt.get('num_layers', 1)
        
        self.enc = model_def.Encoder(len(self.src_itos), emb, hid, self.src_stoi[self.PAD], num_layers)
        self.dec = model_def.Decoder(len(self.tgt_itos), emb, hid, hid, self.tgt_stoi[self.PAD], num_layers)
        
        full_model = ckpt['model']
        self.enc.load_state_dict({k[4:]: v for k, v in full_model.items() if k.startswith('enc.')})
        self.dec.load_state_dict({k[4:]: v for k, v in full_model.items() if k.startswith('dec.')})
        
        self.enc.to(self.device).eval()
        self.dec.to(self.device).eval()
        self.model_def = model_def

    def predict(self, context, model_input):
        results = []
        for row in model_input:
            text = str(row.get("text", ""))
            target_lang = str(row.get("target_lang", "fr"))
            
            # Format attendu par ton modèle : ">>fra<< Hello"
            src_with_tag = f">>{target_lang}<< {text.capitalize()}"
            clean = self.model_def.preprocess(src_with_tag)
            
            ids = [self.src_stoi.get(t, self.src_stoi[self.UNK]) for t in self.model_def.tokenize(clean)]
            ids.append(self.src_stoi[self.EOS])
            src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                enc_outputs, h, c = self.enc(src_tensor)
                inp = torch.tensor([self.tgt_stoi[self.SOS]], dtype=torch.long).to(self.device)
                result_tokens = []
                for _ in range(self.MAX_LEN):
                    logits, h, c = self.dec(inp, h, c, enc_outputs)
                    top1 = logits.argmax(1).item()
                    if self.tgt_itos[top1] == self.EOS: break
                    result_tokens.append(self.tgt_itos[top1])
                    inp = torch.tensor([top1], dtype=torch.long).to(self.device)
                    
            results.append(self.model_def.postprocess(result_tokens))
        return results

def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = os.getenv("MODEL_NAME", "deepo-translator")
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "deepo-training"))

    with mlflow.start_run() as run:
        mlflow.log_param("git_sha", get_git_sha())
        mlflow.log_param("model_kind", "seq2seq_lstm")

        # Test smoke
        t0 = time.perf_counter()
        wrapper = DeepoTranslatorWrapper()
        
        # On log le modèle dans MLflow avec ses dépendances
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            registered_model_name=model_name,
            artifacts={"ckpt": str(CKPT_PATH)},
            code_paths=[str(ROOT / "ml" / "model_def.py")] # Injecte tes classes
        )
        
        latency_ms = int((time.perf_counter() - t0) * 1000)
        mlflow.log_metric("smoke_ok", 1.0)
        mlflow.log_metric("latency_ms", latency_ms)

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        newest = max(int(v.version) for v in versions)
        ART.joinpath("model_version.txt").write_text(str(newest), encoding="utf-8")
        ART.joinpath("run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        print(f"Modèle enregistré (Version {newest})")

if __name__ == "__main__":
    main()