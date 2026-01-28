from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "ml" / "artifacts" / "metrics.json"


def main() -> None:
    if not METRICS.exists():
        raise SystemExit("metrics.json not found")

    data = json.loads(METRICS.read_text(encoding="utf-8"))
    smoke_ok = float(data.get("smoke_ok", 0.0))
    latency_ms = float(data.get("latency_ms", 10_000.0))

    min_smoke = float(os.getenv("GATE_SMOKE_MIN", "1.0"))
    max_latency = float(os.getenv("GATE_LATENCY_MAX_MS", "500.0"))

    if smoke_ok < min_smoke:
        raise SystemExit(f"gate failed: smoke_ok {smoke_ok} < {min_smoke}")

    if latency_ms > max_latency:
        raise SystemExit(f"gate failed: latency_ms {latency_ms} > {max_latency}")

    print("gates passed")


if __name__ == "__main__":
    main()