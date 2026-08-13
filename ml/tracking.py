"""Minimal experiment tracking: one JSON per run under ``experiments/``.

Every entry records the config, metrics, timestamp, and git commit so results
in ``reports/`` can be traced to the exact code and parameters that produced
them. Deliberately not a service — files diff cleanly and survive forever.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def log_experiment(
    kind: str,
    config: dict,
    metrics: dict,
    out_dir: str | Path = "experiments",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    experiment_id = f"{stamp.strftime('%Y%m%d_%H%M%S')}_{kind}_{uuid.uuid4().hex[:6]}"
    payload = {
        "experiment_id": experiment_id,
        "kind": kind,
        "created_at": stamp.isoformat(),
        "git_commit": git_commit(),
        "config": config,
        "metrics": metrics,
    }
    path = out / f"{experiment_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
