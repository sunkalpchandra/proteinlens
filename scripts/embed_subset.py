"""Embed the evaluation subset with any registered checkpoint.

Same interface for every model: sequences go through the frozen encoder, mean
and max pooling are stored (BOS too for ESM family), and artifacts land under
``data/scaling/{slug}/`` so checkpoints never collide. The default serving
store is untouched.

Usage:
    python scripts/embed_subset.py --model facebook/esm2_t6_8M_UR50D
    python scripts/embed_subset.py --model facebook/esm2_t30_150M_UR50D
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.tracking import log_experiment  # noqa: E402
from models.encoder import ESM2Encoder  # noqa: E402
from models.pooling import Pooler  # noqa: E402
from models.registry import check_memory, spec_for, study_dir  # noqa: E402


def host_ram_gb() -> float:
    import subprocess

    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    try:
        return int(out.stdout.strip()) / 1024**3
    except ValueError:
        return 16.0  # non-macOS: assume enough and let the OS complain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset", type=Path, default=Path("data/processed/eval_subset.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/proteins.parquet"))
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()

    spec = spec_for(args.model)
    warning = check_memory(args.model, host_ram_gb())
    if warning:
        print(f"WARNING: {warning}")

    subset = json.loads(args.subset.read_text())["accessions"]
    df = pd.read_parquet(args.corpus).set_index("accession")
    sequences = df.loc[subset, "sequence"].tolist()

    encoder = ESM2Encoder(args.model)
    pooler = Pooler(None)
    poolings = ["mean", "max", "bos"]
    matrices = {p: np.zeros((len(subset), encoder.hidden_size), dtype=np.float32)
                for p in poolings}

    print(f"Embedding {len(subset)} proteins with {args.model} "
          f"({spec.params_m}M params, {encoder.hidden_size}-d) on {encoder.device}")
    t0 = time.time()
    for start in range(0, len(subset), args.chunk):
        encoded = encoder.encode_batch(sequences[start : start + args.chunk])
        for offset, enc in enumerate(encoded):
            for pooling in poolings:
                vec, _ = pooler.pool(enc.residue_embeddings, enc.bos_embedding, pooling)
                matrices[pooling][start + offset] = vec.numpy()
        done = min(start + args.chunk, len(subset))
        rate = done / (time.time() - t0)
        print(f"  {done:>5}/{len(subset)}  {rate:5.1f}/s  "
              f"ETA {(len(subset)-done)/max(rate,1e-9)/60:5.1f} min", flush=True)

    out = study_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    for pooling, matrix in matrices.items():
        np.save(out / f"subset_{pooling}.npy", matrix)
    (out / "meta.json").write_text(json.dumps({
        "model": args.model,
        "slug": spec.slug,
        "dim": encoder.hidden_size,
        "n": len(subset),
        "poolings": poolings,
        "subset_file": str(args.subset),
        "wall_time_s": round(time.time() - t0, 1),
        "device": str(encoder.device),
        "created_at": datetime.now(UTC).isoformat(),
    }, indent=2))
    print(f"Done in {(time.time()-t0)/60:.1f} min → {out}")

    log_experiment("embed_subset",
                   config={"model": args.model, "n": len(subset)},
                   metrics={"wall_time_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
