"""Measure batched checkpoint persistence in a fresh temporary directory.

No providers or document conversion are called. Use --package-path to compare
an older checkout, and --format json to exercise the legacy JSON backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--format", choices=("json", "sqlite3"), default="sqlite3")
    parser.add_argument("--package-path", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    args = parser.parse_args()
    if min(args.blocks, args.batch_size) < 1:
        parser.error("counts must be positive")
    sys.path.insert(0, str(args.package_path))
    from daru.translation.checkpoint import TranslationCheckpointStore

    with tempfile.TemporaryDirectory(prefix="daru-checkpoint-benchmark-") as directory:
        path = Path(directory) / f"job.checkpoint.{args.format}"
        options = dict(path=path, job_type="cad", document_sha256="synthetic",
                       source_lang="en", target_lang="ru", settings_fingerprint="test")
        store = TranslationCheckpointStore(**options)
        save_seconds = 0.0
        started = time.perf_counter()
        for start in range(0, args.blocks, args.batch_size):
            for index in range(start, min(args.blocks, start + args.batch_size)):
                store.upsert(
                    namespace="cad", block_id=str(index),
                    source_text=f"Inspect the safety valve {index}. " * 4,
                    translated_text=f"Проверьте предохранительный клапан {index}. " * 4,
                    extra={"page": index // 50, "bbox": [10, 20, 300, 40]},
                )
            save_started = time.perf_counter()
            store.save()
            save_seconds += time.perf_counter() - save_started
        elapsed = time.perf_counter() - started
        load_started = time.perf_counter()
        loaded = TranslationCheckpointStore(**options)
        load_seconds = time.perf_counter() - load_started
        assert len(loaded.blocks) == args.blocks
        assert loaded.get("cad", str(args.blocks - 1)) == store.get("cad", str(args.blocks - 1))
        print(json.dumps({"format": args.format, "blocks": args.blocks,
                          "batch_size": args.batch_size, "seconds": round(elapsed, 3),
                          "save_seconds": round(save_seconds, 3), "load_seconds": round(load_seconds, 3),
                          "file_mib": round(path.stat().st_size / 1024**2, 3)}))


if __name__ == "__main__":
    main()
