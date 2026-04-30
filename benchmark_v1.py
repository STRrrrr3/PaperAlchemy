from __future__ import annotations

import time

from src.benchmark_v1.ui import build_app
from src.services.artifact_store import OUTPUT_DIR


def main() -> None:
    print("Building PaperAlchemy Benchmark V1 app...", flush=True)
    app = build_app()
    print("Launching PaperAlchemy Benchmark V1 on http://127.0.0.1:7861 ...", flush=True)
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        allowed_paths=[str(OUTPUT_DIR.resolve())],
        prevent_thread_lock=True,
    )
    print("PaperAlchemy Benchmark V1 is running on http://127.0.0.1:7861", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
