from __future__ import annotations

import argparse
import json

from src.services.experiment_export import export_live_experiment_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the current live draft as a clean experiment snapshot."
    )
    parser.add_argument("--paper-folder-name", required=True, help="Paper output folder name under data/output.")
    parser.add_argument("--export-name", required=True, help="Experiment snapshot directory name.")
    args = parser.parse_args()

    try:
        metadata = export_live_experiment_snapshot(
            paper_folder_name=args.paper_folder_name,
            export_name=args.export_name,
        )
    except Exception as exc:
        raise SystemExit(f"Experiment export failed: {exc}") from exc

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
