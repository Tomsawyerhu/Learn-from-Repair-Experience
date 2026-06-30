"""Package all experimental outputs (except vul4j) into zip files."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    """Zip each experiment directory under `minisweagent/output` (except vul4j)."""

    workspace = Path(__file__).resolve().parents[1]
    output_dir = workspace / "minisweagent" / "output"
    result_dir = workspace / "result"

    if not output_dir.exists():
        raise SystemExit(f"output directory not found: {output_dir}")

    result_dir.mkdir(parents=True, exist_ok=True)

    for experiment in sorted(output_dir.iterdir()):
        if not experiment.is_dir():
            continue
        if experiment.name == "vul4j":
            continue

        archive_path = result_dir / f"{experiment.name}.zip"
        # ensure existing archives are overwritten
        if archive_path.exists():
            archive_path.unlink()

        shutil.make_archive(str(result_dir / experiment.name), "zip", root_dir=experiment)


if __name__ == "__main__":
    main()
