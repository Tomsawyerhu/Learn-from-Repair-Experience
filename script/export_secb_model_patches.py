#!/usr/bin/env python3
"""Export SEC-bench model patches to individual files."""

import argparse
import json
from pathlib import Path
from typing import Iterable

from jsonlines import jsonlines


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rejected_ids(prefilter_summary: Path | None, prefilter_rejections: Path | None) -> set[str]:
    rejected: set[str] = set()
    if prefilter_summary and prefilter_summary.exists():
        try:
            payload = _load_json(prefilter_summary)
            for item in payload.get("rejected_instances", []) or []:
                value = str(item).strip()
                if value:
                    rejected.add(value)
        except Exception:
            pass
    if prefilter_rejections and prefilter_rejections.exists():
        try:
            with jsonlines.open(prefilter_rejections, "r") as reader:
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    value = str(row.get("instance_id", "")).strip()
                    if value:
                        rejected.add(value)
        except Exception:
            pass
    return rejected


def _iter_preds(path: Path) -> Iterable[tuple[str, dict]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return []
    return payload.items()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SEC-bench model patches into individual files")
    parser.add_argument("--preds", required=True, help="Path to preds.json")
    parser.add_argument("--out-dir", required=True, help="Output directory to store patch files")
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Only export patches that passed prefilter (exclude rejected instances)",
    )
    parser.add_argument(
        "--prefilter-summary",
        default=None,
        help="Path to prefilter_summary.json (optional, used with --valid-only)",
    )
    parser.add_argument(
        "--prefilter-rejections",
        default=None,
        help="Path to prefilter_rejections.jsonl (optional, used with --valid-only)",
    )

    args = parser.parse_args()

    preds_path = Path(args.preds).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefilter_summary = Path(args.prefilter_summary).expanduser().resolve() if args.prefilter_summary else None
    prefilter_rejections = (
        Path(args.prefilter_rejections).expanduser().resolve() if args.prefilter_rejections else None
    )
    rejected_ids = _load_rejected_ids(prefilter_summary, prefilter_rejections) if args.valid_only else set()

    exported = 0
    skipped = 0
    manifest: dict[str, str] = {}

    for instance_id, pred in _iter_preds(preds_path):
        if args.valid_only and instance_id in rejected_ids:
            skipped += 1
            continue
        model_patch = ""
        if isinstance(pred, dict):
            model_patch = pred.get("model_patch", "")
        if model_patch is None:
            model_patch = ""
        if not isinstance(model_patch, str):
            model_patch = str(model_patch)

        filename = f"{instance_id}.patch"
        target = out_dir / filename
        target.write_text(model_patch, encoding="utf-8")
        manifest[instance_id] = str(target)
        exported += 1

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "preds": str(preds_path),
        "out_dir": str(out_dir),
        "exported": exported,
        "skipped": skipped,
        "valid_only": bool(args.valid_only),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
