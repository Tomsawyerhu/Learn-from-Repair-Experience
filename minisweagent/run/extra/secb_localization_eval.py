#!/usr/bin/env python3
"""Compute SEC-bench localization coverage based on gold patches and model patches."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonlines import jsonlines

_HUNK_FUNC_RE = re.compile(r"(~?[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)\s*\(")
_INVALID_FUNC_NAMES = {"if", "for", "while", "switch", "return", "sizeof"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_preds(preds_path: Path) -> dict[str, Any]:
    if not preds_path.exists():
        return {}
    payload = _load_json(preds_path)
    return payload if isinstance(payload, dict) else {}


def _load_valid_instance_ids(
    eval_dir: Path,
    preds: dict[str, Any],
) -> list[str]:
    summary_path = eval_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = _load_json(summary_path)
            prediction_summary = summary.get("prediction_summary", {})
            valid_ids = prediction_summary.get("valid_patch_instances", [])
            if isinstance(valid_ids, list):
                return [str(item) for item in valid_ids if str(item).strip()]
        except Exception:
            pass

    prefilter_summary = eval_dir / "prefilter_summary.json"
    prefilter_rejections = eval_dir / "prefilter_rejections.jsonl"
    rejected: set[str] = set()

    if prefilter_summary.exists():
        try:
            payload = _load_json(prefilter_summary)
            for item in payload.get("rejected_instances", []) or []:
                value = str(item).strip()
                if value:
                    rejected.add(value)
        except Exception:
            pass

    if prefilter_rejections.exists():
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

    return sorted(set(preds) - rejected)


def _extract_function_name(context: str) -> str | None:
    if not context:
        return None
    matches = _HUNK_FUNC_RE.findall(context)
    if not matches:
        return None
    func_name = matches[-1]
    if func_name in _INVALID_FUNC_NAMES:
        return None
    return func_name


def _parse_patch(patch_text: str) -> tuple[list[str], list[str]]:
    current_file = None
    files: list[str] = []
    methods: list[str] = []
    seen_methods: set[str] = set()

    for raw_line in patch_text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b_path = parts[3]
                if b_path.startswith("b/"):
                    b_path = b_path[2:]
                if b_path != "/dev/null":
                    current_file = b_path
                    if b_path not in files:
                        files.append(b_path)
            continue

        if line.startswith("@@") and current_file:
            # Hunk header: @@ -a,b +c,d @@ context
            try:
                context = line.split("@@", 2)[-1].strip()
            except Exception:
                context = ""
            func_name = _extract_function_name(context)
            if func_name:
                key = f"{current_file}::{func_name}"
                if key not in seen_methods:
                    seen_methods.add(key)
                    methods.append(key)

    return files, methods


def _limit_gold_methods(gold_methods: list[str], limit: int = 3) -> list[str]:
    if len(gold_methods) <= limit:
        return gold_methods
    return gold_methods[:limit]


def _iter_details(rows: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    yield from rows


def compute_localization_coverage(
    preds_path: Path,
    eval_dir: Path,
    gold_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    preds = _load_preds(preds_path)
    valid_ids = _load_valid_instance_ids(eval_dir, preds)

    total_gold_files = 0
    covered_gold_files = 0
    total_gold_methods = 0
    covered_gold_methods = 0

    missing_gold = []
    missing_pred = []
    method_unresolved = []

    details: list[dict[str, Any]] = []

    for instance_id in valid_ids:
        pred = preds.get(instance_id, {})
        model_patch = pred.get("model_patch", "") if isinstance(pred, dict) else ""
        if model_patch is None:
            model_patch = ""
        if not isinstance(model_patch, str):
            model_patch = str(model_patch)

        gold_path = gold_dir / f"{instance_id}.patch"
        if not gold_path.exists():
            missing_gold.append(instance_id)
            continue
        if not model_patch.strip():
            missing_pred.append(instance_id)
            continue

        gold_patch = gold_path.read_text(encoding="utf-8", errors="replace")
        gold_files, gold_methods = _parse_patch(gold_patch)
        model_files, model_methods = _parse_patch(model_patch)

        gold_methods = _limit_gold_methods(gold_methods, limit=3)

        gold_file_count = len(gold_files)
        covered_file_count = len(set(gold_files) & set(model_files))
        total_gold_files += gold_file_count
        covered_gold_files += covered_file_count

        if gold_methods:
            gold_method_count = len(gold_methods)
            covered_method_count = len(set(gold_methods) & set(model_methods))
            total_gold_methods += gold_method_count
            covered_gold_methods += covered_method_count
        else:
            method_unresolved.append(instance_id)
            gold_method_count = 0
            covered_method_count = 0

        details.append(
            {
                "instance_id": instance_id,
                "gold_file_count": gold_file_count,
                "covered_file_count": covered_file_count,
                "gold_method_count": gold_method_count,
                "covered_method_count": covered_method_count,
                "gold_files": gold_files,
                "model_files": model_files,
                "gold_methods": gold_methods,
                "model_methods": model_methods,
            }
        )

    file_coverage = (covered_gold_files / total_gold_files) if total_gold_files else 0.0
    method_coverage = (covered_gold_methods / total_gold_methods) if total_gold_methods else 0.0

    summary = {
        "preds_path": str(preds_path),
        "gold_dir": str(gold_dir),
        "eval_dir": str(eval_dir),
        "valid_patch_count": len(valid_ids),
        "missing_gold_patch_count": len(missing_gold),
        "missing_model_patch_count": len(missing_pred),
        "method_unresolved_count": len(method_unresolved),
        "file_total_required": total_gold_files,
        "file_total_covered": covered_gold_files,
        "file_coverage": file_coverage,
        "method_total_required": total_gold_methods,
        "method_total_covered": covered_gold_methods,
        "method_coverage": method_coverage,
        "missing_gold_patch_instances": missing_gold,
        "missing_model_patch_instances": missing_pred,
        "method_unresolved_instances": method_unresolved,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "localization_summary.json"
    details_path = output_dir / "localization_details.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with jsonlines.open(details_path, "w") as writer:
        for row in _iter_details(details):
            writer.write(row)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SEC-bench localization coverage")
    parser.add_argument("--preds", required=True, help="Path to preds.json")
    parser.add_argument("--eval-dir", required=True, help="Evaluation directory (contains summary.json)")
    parser.add_argument("--gold-dir", required=True, help="Directory containing gold patches")
    parser.add_argument("--output-dir", required=True, help="Output directory for localization_summary.json")

    args = parser.parse_args()
    compute_localization_coverage(
        preds_path=Path(args.preds).expanduser().resolve(),
        eval_dir=Path(args.eval_dir).expanduser().resolve(),
        gold_dir=Path(args.gold_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
