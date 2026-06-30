#!/usr/bin/env python3
"""Export SEC-bench gold patches from official Docker images."""

import argparse
import json
import subprocess
from pathlib import Path


def _read_instance_ids(dataset_path: Path) -> list[str]:
    ids: list[str] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id", "")).strip()
            if instance_id:
                ids.append(instance_id)
    return ids


def _docker_cat_any_patch(image: str, patch_paths: list[str]) -> tuple[subprocess.CompletedProcess, str | None]:
    path_expr = " ".join(patch_paths)
    shell = (
        "for p in "
        + path_expr
        + "; do if [ -f \"$p\" ]; then echo \"FOUND_PATCH_PATH=$p\" 1>&2; cat \"$p\"; exit 0; fi; done; exit 3"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        image,
        "/bin/bash",
        "-lc",
        shell,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    found_path = None
    for line in result.stderr.splitlines():
        if line.startswith("FOUND_PATCH_PATH="):
            found_path = line.split("=", 1)[1].strip()
            break
    return result, found_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gold patches from SEC-bench Docker images")
    parser.add_argument("--dataset", required=True, help="Path to SEC-bench dataset jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory for gold patches")
    parser.add_argument(
        "--image-prefix",
        default="hwiwonlee/secb.eval.x86_64",
        help="Docker image prefix (default: hwiwonlee/secb.eval.x86_64)",
    )
    parser.add_argument(
        "--image-tag",
        default="patch",
        help="Docker image tag (default: patch)",
    )
    parser.add_argument(
        "--tags",
        default="patch,latest",
        help="Comma-separated Docker tags to try in order (default: patch,latest)",
    )
    parser.add_argument(
        "--paths",
        default="/testcase/repo_changes.diff,/testcase/model_patch.diff,/testcase/gold_patch.diff",
        help="Comma-separated candidate patch paths inside container",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip instances that already have an exported patch file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of instances",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = _read_instance_ids(dataset_path)
    if args.limit:
        instance_ids = instance_ids[: args.limit]

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    patch_paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    failures: list[str] = []
    failure_details: dict[str, str] = {}
    success_details: dict[str, dict[str, str]] = {}
    skipped = 0
    success = 0

    for instance_id in instance_ids:
        target = out_dir / f"{instance_id}.patch"
        if args.skip_existing and target.exists():
            skipped += 1
            continue

        patch_text = ""
        used_tag = None
        used_path = None
        last_err = ""

        for tag in tags:
            image = f"{args.image_prefix}.{instance_id}:{tag}"
            result, found_path = _docker_cat_any_patch(image, patch_paths)
            if result.returncode == 0 and result.stdout.strip():
                patch_text = result.stdout
                used_tag = tag
                used_path = found_path
                break
            err = result.stderr.strip()
            if err:
                last_err = err

        if not patch_text.strip():
            failures.append(instance_id)
            failure_details[instance_id] = last_err or "no patch file found"
            continue

        target.write_text(patch_text, encoding="utf-8")
        success += 1
        if used_tag:
            success_details[instance_id] = {"tag": used_tag, "path": used_path or "unknown"}

    summary = {
        "dataset": str(dataset_path),
        "out_dir": str(out_dir),
        "count_total": len(instance_ids),
        "count_success": success,
        "count_failed": len(failures),
        "count_skipped": skipped,
        "failed_instances": failures,
        "failure_details": failure_details,
        "success_details": success_details,
        "image_prefix": args.image_prefix,
        "tags": tags,
        "paths": patch_paths,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
