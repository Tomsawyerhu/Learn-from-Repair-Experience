#!/usr/bin/env python3
"""Batch-pull SEC evaluation images for EvoRepair.

The script reads SEC instances from the default EvoRepair dataset and pulls the
corresponding pre-built evaluation images from Docker Hub. Image naming stays
aligned with ``minisweagent.run.extra.secb_exp``:

    hwiwonlee/secb.eval.x86_64.<instance_id>:patch
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_IMAGE_PREFIX = "hwiwonlee/secb.eval.x86_64"
DEFAULT_IMAGE_TAG = "patch"
DEFAULT_MAX_WORKERS = 8
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "minisweagent" / "data" / "secb_official_cve201_dataset.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull SEC-bench evaluation images used by multi_turn_runner_secb.py."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"SEC dataset JSONL path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--image-prefix",
        default=DEFAULT_IMAGE_PREFIX,
        help=f"Image prefix (default: {DEFAULT_IMAGE_PREFIX})",
    )
    parser.add_argument(
        "--image-tag",
        default=DEFAULT_IMAGE_TAG,
        help=f"Image tag (default: {DEFAULT_IMAGE_TAG})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Parallel pull workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("pull_secb_images.log"),
        help="Path to pull log file (default: pull_secb_images.log)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the images that would be pulled without executing docker pull.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pull even if the image already exists locally.",
    )
    return parser.parse_args()


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pull_secb_images")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def ensure_docker_available() -> None:
    docker_path = shutil.which("docker")
    if docker_path is None:
        raise FileNotFoundError("The 'docker' executable was not found in PATH.")


def load_images(dataset_path: Path, image_prefix: str, image_tag: str) -> list[str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    seen: set[str] = set()
    images: list[str] = []

    with dataset_path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}") from exc
            instance_id = str(record.get("instance_id", "")).strip()
            if not instance_id:
                raise ValueError(f"Missing instance_id on line {line_number} of {dataset_path}")
            image = f"{image_prefix}.{instance_id}:{image_tag}"
            if image not in seen:
                seen.add(image)
                images.append(image)

    if not images:
        raise ValueError(f"No SEC images were resolved from dataset: {dataset_path}")
    return images


def image_exists_locally(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def pull_image(image: str, logger: logging.Logger, force: bool) -> tuple[str, bool, str]:
    if not force and image_exists_locally(image):
        logger.info("[%s] already exists locally, skipping", image)
        return image, True, "skipped"

    logger.info("[%s] pulling", image)
    result = subprocess.run(
        ["docker", "pull", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode == 0:
        logger.info("[%s] pull finished", image)
        return image, True, "pulled"

    output = (result.stdout or "").strip()
    if output:
        logger.error("[%s] pull failed: %s", image, output.splitlines()[-1])
    else:
        logger.error("[%s] pull failed with exit code %s", image, result.returncode)
    return image, False, output


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    images = load_images(args.dataset.resolve(), args.image_prefix.strip(), args.image_tag.strip())
    logger.info("Resolved %s SEC evaluation images from %s", len(images), args.dataset.resolve())

    if args.dry_run:
        for image in images:
            logger.info("[dry-run] %s", image)
        logger.info("Dry run completed")
        return 0

    ensure_docker_available()

    success = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(pull_image, image, logger, args.force): image
            for image in images
        }
        for future in as_completed(futures):
            image, ok, status = future.result()
            if ok:
                if status == "skipped":
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
                logger.error("[%s] marked as failed", image)

    logger.info(
        "Finished pulling SEC images. pulled=%s skipped=%s failed=%s total=%s",
        success,
        skipped,
        failed,
        len(images),
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
