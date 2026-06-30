#!/usr/bin/env python3
"""
Collect patch submissions from VulAgentEvolver/minisweagent/output_patcheval/ directories.

This script reads JSON files from subdirectories starting with a given prefix,
extracts the 'submission' field from the 'info' object, and writes a JSON array
in the format of example_patch.json.

Usage:
    python collect_patches.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                              [--prefix PREFIX] [--output-file OUTPUT_FILE]

Defaults:
    input_dir:   /mnt/data/hhc/VulAgentEvolver/minisweagent/output_patcheval/
    output_dir:  /mnt/data/hhc/PatchEval/patcheval/evaluation/
    prefix:      CVE
    output_file: collected_patches.json
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Invalid patch patterns that should be filtered out
INVALID_PATCH_PATTERNS = [
    "time or money limit exceeded",
    "timed out after",
    "Command '['docker'",
]


@dataclass
class CollectionStats:
    """Statistics for patch collection."""
    total_dirs: int = 0
    dirs_with_json: int = 0
    dirs_without_json: int = 0
    valid_patches: int = 0
    empty_patches: int = 0
    timeout_patches: int = 0
    invalid_format_patches: int = 0
    no_submission_field: int = 0
    skipped_cves: List[Tuple[str, str]] = field(default_factory=list)  # (cve_name, reason)

    def print_summary(self) -> None:
        """Print collection statistics summary."""
        logger.info("=" * 60)
        logger.info("Patch Collection Statistics Summary")
        logger.info("=" * 60)
        logger.info(f"Total directories scanned:     {self.total_dirs}")
        logger.info(f"Directories with JSON files:   {self.dirs_with_json}")
        logger.info(f"Directories without JSON:      {self.dirs_without_json}")
        logger.info("-" * 60)
        logger.info(f"Valid patches collected:       {self.valid_patches}")
        logger.info(f"Empty patches (skipped):       {self.empty_patches}")
        logger.info(f"Timeout errors (skipped):      {self.timeout_patches}")
        logger.info(f"Invalid format (skipped):      {self.invalid_format_patches}")
        logger.info(f"No submission field (skipped): {self.no_submission_field}")
        logger.info("-" * 60)

        if self.skipped_cves:
            logger.info("Skipped CVEs:")
            for cve, reason in self.skipped_cves:
                logger.info(f"  - {cve}: {reason}")

        logger.info("=" * 60)


def is_valid_patch(submission: str) -> Tuple[bool, str]:
    """
    Check if the submission is a valid patch.

    Args:
        submission: The submission string to validate.

    Returns:
        Tuple of (is_valid, reason) where reason explains why it's invalid.
    """
    if not submission:
        return False, "empty"

    # Check for known invalid patterns
    submission_lower = submission.lower()
    for pattern in INVALID_PATCH_PATTERNS:
        if pattern.lower() in submission_lower:
            if "timed out" in submission_lower or "timeout" in submission_lower:
                return False, "timeout"
            return False, "invalid_pattern"

    # Check if it looks like a valid patch format
    # Valid patches typically contain diff markers or patch-like content
    valid_indicators = [
        "diff --git",
        "diff -",
        "--- ",
        "+++ ",
        "@@",
        "*** Begin Patch",
        "*** Update File",
        "From ",  # git format-patch
        "index ",
    ]

    has_valid_indicator = any(indicator in submission for indicator in valid_indicators)

    if not has_valid_indicator:
        # If no valid indicator found, it might be an error message or invalid content
        # Check if it's too short to be a meaningful patch
        if len(submission.strip()) < 50:
            return False, "too_short"
        # Check if it starts with common error patterns
        if submission.strip().startswith("Command '"):
            return False, "command_error"

    return True, "valid"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect patch submissions from CVE subdirectories."
    )
    parser.add_argument(
        '--input-dir',
        default='/mnt/data/hhc/VulAgentEvolver/minisweagent/output_patcheval_exp/',
        help='Directory containing CVE subdirectories (default: %(default)s)'
    )
    parser.add_argument(
        '--output-dir',
        default='/mnt/data/hhc/PatchEval/patcheval/evaluation/',
        help='Directory to write output JSON file (default: %(default)s)'
    )
    parser.add_argument(
        '--prefix',
        default='CVE',
        help='Prefix of subdirectories to process (default: %(default)s)'
    )
    parser.add_argument(
        '--output-file',
        default='collected_patches.json',
        help='Output JSON filename (default: %(default)s)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without writing output'
    )
    return parser.parse_args()


def find_cve_directories(input_dir: Path, prefix: str) -> List[Path]:
    """
    Find all subdirectories in input_dir that start with the given prefix.

    Args:
        input_dir: Path to the parent directory.
        prefix: Prefix to match.

    Returns:
        List of Path objects for matching subdirectories.
    """
    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        return []

    cve_dirs = []
    for entry in input_dir.iterdir():
        if entry.is_dir() and entry.name.startswith(prefix):
            cve_dirs.append(entry)

    logger.info(f"Found {len(cve_dirs)} directories with prefix '{prefix}'")
    return cve_dirs


def find_json_files_in_dir(dir_path: Path) -> List[Path]:
    """
    Find all JSON files in a directory.

    Args:
        dir_path: Directory to search.

    Returns:
        List of Path objects for JSON files.
    """
    json_files = list(dir_path.glob('*.json'))
    # Also look for files ending with .json but possibly with extra suffixes
    if not json_files:
        json_files = list(dir_path.glob('**/*.json'))
    return json_files


def extract_submission_from_json(json_file: Path) -> Optional[str]:
    """
    Read a JSON file and extract the submission field.

    Args:
        json_file: Path to JSON file.

    Returns:
        Submission string if found, else None.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to parse JSON file {json_file}: {e}")
        return None
    except IOError as e:
        logger.warning(f"Failed to read file {json_file}: {e}")
        return None

    # Navigate to info.submission
    info = data.get('info')
    if not info:
        logger.warning(f"No 'info' field in {json_file}")
        return None

    submission = info.get('submission')
    if submission is None:
        logger.warning(f"No 'submission' field in {json_file}")
        return None

    if not isinstance(submission, str):
        logger.warning(f"Submission field is not a string in {json_file}")
        return None

    return submission


def collect_patches(input_dir: Path, prefix: str) -> Tuple[List[Dict[str, Any]], CollectionStats]:
    """
    Collect patches from all CVE subdirectories.

    Args:
        input_dir: Parent directory containing CVE subdirectories.
        prefix: Prefix for subdirectory names.

    Returns:
        Tuple of (patches list, collection statistics).
    """
    stats = CollectionStats()
    cve_dirs = find_cve_directories(input_dir, prefix)
    stats.total_dirs = len(cve_dirs)

    if not cve_dirs:
        return [], stats

    patches = []
    for cve_dir in cve_dirs:
        cve_name = cve_dir.name
        json_files = find_json_files_in_dir(cve_dir)

        if not json_files:
            logger.warning(f"No JSON files found in {cve_dir}")
            stats.dirs_without_json += 1
            stats.skipped_cves.append((cve_name, "no JSON files"))
            continue

        stats.dirs_with_json += 1

        # Use the first JSON file that yields a submission
        submission = None
        for json_file in json_files:
            submission = extract_submission_from_json(json_file)
            if submission is not None:
                break

        if submission is None:
            logger.warning(f"No valid submission found in {cve_dir}")
            stats.no_submission_field += 1
            stats.skipped_cves.append((cve_name, "no submission field"))
            continue

        # Validate the patch
        is_valid, reason = is_valid_patch(submission)
        if not is_valid:
            if reason == "empty":
                stats.empty_patches += 1
                logger.warning(f"Empty patch for {cve_name}")
            elif reason == "timeout":
                stats.timeout_patches += 1
                logger.warning(f"Timeout error for {cve_name}")
            else:
                stats.invalid_format_patches += 1
                logger.warning(f"Invalid patch format for {cve_name}: {reason}")
            stats.skipped_cves.append((cve_name, reason))
            continue

        patches.append({
            'cve': cve_name,
            'fix_patch': submission
        })
        stats.valid_patches += 1
        logger.info(f"Collected valid patch for {cve_name}")

    return patches, stats


def write_output(patches: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write patches as a JSON array to output_path.

    Args:
        patches: List of patch dictionaries.
        output_path: Path to output JSON file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patches, f, indent=4, ensure_ascii=False)
        logger.info(f"Written {len(patches)} patches to {output_path}")
    except IOError as e:
        logger.error(f"Failed to write output file {output_path}: {e}")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_file = output_dir / args.output_file

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Prefix: {args.prefix}")

    # Ensure output directory exists
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Collect patches
    patches, stats = collect_patches(input_dir, args.prefix)

    # Print statistics summary
    stats.print_summary()

    if not patches:
        logger.error("No valid patches collected. Exiting.")
        sys.exit(1)

    logger.info(f"Successfully collected {len(patches)} valid patches")

    if args.dry_run:
        logger.info("Dry run - would write the following patches:")
        for patch in patches:
            print(f"{patch['cve']}: {patch['fix_patch'][:100]}...")
        return

    # Write output
    write_output(patches, output_file)


if __name__ == '__main__':
    main()