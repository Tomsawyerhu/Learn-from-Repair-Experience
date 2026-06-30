#!/usr/bin/env python3
"""
Aggregate Patches and Run Evaluation Script

This script aggregates patches from multiple turns of a multi-turn vulnerability
repair experiment and runs a unified evaluation.

Usage:
    python aggregate_patches.py --prefix <experiment_prefix>

Examples:
    # Process all experiments matching prefix 'deepseek-230'
    python aggregate_patches.py --prefix deepseek-230

    # Process with custom root directory
    python aggregate_patches.py --prefix deepseek-230 --root /path/to/output

    # Dry run (only aggregate, skip evaluation)
    python aggregate_patches.py --prefix deepseek-230 --dry-run
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "minisweagent" / "output"
PATCHEVAL_ROOT = PROJECT_ROOT / "PatchEval"
DEFAULT_EVAL_SCRIPT = PATCHEVAL_ROOT / "patcheval" / "evaluation" / "run_evaluation.py"
DEFAULT_INPUT_FILE = PATCHEVAL_ROOT / "datasets" / "input.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def normalize_cve_id(cve: str) -> str:
    """Normalize CVE ID to standard format."""
    cve = cve.strip().upper()
    # Extract CVE-YYYY-NNNNN pattern if present
    import re

    match = re.search(r"(CVE-\d{4}-\d+)", cve)
    if match:
        return match.group(1)
    return cve


def load_patches(patches_file: Path) -> dict[str, dict[str, Any]]:
    """
    Load patches from patches.json file.

    Returns:
        Dict mapping CVE ID to patch object {cve, fix_patch}
    """
    if not patches_file.exists():
        return {}

    try:
        with open(patches_file, encoding="utf-8") as f:
            patches_list = json.load(f)

        if not isinstance(patches_list, list):
            logger.warning(f"Invalid patches.json format in {patches_file}")
            return {}

        return {normalize_cve_id(p["cve"]): p for p in patches_list if "cve" in p and "fix_patch" in p}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse {patches_file}: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Error reading {patches_file}: {e}")
        return {}


def load_successful_cves(summary_file: Path) -> set[str]:
    """
    Load successful CVEs from evaluation summary.json.

    Returns:
        Set of CVE IDs that were successfully repaired (strict evaluation)
    """
    if not summary_file.exists():
        return set()

    try:
        with open(summary_file, encoding="utf-8") as f:
            summary = json.load(f)

        strict_eval = summary.get("strict_evaluation", {})
        successful_cves_dict = strict_eval.get("successful_cves", {})

        if not isinstance(successful_cves_dict, dict):
            return set()

        success_cves: set[str] = set()
        for _lang, cve_list in successful_cves_dict.items():
            if isinstance(cve_list, list):
                for cve in cve_list:
                    success_cves.add(normalize_cve_id(cve))

        return success_cves
    except Exception as e:
        logger.warning(f"Error reading {summary_file}: {e}")
        return set()


def process_experiment(
    exp_dir: Path,
    eval_script: Path,
    input_file: Path,
    max_workers: int,
    max_turn: int | None,
    dry_run: bool,
) -> bool:
    """
    Process a single experiment directory.

    Args:
        exp_dir: Experiment directory path
        eval_script: Path to evaluation script
        input_file: Path to input.json
        max_workers: Max workers for evaluation
        max_turn: Maximum turn number to include (inclusive)
        dry_run: If True, only aggregate patches, skip evaluation

    Returns:
        True if successful, False otherwise
    """
    exp_name = exp_dir.name
    logger.info("=" * 60)
    logger.info(f"Processing experiment: {exp_name}")
    logger.info("=" * 60)

    # Check if statistics directory exists
    statistics_dir = exp_dir / "statistics"
    if not statistics_dir.exists():
        logger.warning(f"Skip {exp_name}: 'statistics' directory not found")
        return False

    # Find all turn directories
    all_turn_dirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("turn_")],
        key=lambda x: int(x.name.split("_")[1]),
    )

    # Apply max turn filter if specified
    if max_turn is not None:
        turn_dirs = [d for d in all_turn_dirs if int(d.name.split("_")[1]) <= max_turn]
        logger.info(f"Applying max turn filter: <= {max_turn}")
    else:
        turn_dirs = all_turn_dirs

    if not turn_dirs:
        logger.warning(f"Skip {exp_name}: no turn directories found")
        return False

    logger.info(f"Found {len(turn_dirs)} turn directories")

    # Track successful CVEs across all turns
    ever_success_cves: set[str] = set()

    # Track final patches (cve -> patch_obj)
    final_patches: dict[str, dict[str, Any]] = {}

    # Process each turn in order
    for turn_dir in turn_dirs:
        turn_name = turn_dir.name
        turn_num = int(turn_name.split("_")[1])

        # Load patches for this turn
        patches_file = turn_dir / "patches.json"
        turn_patches = load_patches(patches_file)

        if not turn_patches:
            logger.info(f"  Turn {turn_num}: no patches found, skipping")
            continue

        # Load successful CVEs from evaluation
        summary_file = turn_dir / "evaluation" / "summary.json"
        turn_success_cves = load_successful_cves(summary_file)

        logger.info(f"  Turn {turn_num}: {len(turn_patches)} patches, {len(turn_success_cves)} successful")

        # Update ever_success_cves
        ever_success_cves.update(turn_success_cves)

        # Merge patches with overwrite logic:
        # - Success patches: always overwrite
        # - Failure patches: only overwrite if not in ever_success_cves
        for cve, patch_obj in turn_patches.items():
            is_success = cve in turn_success_cves

            if is_success:
                # Success patch always overwrites
                final_patches[cve] = patch_obj
            elif cve not in ever_success_cves:
                # Failure patch only overwrites if never succeeded before
                final_patches[cve] = patch_obj
            # else: skip - this CVE succeeded before, don't overwrite with failure

    # Check final patch count
    final_count = len(final_patches)
    logger.info(f"Final patch count: {final_count}")
    logger.info(f"  - Ever successful CVEs: {len(ever_success_cves)}")

    if final_count != 230:
        logger.error(f"ERROR: Final patch count ({final_count}) != 230. Skipping {exp_name}")
        return False

    # Create all/ subdirectory
    all_dir = exp_dir / "all"
    if all_dir.exists():
        logger.info("Removing existing all/ directory")
        shutil.rmtree(all_dir)
    all_dir.mkdir(parents=True, exist_ok=True)

    # Write aggregated patches
    all_patches_file = all_dir / "all_patches.json"
    patches_list = list(final_patches.values())
    with open(all_patches_file, "w", encoding="utf-8") as f:
        json.dump(patches_list, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(patches_list)} patches to {all_patches_file}")

    # Write merge report
    merge_report = {
        "experiment_name": exp_name,
        "processed_at": datetime.now().isoformat(),
        "turn_count": len(turn_dirs),
        "ever_success_cves_count": len(ever_success_cves),
        "final_patch_count": final_count,
        "patches_file": str(all_patches_file),
    }
    merge_report_file = all_dir / "merge_report.json"
    with open(merge_report_file, "w", encoding="utf-8") as f:
        json.dump(merge_report, f, indent=2, ensure_ascii=False)

    if dry_run:
        logger.info("Dry run: skipping evaluation")
        return True

    # Run evaluation
    logger.info("Running evaluation...")

    eval_cwd = eval_script.parent
    eval_run_name = f"run_{int(time.time())}"

    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--patch_file",
        str(all_patches_file),
        "--input_file",
        str(input_file),
        "--output",
        eval_run_name,
        "--max_workers",
        str(max_workers),
    ]

    logger.info(f"Evaluation command: {' '.join(eval_cmd)}")

    try:
        result = subprocess.run(
            eval_cmd,
            cwd=str(eval_cwd),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Evaluation failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"stderr: {result.stderr[:1000]}")
            return False

        # Copy evaluation results to all/ directory
        eval_results_dir = eval_cwd / "evaluation_output" / eval_run_name
        target_eval_dir = all_dir / "evaluation"

        if eval_results_dir.exists():
            if target_eval_dir.exists():
                shutil.rmtree(target_eval_dir)
            shutil.copytree(eval_results_dir, target_eval_dir)
            logger.info(f"Copied evaluation results to {target_eval_dir}")

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate patches from multi-turn experiments and run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_patches.py --prefix deepseek-230
  python aggregate_patches.py --prefix deepseek-230 --root /path/to/output
  python aggregate_patches.py --prefix deepseek-230 --dry-run
        """,
    )

    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix to match experiment directories",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help=f"Root directory containing experiments (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--eval-script",
        type=str,
        default=str(DEFAULT_EVAL_SCRIPT),
        help=f"Path to evaluation script (default: {DEFAULT_EVAL_SCRIPT})",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help=f"Path to input.json (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Max workers for evaluation (default: 32)",
    )
    parser.add_argument(
        "--max-turn",
        type=int,
        default=None,
        help="Maximum turn number to aggregate (inclusive, default: no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only aggregate patches, skip evaluation",
    )

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    eval_script = Path(args.eval_script).resolve()
    input_file = Path(args.input_file).resolve()

    # Validate paths
    if not root_dir.exists():
        logger.error(f"Root directory does not exist: {root_dir}")
        sys.exit(1)

    if not eval_script.exists():
        logger.error(f"Evaluation script not found: {eval_script}")
        sys.exit(1)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    # Find matching experiment directories
    exp_dirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith(args.prefix)], key=lambda x: x.name
    )

    if not exp_dirs:
        logger.error(f"No directories found matching prefix '{args.prefix}' in {root_dir}")
        sys.exit(1)

    logger.info(f"Found {len(exp_dirs)} experiment directories matching prefix '{args.prefix}'")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Evaluation script: {eval_script}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Max turn: {args.max_turn if args.max_turn is not None else 'no limit'}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # Process each experiment
    success_count = 0
    fail_count = 0

    for exp_dir in exp_dirs:
        result = process_experiment(
            exp_dir=exp_dir,
            eval_script=eval_script,
            input_file=input_file,
            max_workers=args.max_workers,
            max_turn=args.max_turn,
            dry_run=args.dry_run,
        )

        if result:
            success_count += 1
        else:
            fail_count += 1

    logger.info("=" * 60)
    logger.info("Processing complete")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {fail_count}")
    logger.info("=" * 60)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
