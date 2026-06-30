#!/usr/bin/env python3
"""
Reorder a preds.json file by the numeric portion of each VUL4J ID.

Usage:
    python sort_preds.py --input /path/to/preds.json
    python sort_preds.py --input /path/to/preds.json --output /path/to/sorted_preds.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


def extract_vul4j_number(vul_id: str) -> int:
    """
    Extract the numeric portion from a VUL4J ID.

    Args:
        vul_id: VUL4J ID string, such as "VUL4J-1" or "VUL4J-100"

    Returns:
        The extracted integer, such as 1 or 100.
    """
    match = re.search(r"VUL4J-(\d+)", vul_id)
    if match:
        return int(match.group(1))
    return 0


def sort_preds_by_vul4j_id(data: dict[str, Any]) -> dict[str, Any]:
    """
    Sort preds.json data by the numeric portion of each VUL4J ID.

    Args:
        data: Original preds.json content.

    Returns:
        Sorted data.
    """
    # Collect all keys and sort them by the numeric VUL4J suffix.
    sorted_keys = sorted(data.keys(), key=extract_vul4j_number)

    # Build an ordered dictionary in the sorted order.
    sorted_data = {}
    for key in sorted_keys:
        sorted_data[key] = data[key]

    return sorted_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder a preds.json file by ascending numeric VUL4J ID.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input preds.json file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (defaults to overwriting the input file)",
    )
    parser.add_argument(
        "--backup",
        "-b",
        action="store_true",
        help="Create a backup before overwriting the input file",
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output if args.output else input_path

    # Check whether the input file exists.
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}")
        return 1

    # Load the original data.
    print(f"Reading file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Original entry count: {len(data)}")

    # Sort the data.
    print("Sorting by VUL4J ID...")
    sorted_data = sort_preds_by_vul4j_id(data)

    # Show a few keys before and after sorting.
    original_keys = list(data.keys())[:5]
    sorted_keys = list(sorted_data.keys())[:5]
    print(f"First 5 keys before sorting: {original_keys}")
    print(f"First 5 keys after sorting: {sorted_keys}")

    # Create a backup if requested.
    if args.backup and output_path == input_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        print(f"Creating backup: {backup_path}")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Write the sorted data.
    print(f"Writing file: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)

    print("Sorting complete.")
    return 0


if __name__ == "__main__":
    exit(main())
