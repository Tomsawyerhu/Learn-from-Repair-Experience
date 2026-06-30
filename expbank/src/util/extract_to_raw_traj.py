#!/usr/bin/env python3
"""
Extract all messages (system, user, assistant) from trajectory JSON files.

Extracts system, user and assistant dialogue content from trajectories of mini-agent solving patchEval problems.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def get_test_succ_set(result_path="./evaluation/summary.json"):
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
            succ_set = set()
            for succ_list in result_data["strict_evaluation"]["successful_cves"].values():
                succ_set.update(succ_list)

            print(f"total success id: {len(succ_set)}")
            return succ_set

    else:
        print("got no result file")
        return None


def load_vulnerability_info(input_json_path):
    """Load vulnerability information from the input JSON file."""
    if not os.path.exists(input_json_path):
        print(f"Warning: Input JSON file not found at {input_json_path}", file=sys.stderr)
        return {}

    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vul_info_map = {}
        for item in data:
            cve_id = item.get("cve_id")
            if cve_id:
                vul_info_map[cve_id] = {
                    "cve_id": cve_id,
                    "cve_description": item.get("cve_description"),
                    "cwe_info": item.get("cwe_info"),
                    "vul_func": item.get("vul_func"),
                    "fix_func": item.get("fix_func"),
                    "programming_language": item.get("programing_language"),
                }
        return vul_info_map
    except Exception as e:
        print(f"Error loading vulnerability info: {e}", file=sys.stderr)
        return {}


def extract_all_messages(messages):
    """Extract all messages (system, user, assistant) with role, content, and optional tool_calls.

    1. Iterate through each message dictionary in the messages list
    2. For each message, extract role and content fields
    3. If role is 'assistant', also extract tool_calls field (refer to original function for extraction logic)
    4. Compose dictionary with role, content, tool_calls (if assistant) for each message, add to result list
    5. Return result list
    """
    traj_list = []
    for message in messages:
        netraj = {
            "role": message["role"],
            "content": message["content"],
        }
        if message["role"] == "assistant":
            netraj["tool_calls"] = message["extra"]["tool_calls"]
        traj_list.append(netraj)

    return traj_list


def process_json_file(json_path, output_dir, succ_set, vul_info_map):
    """Process a single JSON file and write extracted data."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}", file=sys.stderr)
        return

    messages = data.get("messages")
    if not isinstance(messages, list):
        print(f"Warning: No messages list in {json_path}", file=sys.stderr)
        return

    # all_data = extract_all_messages(messages)
    # If all_data is empty, you can warn and continue because the file may simply contain no messages.
    # Note: the original function extracted only assistant messages; this version extracts all messages.
    all_data = extract_all_messages(messages)

    stem = json_path.stem.split(".")[0]

    if stem in succ_set:
        output_path = output_dir / f"{stem}_success.json"
    else:
        output_path = output_dir / f"{stem}_fail.json"

    # Get vulnerability information
    vul_info = vul_info_map.get(
        stem,
        {
            "cve_id": stem,
            "cve_description": None,
            "cwe_info": None,
            "vul_func": None,
            "fix_func": None,
            "programming_language": None,
        },
    )
    vul_info.pop("fix_func", None)

    # Build the new data structure
    final_data = {"traj": all_data, "vulnerability_info": vul_info}

    # TODO: Write final_data to output_path using json.dump
    # After successful write, print hint message, e.g.: f"Extracted {len(all_data)} messages to {output_path}"
    # Note: If all_data is empty, you can choose to skip writing or write empty list

    if all_data:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    final_data,
                    f,
                    ensure_ascii=False,  # Preserve non-ASCII characters
                    indent=4,  # Indent with 4 spaces
                    # sort_keys=True      # Sort keys in ascending order
                )
            print(f"Extracted {len(all_data)} messages to {output_path}")
        except Exception as e:
            print(f"error:{e}, extract failed")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all messages (system, user, assistant) from trajectory JSON files."
    )
    parser.add_argument(
        "--input_dir",
        default="./raw_traj",
        help="Root directory containing CVE‑prefixed subdirectories (default: %(default)s)",
    )
    parser.add_argument("--prefix", default="CVE", help="Prefix of subdirectories to scan (default: %(default)s)")
    parser.add_argument(
        "--output_dir",
        default="./extracted",
        help="Directory where extracted JSON files will be written (default: %(default)s)",
    )
    parser.add_argument(
        "--test_result",
        default="./evaluation/summary.json",
        help="File that contains test result (json style) (default: %(default)s)",
    )
    parser.add_argument(
        "--input_json",
        default="./datasets/input.json",
        help="Path to input.json containing vulnerability info (default: %(default)s)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find subdirectories with the given prefix
    subdirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith(args.prefix)]
    if not subdirs:
        print(f"No subdirectories starting with '{args.prefix}' found in {input_dir}", file=sys.stderr)
        sys.exit(0)

    test_result = Path(args.test_result).expanduser().resolve()
    succ_set = get_test_succ_set(test_result)

    input_json_path = Path(args.input_json).expanduser().resolve()
    vul_info_map = load_vulnerability_info(input_json_path)

    total_processed = 0
    for subdir in subdirs:
        # Look for JSON files (assume .traj.json or .json)
        json_files = list(subdir.glob("*.traj.json"))
        if not json_files:
            print(f"No JSON files found in {subdir}", file=sys.stderr)
            continue
        for json_file in json_files:
            process_json_file(json_file, output_dir, succ_set, vul_info_map)
            total_processed += 1

    print(f"\nDone. Processed {total_processed} JSON file(s).")


if __name__ == "__main__":
    main()
