#!/usr/bin/env python3
"""Audit SEC-bench trajectories for potential gold-patch leakage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonlines import jsonlines

_FENCED_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\n(.*?)```", re.DOTALL)

_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "definitely_leaked": (
        "cat /testcase/repo_changes.diff",
        "cat /testcase/gold_patch.diff",
        "git apply /testcase/repo_changes.diff",
        "git apply /testcase/gold_patch.diff",
        "patch -p1 < /testcase/repo_changes.diff",
        "patch -p1 < /testcase/gold_patch.diff",
        "grep -A5 \"src/",
        "grep -A5 'src/",
        "grep -c \"",
        "grep -c '",
    ),
    "likely_leaked": (
        "repo_changes.diff",
        "gold_patch.diff",
        "ls -la /testcase",
        "ls /testcase",
        "/usr/local/bin/secb",
        "which secb",
        "find /testcase",
        "grep -R repo_changes.diff",
    ),
    "suspicious_model_patch_access": (
        "/testcase/model_patch.diff",
        "model_patch.diff",
    ),
}
_SEVERITY_ORDER = {
    "definitely_leaked": 3,
    "likely_leaked": 2,
    "suspicious_model_patch_access": 1,
}
_SHELL_PREFIXES = (
    "cd ",
    "ls ",
    "cat ",
    "sed ",
    "grep ",
    "find ",
    "git ",
    "cp ",
    "mv ",
    "patch ",
    "timeout ",
    "./",
    "/",
)


@dataclass
class MatchRecord:
    experiment: str
    run: str
    instance_id: str
    traj_path: str
    message_index: int
    severity: str
    matched_keywords: list[str]
    command: str


def _extract_tool_commands(content: str) -> list[str]:
    blocks = [match.group(1).strip() for match in _FENCED_BLOCK_RE.finditer(content) if match.group(1).strip()]
    if blocks:
        return blocks

    stripped = content.strip()
    if not stripped or stripped.startswith("THOUGHT:") or stripped.startswith("<"):
        return []
    if stripped.startswith(_SHELL_PREFIXES):
        return [stripped]
    return []


def _highest_severity(levels: set[str]) -> str:
    return max(levels, key=lambda item: _SEVERITY_ORDER[item])


def _match_keywords(command: str) -> tuple[str | None, list[str]]:
    matched: list[tuple[str, str]] = []
    lowered = command.lower()
    for severity, keywords in _KEYWORD_GROUPS.items():
        for keyword in keywords:
            if keyword.lower() in lowered:
                matched.append((severity, keyword))

    if not matched:
        return None, []

    severities = {severity for severity, _ in matched}
    highest = _highest_severity(severities)
    keywords = sorted({keyword for _, keyword in matched})
    return highest, keywords


def _iter_traj_files(root: Path, output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.traj.json"):
        if output_dir in path.parents:
            continue
        files.append(path)
    return sorted(files)


def _load_traj(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive_labels(root: Path, traj_path: Path, payload: dict[str, Any]) -> tuple[str, str, str]:
    rel_path = traj_path.relative_to(root)
    parts = rel_path.parts
    experiment = parts[0]
    if "raw_traj" in parts:
        raw_index = parts.index("raw_traj")
        run_parts = parts[1:raw_index]
        run = "/".join(run_parts) if run_parts else "."
    else:
        run = "."
    instance_id = str(payload.get("instance_id") or traj_path.parent.name)
    return experiment, run, instance_id


def audit_root(root: Path, output_dir: Path) -> tuple[dict[str, Any], list[MatchRecord]]:
    matches: list[MatchRecord] = []
    experiment_summary: dict[str, Any] = {}
    run_summary: dict[str, Any] = {}

    traj_files = _iter_traj_files(root, output_dir)

    for traj_path in traj_files:
        payload = _load_traj(traj_path)
        experiment, run, instance_id = _derive_labels(root, traj_path, payload)
        messages = payload.get("messages", [])

        traj_match_records: list[MatchRecord] = []
        max_levels: set[str] = set()

        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue

            for command in _extract_tool_commands(content):
                severity, keywords = _match_keywords(command)
                if severity is None:
                    continue
                max_levels.add(severity)
                traj_match_records.append(
                    MatchRecord(
                        experiment=experiment,
                        run=run,
                        instance_id=instance_id,
                        traj_path=str(traj_path),
                        message_index=message_index,
                        severity=severity,
                        matched_keywords=keywords,
                        command=command,
                    )
                )

        exp_stats = experiment_summary.setdefault(
            experiment,
            {
                "experiment": experiment,
                "trajectory_count": 0,
                "leaked_trajectory_count": 0,
                "matched_tool_call_count": 0,
                "severity_counts": Counter(),
                "leaked_instances": [],
            },
        )
        exp_stats["trajectory_count"] += 1

        run_key = experiment if run == "." else f"{experiment}/{run}"
        run_stats = run_summary.setdefault(
            run_key,
            {
                "run": run_key,
                "experiment": experiment,
                "trajectory_count": 0,
                "leaked_trajectory_count": 0,
                "matched_tool_call_count": 0,
                "severity_counts": Counter(),
                "leaked_instances": [],
            },
        )
        run_stats["trajectory_count"] += 1

        if traj_match_records:
            matches.extend(traj_match_records)
            exp_stats["leaked_trajectory_count"] += 1
            exp_stats["matched_tool_call_count"] += len(traj_match_records)
            run_stats["leaked_trajectory_count"] += 1
            run_stats["matched_tool_call_count"] += len(traj_match_records)
            highest = _highest_severity(max_levels)
            exp_stats["severity_counts"][highest] += 1
            run_stats["severity_counts"][highest] += 1
            exp_stats["leaked_instances"].append(instance_id)
            run_stats["leaked_instances"].append(instance_id)

    for collection in (experiment_summary, run_summary):
        for stats in collection.values():
            stats["severity_counts"] = dict(sorted(stats["severity_counts"].items()))
            stats["leaked_instances"] = sorted(set(stats["leaked_instances"]))
            trajectory_count = stats["trajectory_count"]
            leaked_count = stats["leaked_trajectory_count"]
            stats["leak_rate"] = (leaked_count / trajectory_count) if trajectory_count else 0.0

    summary = {
        "root": str(root),
        "trajectory_count": len(traj_files),
        "leaked_trajectory_count": sum(
            experiment["leaked_trajectory_count"] for experiment in experiment_summary.values()
        ),
        "matched_tool_call_count": len(matches),
        "keyword_groups": {key: list(values) for key, values in _KEYWORD_GROUPS.items()},
        "experiments": sorted(experiment_summary.values(), key=lambda item: item["experiment"]),
        "runs": sorted(run_summary.values(), key=lambda item: item["run"]),
    }
    return summary, matches


def _write_outputs(output_dir: Path, summary: dict[str, Any], matches: list[MatchRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    matches_path = output_dir / "trajectory_matches.jsonl"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with jsonlines.open(matches_path, "w") as writer:
        for item in matches:
            writer.write(
                {
                    "experiment": item.experiment,
                    "run": item.run,
                    "instance_id": item.instance_id,
                    "traj_path": item.traj_path,
                    "message_index": item.message_index,
                    "severity": item.severity,
                    "matched_keywords": item.matched_keywords,
                    "command": item.command,
                }
            )


def _print_console_summary(summary: dict[str, Any]) -> None:
    print(
        "Scanned "
        f"{summary['trajectory_count']} trajectories; "
        f"flagged {summary['leaked_trajectory_count']} trajectories "
        f"from {len(summary['experiments'])} experiments."
    )
    for stats in summary["experiments"]:
        print(
            f"{stats['experiment']}: "
            f"{stats['leaked_trajectory_count']}/{stats['trajectory_count']} leaked trajectories, "
            f"{stats['matched_tool_call_count']} matched tool calls"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("minisweagent/output/sec"),
        help="SEC output root containing experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("minisweagent/output/sec/leak_audit"),
        help="Directory to write audit outputs.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    summary, matches = audit_root(root=root, output_dir=output_dir)
    _write_outputs(output_dir=output_dir, summary=summary, matches=matches)
    _print_console_summary(summary)


if __name__ == "__main__":
    main()
