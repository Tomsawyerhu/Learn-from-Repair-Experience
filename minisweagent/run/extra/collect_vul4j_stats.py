#!/usr/bin/env python3
"""
Collect and summarize evaluation results from a VUL4J evaluation logs directory.

This follows the same result collection approach used in
minisweagent/run/extra/vul4j_evaluate.py.

Usage:
    python collect_vul4j_stats.py --logs-dir /path/to/evaluation/logs
    python collect_vul4j_stats.py --logs-dir /path/to/evaluation/logs --output /path/to/output.json
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalResult:
    """Evaluation result for a single vulnerability instance."""

    vul_id: str
    status: str  # success, empty_patch, patch_apply_failed, compilation_failed, tests_failed, timeout, error
    patch_applied: bool = False
    compiled: bool = False
    tests_passed: bool = False
    error_message: str = ""
    duration: float = 0.0
    overall_metrics: dict = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Aggregate evaluation statistics."""

    total: int = 0
    success: int = 0
    empty_patch: int = 0
    not_a_diff_patch: int = 0
    invalid_diff_format: int = 0
    patch_apply_failed: int = 0
    compilation_failed: int = 0
    tests_failed: int = 0
    timeout: int = 0
    error: int = 0
    skipped: int = 0
    duration: float = 0.0

    def get_success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.success / (self.total - self.skipped)) * 100

    def to_dict(self) -> dict:
        return asdict(self)


def extract_vul4j_number(vul_id: str) -> int:
    """Extract the numeric portion from a VUL4J ID."""
    match = re.search(r"VUL4J-(\d+)", vul_id)
    if match:
        return int(match.group(1))
    return 0


def parse_test_results(test_output: str) -> bool:
    """
    Parse test output and determine whether tests passed.

    Mirrors the _parse_test_results logic in vul4j_evaluate.py.
    """
    try:
        # Try to parse the payload as JSON.
        data = json.loads(test_output)

        # Check for failing tests in the VUL4J-specific structure.
        if isinstance(data, dict):
            # VUL4J-specific structure: tests.overall_metrics.number_failing / number_error
            tests = data.get("tests", {})
            if isinstance(tests, dict):
                metrics = tests.get("overall_metrics", {})
                if isinstance(metrics, dict):
                    number_failing = metrics.get("number_failing", 0)
                    number_error = metrics.get("number_error", 0)
                    if number_failing > 0 or number_error > 0:
                        return False

            # Check common result fields.
            for key in ["failed", "failures", "errors", "Failing"]:
                if key in data and data[key] > 0:
                    return False
            return True
        elif isinstance(data, list):
            # Check result lists when the payload is a list.
            for item in data:
                if isinstance(item, dict):
                    for key in ["status", "result"]:
                        if key in item:
                            status = item[key].lower() if isinstance(item[key], str) else item[key]
                            if status in ["fail", "failed", "error"]:
                                return False
            return True
    except (json.JSONDecodeError, TypeError):
        pass

    # If parsing fails, search the raw output for failure markers.
    failed_patterns = [
        r"FAILED",
        r"FAILURE",
        r"ERROR",
        r"Tests run:.*Failures: [1-9]",
        r"Tests run:.*Errors: [1-9]",
    ]
    for pattern in failed_patterns:
        if re.search(pattern, test_output, re.IGNORECASE):
            return False

    # No failure markers found, assume the tests passed.
    return True


def load_results_from_logs(logs_dir: Path) -> dict[str, EvalResult]:
    """
    Load all evaluation results from the logs directory.

    Args:
        logs_dir: Path to the logs directory.

    Returns:
        A mapping from vulnerability ID to EvalResult.
    """
    results = {}

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    # Iterate through all vulnerability subdirectories.
    for vul_dir in logs_dir.iterdir():
        if not vul_dir.is_dir():
            continue

        vul_id = vul_dir.name
        result_file = vul_dir / "result.json"
        testing_results_file = vul_dir / "testing_results.json"

        overall_metrics = {}
        tests_passed = False
        status = "unknown"

        # Re-evaluate test success using testing_results.json when available.
        if testing_results_file.exists():
            try:
                with open(testing_results_file, "r", encoding="utf-8") as f:
                    test_data = json.load(f)
                tests = test_data.get("tests", {})
                if isinstance(tests, dict):
                    overall_metrics = tests.get("overall_metrics", {})

                # Reuse parse_test_results to determine the final test outcome.
                test_output_str = json.dumps(test_data)
                tests_passed = parse_test_results(test_output_str)
            except Exception:
                pass

        if result_file.exists():
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Recompute the final status from the loaded data.
                old_status = data.get("status", "unknown")
                compiled = data.get("compiled", False)

                if not compiled:
                    status = "compilation_failed"
                elif not data.get("patch_applied", False):
                    status = "patch_apply_failed"
                elif tests_passed:
                    status = "success"
                else:
                    status = "tests_failed"

                result = EvalResult(
                    vul_id=vul_id,
                    status=status,
                    patch_applied=data.get("patch_applied", False),
                    compiled=compiled,
                    tests_passed=tests_passed,
                    error_message=data.get("error_message", ""),
                    duration=data.get("duration", 0.0),
                    overall_metrics=overall_metrics,
                )
                results[vul_id] = result
            except Exception as e:
                print(f"Warning: Failed to read result for {vul_id}: {e}")

    return results


def generate_summary(results: dict[str, EvalResult]) -> EvalSummary:
    """Generate aggregate summary statistics."""
    summary = EvalSummary(total=len(results))

    for vul_id, result in results.items():
        if result.status == "success":
            summary.success += 1
        elif result.status == "empty_patch":
            summary.empty_patch += 1
        elif result.status == "not_a_diff_patch":
            summary.not_a_diff_patch += 1
        elif result.status == "invalid_diff_format":
            summary.invalid_diff_format += 1
        elif result.status == "patch_apply_failed":
            summary.patch_apply_failed += 1
        elif result.status == "compilation_failed":
            summary.compilation_failed += 1
        elif result.status == "tests_failed":
            summary.tests_failed += 1
        elif result.status == "timeout":
            summary.timeout += 1
        else:
            summary.error += 1

        # Accumulate total duration.
        summary.duration += result.duration

    return summary


def save_results_json(results: dict[str, EvalResult], output_path: Path) -> None:
    """Save detailed results as JSON."""
    results_data = {}
    for vul_id, result in results.items():
        result_dict = {
            "status": result.status,
            "patch_applied": result.patch_applied,
            "compiled": result.compiled,
            "tests_passed": result.tests_passed,
            "error_message": result.error_message,
            "duration": result.duration,
        }
        # Include overall_metrics when available.
        if result.overall_metrics:
            result_dict["overall_metrics"] = result.overall_metrics
        results_data[vul_id] = result_dict

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"Results saved to {output_path}")


def save_summary_json(summary: EvalSummary, output_path: Path) -> None:
    """Save the summary as JSON."""
    # Add a formatted success rate.
    summary_dict = summary.to_dict()
    summary_dict["success_rate"] = f"{summary.get_success_rate():.2f}%"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"Summary saved to {output_path}")


def save_summary_text(summary: EvalSummary, output_path: Path) -> None:
    """Save a plain-text summary report."""
    lines = [
        "=" * 60,
        "           VUL4J Evaluation Summary Report",
        "=" * 60,
        f"Total Instances: {summary.total}",
        f"Evaluated: {summary.total - summary.skipped}",
        f"",
        "Results Breakdown:",
        "-" * 40,
        f"✅ Success (patch applied + compiled + tests passed): {summary.success}",
        f"⚠️  Empty Patch: {summary.empty_patch}",
        f"⚠️  Not a Diff Patch: {summary.not_a_diff_patch}",
        f"⚠️  Invalid Diff Format: {summary.invalid_diff_format}",
        f"❌ Patch Apply Failed: {summary.patch_apply_failed}",
        f"❌ Compilation Failed: {summary.compilation_failed}",
        f"❌ Tests Failed: {summary.tests_failed}",
        f"⏱️  Timeout: {summary.timeout}",
        f"🔴 Error: {summary.error}",
        f"",
        "-" * 40,
        f"Success Rate: {summary.get_success_rate():.2f}% ({summary.success}/{summary.total - summary.skipped})",
        f"Total Duration: {summary.duration:.2f}s",
        "=" * 60,
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Summary report saved to {output_path}")

    # Also print the report to the console.
    print("\n" + "\n".join(lines))


def save_errors_json(results: dict[str, EvalResult], output_path: Path) -> None:
    """Save error cases as JSON."""
    errors = {}
    for vul_id, result in results.items():
        if result.status not in ["success", "empty_patch"]:
            errors[vul_id] = {
                "status": result.status,
                "error_message": result.error_message,
                "patch_applied": result.patch_applied,
                "compiled": result.compiled,
            }

    if errors:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)

        print(f"Errors saved to {output_path}")


def print_summary_table(summary: EvalSummary) -> None:
    """Print a summary table."""
    table_lines = [
        "┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓",
        "┃ Status                 ┃ Count ┃ Percentage ┃",
        "┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩",
    ]

    total = summary.total - summary.skipped
    if total == 0:
        total = 1

    rows = [
        ("✅ Success", str(summary.success), f"{summary.success / total * 100:.1f}%"),
        ("⚠️  Empty Patch", str(summary.empty_patch), f"{summary.empty_patch / total * 100:.1f}%"),
        ("⚠️  Not a Diff Patch", str(summary.not_a_diff_patch), f"{summary.not_a_diff_patch / total * 100:.1f}%"),
        (
            "⚠️  Invalid Diff Format",
            str(summary.invalid_diff_format),
            f"{summary.invalid_diff_format / total * 100:.1f}%",
        ),
        (
            "❌ Patch Apply Failed",
            str(summary.patch_apply_failed),
            f"{summary.patch_apply_failed / total * 100:.1f}%",
        ),
        (
            "❌ Compilation Failed",
            str(summary.compilation_failed),
            f"{summary.compilation_failed / total * 100:.1f}%",
        ),
        ("❌ Tests Failed", str(summary.tests_failed), f"{summary.tests_failed / total * 100:.1f}%"),
        ("⏱️  Timeout", str(summary.timeout), f"{summary.timeout / total * 100:.1f}%"),
        ("🔴 Error", str(summary.error), f"{summary.error / total * 100:.1f}%"),
    ]

    for row in rows:
        table_lines.append(f"│ {row[0]:<22} │ {row[1]:>5} │ {row[2]:>10} │")

    table_lines.append("└────────────────────────┴───────┴────────────┘")
    table_lines.append("")
    table_lines.append(f"Success Rate: {summary.get_success_rate():.2f}% ({summary.success}/{total})")

    print("\n" + "\n".join(table_lines))


def print_status_breakdown(results: dict[str, EvalResult]) -> None:
    """Print vulnerability IDs grouped by status."""
    status_groups = defaultdict(list)

    for vul_id, result in results.items():
        status_groups[result.status].append(vul_id)

    print("\n" + "=" * 60)
    print("Status Breakdown by VUL4J-ID")
    print("=" * 60)

    for status in [
        "success",
        "empty_patch",
        "not_a_diff_patch",
        "invalid_diff_format",
        "patch_apply_failed",
        "compilation_failed",
        "tests_failed",
        "timeout",
        "error",
    ]:
        if status in status_groups and status_groups[status]:
            vul_ids = sorted(status_groups[status], key=extract_vul4j_number)
            print(f"\n{status.upper()} ({len(vul_ids)}):")
            print(f"  {', '.join(vul_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and summarize results from a VUL4J evaluation logs directory.")
    parser.add_argument(
        "--logs-dir",
        "-l",
        required=True,
        type=Path,
        help="Path to the logs directory",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (defaults to the parent of logs-dir)",
    )

    args = parser.parse_args()

    logs_dir = args.logs_dir

    # Determine the output directory.
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = logs_dir.parent

    # Create the output directory.
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {logs_dir}")

    # Load the raw results.
    results = load_results_from_logs(logs_dir)
    print(f"Loaded {len(results)} results")

    # Generate the summary.
    summary = generate_summary(results)

    # Save all reports.
    save_results_json(results, output_dir / "results.json")
    save_summary_json(summary, output_dir / "summary.json")
    save_summary_text(summary, output_dir / "summary.txt")
    save_errors_json(results, output_dir / "errors.json")

    # Print the summary to stdout.
    print_summary_table(summary)
    print_status_breakdown(results)

    print(f"\nAll reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
