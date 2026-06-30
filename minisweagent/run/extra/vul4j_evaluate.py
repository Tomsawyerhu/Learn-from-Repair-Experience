#!/usr/bin/env python3
"""
VUL4J automated evaluation script

This script applies model-generated patches inside Docker containers and checks whether the vulnerability is fixed.

Usage:
    python vul4j_evaluate.py --preds /path/to/preds.json
    python vul4j_evaluate.py --preds /path/to/preds.json --workers 4
    python vul4j_evaluate.py --preds /path/to/preds.json --filter-vul-ids VUL4J-1,VUL4J-2
"""

import argparse
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import docker
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# ==================== Configuration Constants ====================

VUL4J_IMAGE = "bqcuongas/vul4j"
DEFAULT_TIMEOUT = 600  # 10 minutes
DEFAULT_WORKERS = 2
CONTAINER_WORKDIR = "/tmp/vul4j"

# ==================== Data Class Definitions ====================


@dataclass
class EvalResult:
    """Evaluation result for a single vulnerability instance"""

    vul_id: str
    status: str  # success, empty_patch, patch_apply_failed, compilation_failed, tests_failed, timeout, error
    patch_applied: bool = False
    compiled: bool = False
    tests_passed: bool = False
    error_message: str = ""
    compile_output: str = ""
    test_output: str = ""
    overall_metrics: dict = field(default_factory=dict)  # test metrics
    duration: float = 0.0


@dataclass
class EvalConfig:
    """Evaluation configuration"""

    preds_path: Path
    output_dir: Path
    workers: int = DEFAULT_WORKERS
    timeout: int = DEFAULT_TIMEOUT
    filter_vul_ids: list[str] = field(default_factory=list)
    skip_existing: bool = True
    verbose: bool = False


@dataclass
class EvalSummary:
    """Summary statistics for the evaluation"""

    total: int = 0
    success: int = 0
    empty_patch: int = 0
    not_a_diff_patch: int = 0  # Invalid patch that does not start with diff
    invalid_diff_format: int = 0  # Malformed diff format
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


# ==================== Logging Setup ====================

console = Console()


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging"""
    logger = logging.getLogger("vul4j_eval")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

    return logger


# ==================== Docker Management ====================


class DockerManager:
    """Docker container manager"""

    def __init__(self, image: str = VUL4J_IMAGE, logger: Optional[logging.Logger] = None):
        self.image = image
        self.logger = logger or logging.getLogger(__name__)
        self.client = docker.from_env()
        self._lock = threading.Lock()

    def start_container(self, container_name: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
        """Start a Docker container"""
        with self._lock:
            try:
                # Ensure the image is available locally, pull if missing
                try:
                    self.client.images.get(self.image)
                except docker.errors.NotFound:
                    self.logger.info(f"Pulling image {self.image}...")
                    self.client.images.pull(self.image)
                    self.logger.info(f"Image {self.image} pulled successfully")

                container = self.client.containers.run(
                    self.image,
                    "/bin/bash",
                    name=container_name,
                    detach=True,
                    tty=True,
                    stdin_open=True,
                    mem_limit="4g",
                    shm_size="2g",
                )
                return container.id
            except Exception as e:
                self.logger.error(f"Failed to start container: {e}")
                return None

    def exec_command(
        self, container_id: str, command: str, cwd: str = "/", timeout: int = DEFAULT_TIMEOUT
    ) -> dict[str, Any]:
        """Run a command inside the container"""
        try:
            container = self.client.containers.get(container_id)

            # Build the full command string (with timeout if provided)
            # Use bash -c to avoid quoting issues
            if timeout > 0:
                full_cmd = f"cd {cwd} && timeout {timeout} {command}"
            else:
                full_cmd = f"cd {cwd} && {command}"

            result = container.exec_run(
                f"/bin/bash -c {repr(full_cmd)}",
                demux=False,  # Do not use demux to avoid decoding issues
            )

            # Capture the output
            output = result.output.decode("utf-8", errors="replace") if result.output else ""

            # Handle timeouts
            returncode = result.exit_code
            if returncode == 124:
                output += "\n[TIMEOUT] Command execution exceeded the time limit"

            return {
                "returncode": returncode,
                "stdout": output,
                "stderr": "",
                "output": output,
            }
        except docker.errors.NotFound:
            return {"returncode": -1, "stdout": "", "stderr": "Container not found", "output": "Container not found"}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e), "output": str(e)}

    def stop_container(self, container_id: str) -> None:
        """Stop and remove the container"""
        with self._lock:
            try:
                container = self.client.containers.get(container_id)
                container.stop(timeout=30)
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except Exception as e:
                self.logger.debug(f"Error stopping container: {e}")

    def is_image_available(self) -> bool:
        """Check whether the image is available"""
        try:
            self.client.images.get(self.image)
            return True
        except docker.errors.NotFound:
            return False
        except Exception:
            return False


# ==================== Core Evaluation Logic ====================


class Vul4jEvaluator:
    """VUL4J vulnerability repair evaluator"""

    def __init__(self, config: EvalConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.docker = DockerManager(logger=self.logger)
        self._results: dict[str, EvalResult] = {}

    def load_predictions(self) -> dict:
        """Load predicted patches"""
        preds_path = self.config.preds_path
        if not preds_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {preds_path}")

        with open(preds_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter the requested vulnerabilities
        if self.config.filter_vul_ids:
            data = {k: v for k, v in data.items() if k in self.config.filter_vul_ids}

        return data

    def _get_container_name(self, vul_id: str) -> str:
        """Generate a container name"""
        return f"vul4j-eval-{vul_id.lower()}-{secrets.token_hex(4)}"

    def _write_patch_file(self, container_id: str, patch_content: str) -> bool:
        """Write patch content into a file inside the container"""
        # Create a temporary tar file
        import tarfile
        import io

        try:
            container = self.docker.client.containers.get(container_id)

            # Create the tar archive contents
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                # Add the patch content to the tar archive
                tar_info = tarfile.TarInfo(name="fix.patch")
                tar_info.size = len(patch_content.encode("utf-8"))
                tar.addfile(tar_info, io.BytesIO(patch_content.encode("utf-8")))

            tar_buffer.seek(0)

            # Use put_archive to extract the tar file into the container root
            container.put_archive("/", tar_buffer)

            # Verify that the file exists
            check_result = self.docker.exec_command(container_id, "ls -la /fix.patch", timeout=30)
            if check_result["returncode"] != 0:
                self.logger.error(f"Patch file verification failed: {check_result['output']}")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to write patch file: {e}")
            return False

    def _checkout_vulnerability(self, container_id: str, vul_id: str) -> bool:
        """Check out the vulnerable code"""
        work_dir = f"{CONTAINER_WORKDIR}/{vul_id}"

        # Clean up any existing directory
        self.docker.exec_command(container_id, f"rm -rf {work_dir}", timeout=30)

        # Check out the vulnerable code (vul4j creates the directory automatically)
        cmd = f"vul4j checkout --id {vul_id} -d {work_dir}"
        result = self.docker.exec_command(container_id, cmd, cwd="/", timeout=300)

        if result["returncode"] != 0:
            self.logger.error(
                f"Failed to checkout {vul_id}: returncode={result['returncode']}, output={result['output']}"
            )
            return False

        # Verify that the directory was created successfully
        check_result = self.docker.exec_command(container_id, f"ls -la {work_dir}/VUL4J", timeout=30)
        if check_result["returncode"] != 0:
            self.logger.error(f"VUL4J directory not created: {check_result['output']}")
            return False

        # Verify that the directory was created successfully
        check_result = self.docker.exec_command(container_id, f"ls -la {work_dir}/VUL4J", timeout=30)
        if check_result["returncode"] != 0:
            self.logger.error(f"VUL4J directory not created: {check_result['output']}")
            return False

        # Delete human_patch to avoid leakage
        self.docker.exec_command(container_id, f"rm -rf {work_dir}/VUL4J/human_patch", timeout=30)

        return True

    def _apply_patch(self, container_id: str, vul_id: str, patch_content: str) -> bool:
        """Apply the patch"""
        work_dir = f"{CONTAINER_WORKDIR}/{vul_id}"

        # Write the patch file
        if not self._write_patch_file(container_id, patch_content):
            self.logger.error(f"Failed to write patch file for {vul_id}")
            return False

        # Verify that the patch file exists
        check_result = self.docker.exec_command(container_id, "ls -la /fix.patch", timeout=30)
        if check_result["returncode"] != 0:
            self.logger.error(f"Patch file not found: {check_result['output']}")
            return False

        # Apply the patch
        patch_file = "/fix.patch"
        cmd = f"git apply {patch_file} --verbose"
        result = self.docker.exec_command(container_id, cmd, cwd=work_dir, timeout=60)

        if result["returncode"] != 0:
            self.logger.debug(f"Git apply failed: output={result['output']}")
            # Try applying the patch with the patch command
            cmd = f"patch -p1 < {patch_file}"
            result = self.docker.exec_command(container_id, cmd, cwd=work_dir, timeout=60)
            if result["returncode"] != 0:
                self.logger.debug(f"Patch command also failed: output={result['output']}")

        return result["returncode"] == 0

    def _compile(self, container_id: str, vul_id: str) -> tuple[bool, str]:
        """Compile the project"""
        work_dir = f"{CONTAINER_WORKDIR}/{vul_id}"

        cmd = f"vul4j compile -d {work_dir}"
        result = self.docker.exec_command(container_id, cmd, cwd="/", timeout=self.config.timeout)

        # Save the compilation output
        compile_output = result["output"]

        # Try reading the compilation log
        log_result = self.docker.exec_command(
            container_id, f"cat {work_dir}/VUL4J/compile.log 2>/dev/null || echo ''", timeout=30
        )
        if log_result["stdout"]:
            compile_output = log_result["stdout"]

        return result["returncode"] == 0, compile_output

    def _run_tests(self, container_id: str, vul_id: str) -> tuple[bool, str, dict]:
        """Run tests"""
        work_dir = f"{CONTAINER_WORKDIR}/{vul_id}"

        cmd = f"vul4j test -d {work_dir}"
        result = self.docker.exec_command(container_id, cmd, cwd="/", timeout=self.config.timeout)

        # Save the test output
        test_output = result["output"]
        overall_metrics = {}

        # Try reading the test results file
        log_result = self.docker.exec_command(
            container_id, f"cat {work_dir}/VUL4J/testing_results.json 2>/dev/null || echo ''", timeout=30
        )
        if log_result["stdout"]:
            try:
                test_data = json.loads(log_result["stdout"])
                test_output = json.dumps(test_data, indent=2)
                # Extract overall_metrics
                tests = test_data.get("tests", {})
                if isinstance(tests, dict):
                    overall_metrics = tests.get("overall_metrics", {})
            except json.JSONDecodeError:
                pass

        return result["returncode"] == 0, test_output, overall_metrics

    def _parse_test_results(self, test_output: str) -> bool:
        """Parse test results"""
        try:
            # Try parsing JSON
            data = json.loads(test_output)

            # Check for failing tests in the VUL4J-specific format
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

                # Check common result fields
                for key in ["failed", "failures", "errors", "Failing"]:
                    if key in data and data[key] > 0:
                        return False
                return True
            elif isinstance(data, list):
                # Check the test list
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

        # If parsing fails, search the output for failure markers
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

        # No failure found, treat the run as passed
        return True

    def _is_valid_patch(self, patch_content: str) -> tuple[bool, str]:
        """
        Check whether the patch content is valid

        Returns:
            (is_valid, reason): Whether it is valid and the reason
        """
        # Check whether the patch is empty
        if not patch_content or not patch_content.strip():
            return False, "empty_patch"

        # Check whether the patch starts with diff
        patch_stripped = patch_content.strip()
        if not patch_stripped.startswith("diff "):
            return False, "not_a_diff_patch"

        # Check whether file paths are present (a valid diff should contain --- a/ or +++ b/)
        if "--- a/" not in patch_content and "+++ b/" not in patch_content:
            return False, "invalid_diff_format"

        return True, ""

    def evaluate_single(self, vul_id: str, patch_content: str) -> EvalResult:
        """Evaluate a single vulnerability instance"""
        start_time = time.time()
        result = EvalResult(vul_id=vul_id, status="error")

        container_name = self._get_container_name(vul_id)

        # Check whether the patch is empty or invalid
        is_valid, reason = self._is_valid_patch(patch_content)
        if not is_valid:
            result.status = reason
            result.duration = time.time() - start_time
            self.logger.debug(f"Skipping {vul_id}: {reason}")
            return result

        container_id = None
        try:
            # Start the container
            self.logger.debug(f"Starting container for {vul_id}...")
            container_id = self.docker.start_container(container_name, timeout=self.config.timeout)
            if not container_id:
                result.status = "error"
                result.error_message = "Failed to start container"
                result.duration = time.time() - start_time
                return result

            # Check out the vulnerable code
            if not self._checkout_vulnerability(container_id, vul_id):
                result.status = "error"
                result.error_message = "Failed to checkout vulnerability"
                result.duration = time.time() - start_time
                self.docker.stop_container(container_id)
                return result

            work_dir = f"{CONTAINER_WORKDIR}/{vul_id}"

            # Apply the patch
            if not self._apply_patch(container_id, vul_id, patch_content):
                result.status = "patch_apply_failed"
                result.patch_applied = False
                result.duration = time.time() - start_time
                self.docker.stop_container(container_id)
                return result

            result.patch_applied = True

            # Compile
            compiled, compile_output = self._compile(container_id, vul_id)
            result.compile_output = compile_output

            if not compiled:
                result.status = "compilation_failed"
                result.duration = time.time() - start_time
                self.docker.stop_container(container_id)
                return result

            result.compiled = True

            # Run tests
            tests_passed, test_output, overall_metrics = self._run_tests(container_id, vul_id)
            result.test_output = test_output
            result.overall_metrics = overall_metrics

            # Important: always use _parse_test_results to decide whether tests truly passed
            # The vul4j test command may return 0 even when tests fail
            tests_passed = self._parse_test_results(test_output)

            result.tests_passed = tests_passed

            if tests_passed:
                result.status = "success"
            else:
                result.status = "tests_failed"

        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            self.logger.error(f"Error evaluating {vul_id}: {e}")
        finally:
            # Clean up the container
            try:
                if container_id:
                    self.docker.stop_container(container_id)
            except Exception:
                pass

        result.duration = time.time() - start_time
        return result

    def evaluate(self, predictions: dict) -> dict[str, EvalResult]:
        """Evaluate all vulnerability instances"""
        self._results = {}

        # Create the output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = self.config.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate each vulnerability
        items = list(predictions.items())

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Evaluating {len(items)} vulnerabilities...", total=len(items))

            for vul_id, pred in items:
                # Check whether to skip existing results
                if self.config.skip_existing:
                    result_file = logs_dir / vul_id / "result.json"
                    if result_file.exists():
                        self.logger.info(f"Skipping {vul_id} (already evaluated)")
                        progress.advance(task)
                        continue

                # Get the patch content
                patch_content = pred.get("model_patch", "")

                # Evaluate
                result = self.evaluate_single(vul_id, patch_content)
                self._results[vul_id] = result

                # Save detailed results
                self._save_result(vul_id, result, logs_dir)

                progress.advance(task)

        return self._results

    def _save_result(self, vul_id: str, result: EvalResult, logs_dir: Path) -> None:
        """Save a single evaluation result"""
        instance_dir = logs_dir / vul_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        # Save the result JSON
        result_data = {
            "vul_id": result.vul_id,
            "status": result.status,
            "patch_applied": result.patch_applied,
            "compiled": result.compiled,
            "tests_passed": result.tests_passed,
            "error_message": result.error_message,
            "duration": result.duration,
        }
        # Include overall_metrics when available
        if result.overall_metrics:
            result_data["overall_metrics"] = result.overall_metrics

        with open(instance_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)

        # Save the compilation output
        if result.compile_output:
            with open(instance_dir / "compile.log", "w", encoding="utf-8") as f:
                f.write(result.compile_output)

        # Save the test output
        if result.test_output:
            with open(instance_dir / "testing_results.json", "w", encoding="utf-8") as f:
                f.write(result.test_output)


# ==================== Report Generation ====================


class ReportGenerator:
    """Evaluation report generator"""

    def __init__(self, results: dict[str, EvalResult], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def generate_summary(self) -> EvalSummary:
        """Generate summary statistics"""
        summary = EvalSummary(total=len(self.results))

        for vul_id, result in self.results.items():
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

        return summary

    def save_results_json(self) -> None:
        """Save detailed results as JSON"""
        results_data = {}
        for vul_id, result in self.results.items():
            result_dict = {
                "status": result.status,
                "patch_applied": result.patch_applied,
                "compiled": result.compiled,
                "tests_passed": result.tests_passed,
                "error_message": result.error_message,
                "duration": result.duration,
            }
            # Include overall_metrics when available
            if result.overall_metrics:
                result_dict["overall_metrics"] = result.overall_metrics
            results_data[vul_id] = result_dict

        output_path = self.output_dir / "results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def save_summary_json(self, summary: EvalSummary) -> None:
        """Save the summary as JSON"""
        # Add a formatted success rate
        summary_dict = summary.to_dict()
        summary_dict["success_rate"] = f"{summary.get_success_rate():.2f}%"

        output_path = self.output_dir / "summary.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        self.logger.info(f"Summary saved to {output_path}")

    def save_summary_text(self, summary: EvalSummary) -> None:
        """Save a plain-text summary report"""
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
            "=" * 60,
        ]

        output_path = self.output_dir / "summary.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info(f"Summary report saved to {output_path}")

        # Also print to the console
        console.print("\n" + "\n".join(lines))

    def save_errors_json(self) -> None:
        """Save error cases"""
        errors = {}
        for vul_id, result in self.results.items():
            if result.status not in ["success", "empty_patch"]:
                errors[vul_id] = {
                    "status": result.status,
                    "error_message": result.error_message,
                    "patch_applied": result.patch_applied,
                    "compiled": result.compiled,
                }

        if errors:
            output_path = self.output_dir / "errors.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(errors, f, indent=2)

            self.logger.info(f"Errors saved to {output_path}")

    def print_summary_table(self, summary: EvalSummary) -> None:
        """Print the summary table"""
        table = Table(title="VUL4J Evaluation Results")

        table.add_column("Status", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right", style="magenta")
        table.add_column("Percentage", justify="right", style="green")

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
            table.add_row(*row)

        console.print(table)
        console.print(
            f"\n[bold green]Success Rate:[/bold green] {summary.get_success_rate():.2f}% ({summary.success}/{total})"
        )


# ==================== Main Program ====================


app = typer.Typer(help="VUL4J automated evaluation script", add_completion=False)


@app.command()
def main(
    preds: Path = typer.Option(
        ...,
        "--preds",
        "-p",
        help="Predicted patch file path (preds.json)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: evaluation beside preds.json)",
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    workers: int = typer.Option(DEFAULT_WORKERS, "--workers", "-w", help="Number of parallel evaluation workers"),
    timeout: int = typer.Option(DEFAULT_TIMEOUT, "--timeout", "-t", help="Timeout for a single evaluation (seconds)"),
    filter_vul_ids: str = typer.Option("", "--filter", "-f", help="Evaluate only the specified vulnerability IDs (comma-separated)"),
    skip_existing: bool = typer.Option(True, "--skip-existing", help="Skip vulnerabilities that have already been evaluated"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    no_skip: bool = typer.Option(False, "--no-skip", help="Do not skip vulnerabilities that have already been evaluated"),
) -> None:
    """Run VUL4J vulnerability repair evaluation"""

    # Set up logging
    logger = setup_logging(verbose)

    # Resolve the output directory
    if output is None:
        output = preds.parent / "evaluation"
    else:
        output = Path(output)

    # Resolve filtered vulnerability IDs
    filter_ids = []
    if filter_vul_ids:
        filter_ids = [x.strip() for x in filter_vul_ids.split(",") if x.strip()]

    # Resolve skip_existing
    skip = not no_skip

    # Build the configuration
    config = EvalConfig(
        preds_path=preds,
        output_dir=output,
        workers=workers,
        timeout=timeout,
        filter_vul_ids=filter_ids,
        skip_existing=skip,
        verbose=verbose,
    )

    logger.info(f"Loading predictions from: {preds}")
    logger.info(f"Output directory: {output}")
    logger.info(f"Workers: {workers}, Timeout: {timeout}s")

    # Check whether Docker is available
    try:
        docker_client = docker.from_env()
        docker_client.ping()
    except Exception as e:
        console.print(f"[red]Error: Docker is not available: {e}[/red]")
        sys.exit(1)

    # Create the evaluator
    evaluator = Vul4jEvaluator(config, logger)

    # Load predictions
    try:
        predictions = evaluator.load_predictions()
        logger.info(f"Loaded {len(predictions)} predictions")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    # Run the evaluation
    start_time = time.time()
    results = evaluator.evaluate(predictions)
    total_duration = time.time() - start_time

    # Generate reports
    report_gen = ReportGenerator(results, output)
    summary = report_gen.generate_summary()
    summary.duration = total_duration

    # Save reports
    report_gen.save_results_json()
    report_gen.save_summary_json(summary)
    report_gen.save_summary_text(summary)
    report_gen.save_errors_json()

    # Print the summary
    report_gen.print_summary_table(summary)

    logger.info(f"Evaluation completed in {total_duration:.2f}s")
    logger.info(f"Results saved to: {output}")


# ==================== Importable API ====================


def evaluate_vul4j(
    preds_path: str | Path,
    output_dir: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    timeout: int = DEFAULT_TIMEOUT,
    filter_vul_ids: list[str] | None = None,
    skip_existing: bool = True,
    verbose: bool = False,
) -> dict[str, EvalResult]:
    """
    Evaluate VUL4J vulnerability repair (importable API)

    Args:
        preds_path: Predicted patch file path
        output_dir: Output directory (default: evaluation beside preds.json)
        workers: Number of parallel workers
        timeout: Single-evaluation timeout (seconds)
        filter_vul_ids: List of vulnerability IDs to evaluate
        skip_existing: Whether to skip already evaluated vulnerabilities
        verbose: Whether to emit verbose logs

    Returns:
        dict[str, EvalResult]: Evaluation result for each vulnerability ID
    """
    preds_path = Path(preds_path)
    if output_dir is None:
        output_dir = preds_path.parent / "evaluation"
    else:
        output_dir = Path(output_dir)

    config = EvalConfig(
        preds_path=preds_path,
        output_dir=output_dir,
        workers=workers,
        timeout=timeout,
        filter_vul_ids=filter_vul_ids or [],
        skip_existing=skip_existing,
        verbose=verbose,
    )

    logger = setup_logging(verbose)
    evaluator = Vul4jEvaluator(config, logger)
    predictions = evaluator.load_predictions()
    results = evaluator.evaluate(predictions)

    return results


def get_evaluation_summary(results: dict[str, EvalResult]) -> EvalSummary:
    """Get summary statistics from evaluation results"""
    report_gen = ReportGenerator(results, Path("/tmp"))
    return report_gen.generate_summary()


def generate_report_from_existing_results(
    output_dir: str | Path,
    preds_path: str | Path | None = None,
) -> dict[str, EvalResult]:
    """
    Generate reports from existing evaluation results without rerunning evaluation

    Args:
        output_dir: evaluation output directory (including the logs subdirectory)
        preds_path: Optional preds.json path used to recover the full vulnerability list

    Returns:
        dict[str, EvalResult]: Evaluation result for each vulnerability ID
    """
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    # Read all existing evaluation results
    results = {}
    for vul_dir in logs_dir.iterdir():
        if vul_dir.is_dir():
            vul_id = vul_dir.name
            result_file = vul_dir / "result.json"
            testing_results_file = vul_dir / "testing_results.json"

            overall_metrics = {}
            tests_passed = False
            status = "unknown"

            # Try to read testing_results.json and re-evaluate whether the tests passed
            if testing_results_file.exists():
                try:
                    with open(testing_results_file, "r", encoding="utf-8") as f:
                        test_data = json.load(f)
                    tests = test_data.get("tests", {})
                    if isinstance(tests, dict):
                        overall_metrics = tests.get("overall_metrics", {})

                    # Re-evaluate using the _parse_test_results logic
                    test_output_str = json.dumps(test_data)
                    tests_passed = _parse_test_results_fixed(test_output_str)
                except Exception:
                    pass

            if result_file.exists():
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Re-evaluate the status
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
                        compile_output="",  # Do not read the full logs
                        test_output="",
                        overall_metrics=overall_metrics,
                        duration=data.get("duration", 0.0),
                    )
                    results[vul_id] = result
                except Exception as e:
                    print(f"Warning: Failed to read result for {vul_id}: {e}")

    # Generate reports
    report_gen = ReportGenerator(results, output_dir)
    summary = report_gen.generate_summary()

    report_gen.save_results_json()
    report_gen.save_summary_json(summary)
    report_gen.save_summary_text(summary)
    report_gen.save_errors_json()

    report_gen.print_summary_table(summary)

    print(f"\nReport generated from existing results: {output_dir}")

    return results


def _parse_test_results_fixed(test_output: str) -> bool:
    """Patched test-result parsing logic (standalone helper independent of the class)"""
    try:
        # Try parsing JSON
        data = json.loads(test_output)

        # Check for failing tests in the VUL4J-specific format
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

            # Check common result fields
            for key in ["failed", "failures", "errors", "Failing"]:
                if key in data and data[key] > 0:
                    return False
            return True
    except (json.JSONDecodeError, TypeError):
        pass
    return True


if __name__ == "__main__":
    app()
