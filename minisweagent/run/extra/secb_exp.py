#!/usr/bin/env python3

"""Run mini-SWE-agent on SEC-bench instances (single-turn baseline)."""

import base64
import concurrent.futures
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import traceback
from enum import Enum
from pathlib import Path
from typing import Any

import typer
import yaml
from jsonlines import jsonlines
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.enhanced import EnhancedAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.tools.base import ToolManager
from minisweagent.tools.evolve_tools import EvolveToolsManager, EvolveToolsManagerConfig
from minisweagent.utils.log import add_file_handler, logger

# ExpBank imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPBANK_ROOT = PROJECT_ROOT / "expbank"
import sys

sys.path.insert(0, str(EXPBANK_ROOT))
from src.rag.experience_retriever import ExperienceRetriever

_HELP_TEXT = """Run mini-SWE-agent on SEC-bench instances (single-turn baseline)."""

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEC_BENCH_ROOT = PROJECT_ROOT / "SEC-bench"

DEFAULT_SECB_IMAGE_PREFIX = "hwiwonlee/secb.eval.x86_64"
DEFAULT_SECB_IMAGE_TAG = "patch"
DEFAULT_TOOL_MODEL_CLASS = "litellm_tool"

_OUTPUT_FILE_LOCK = threading.Lock()
_CVE_PATTERN = re.compile(r"cve-\d{4}-\d+", re.IGNORECASE)
_PATCH_APPLY_FAILURE_MARKERS = (
    "patch does not apply",
    "patch failed",
    "corrupt patch",
    "unrecognized input",
    "no valid patches in input",
)

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


class RunMode(str, Enum):
    """Run mode for compatibility with patcheval_exp CLI."""

    BASELINE = "baseline"
    RETRIEVE_ONLY = "retrieve-only"


class ProgressTrackingAgent(EnhancedAgent):
    """Default agent wrapper with status updates for batch progress."""

    def __init__(
        self,
        *args,
        tool_manager: ToolManager,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        **kwargs,
    ):
        super().__init__(*args, tool_manager=tool_manager, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def _extract_cve_id(instance_id: str) -> str:
    match = _CVE_PATTERN.search(instance_id)
    if match:
        return match.group(0).upper()
    return instance_id.upper()


def _extract_summary_language(instance: dict[str, Any] | None) -> str:
    if not isinstance(instance, dict):
        return "Unknown"
    language = str(
        instance.get("programing_language")
        or instance.get("programming_language")
        or "Unknown"
    ).strip()
    return language or "Unknown"


def _append_grouped_summary_value(grouped: dict[str, list[str]], group: str, value: str) -> None:
    if not value:
        return
    grouped.setdefault(group, []).append(value)


def _finalize_grouped_summary_values(grouped: dict[str, list[str]]) -> dict[str, list[str]]:
    finalized: dict[str, list[str]] = {}
    for key, values in grouped.items():
        unique_values = sorted({value for value in values if value})
        if unique_values:
            finalized[key] = unique_values
    return finalized


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def _parse_cve_id_for_temporal_filter(cve_id: str) -> tuple[int, int]:
    """Parse CVE ID into (year, number) for temporal ordering."""
    try:
        parts = str(cve_id).upper().split("-")
        if len(parts) >= 3 and parts[0] == "CVE":
            return int(parts[1]), int(parts[2])
    except (TypeError, ValueError, IndexError):
        pass
    return (9999, 999999)


def _normalize_experience_retrieval_counts(
    retrieve_k: Any,
    return_n: Any,
) -> tuple[int, int]:
    """Normalize ExpBank retrieval parameters for the current retriever API."""
    try:
        normalized_return_n = max(0, int(return_n))
    except (TypeError, ValueError):
        normalized_return_n = 0

    try:
        normalized_retrieve_k = max(0, int(retrieve_k))
    except (TypeError, ValueError):
        normalized_retrieve_k = 0

    candidate_top_k = max(normalized_retrieve_k, normalized_return_n)
    return normalized_return_n, candidate_top_k


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _model_class_supports_tools(model_class: str | None) -> bool:
    if not model_class:
        return False
    normalized = model_class.strip().lower()
    if normalized in {"litellm_tool", "minisweagent.models.litellm_tool_model.litellmtoolmodel"}:
        return True
    return normalized.endswith(".litellmtoolmodel")


def read_secb_dataset(file_path: str) -> list[dict[str, Any]]:
    """Load SEC-bench JSONL dataset and normalize required fields."""
    records: list[dict[str, Any]] = []
    with jsonlines.open(file_path, "r") as reader:
        for row in reader:
            if not isinstance(row, dict):
                continue
            instance_id = str(row.get("instance_id", "")).strip()
            if not instance_id:
                logger.warning("Skipping row without instance_id")
                continue
            normalized = dict(row)
            normalized["instance_id"] = instance_id
            normalized["cve_id"] = str(normalized.get("cve_id", "")).strip() or _extract_cve_id(instance_id)
            normalized["cve_description"] = (
                str(normalized.get("cve_description", "")).strip() or str(normalized.get("bug_description", "")).strip()
            )
            normalized.setdefault("bug_description", normalized["cve_description"])
            normalized.setdefault("cwe_info", {})
            normalized.setdefault("repo", "")
            normalized.setdefault("work_dir", "/")
            normalized.setdefault("programing_language", "Unknown")
            normalized.setdefault("vul_func", [])
            records.append(normalized)
    return records


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str) -> None:
    with _OUTPUT_FILE_LOCK:
        output_data: dict[str, Any] = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text(encoding="utf-8"))
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")


def remove_from_preds_file(output_path: Path, instance_id: str) -> None:
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text(encoding="utf-8"))
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")


def _collect_successful_ids_from_dict(summary_data: dict[str, Any]) -> set[str]:
    successful: set[str] = set()

    def _collect_ids(raw: Any) -> None:
        if isinstance(raw, list):
            successful.update(str(item).strip() for item in raw if str(item).strip())
        elif isinstance(raw, dict):
            for value in raw.values():
                _collect_ids(value)

    for key in ("strict_evaluation", "medium_evaluation", "generous_evaluation"):
        section = summary_data.get(key, {})
        if not isinstance(section, dict):
            continue
        _collect_ids(section.get("successful_instances", []))
        _collect_ids(section.get("successful_cves", {}))

    _collect_ids(summary_data.get("successful_instances", []))
    _collect_ids(summary_data.get("successful_cves", {}))
    return successful


def load_successful_ids(summary_paths: list[Path]) -> set[str]:
    """Load successful IDs from provided summary.json files or directories."""
    successful_ids: set[str] = set()
    if not summary_paths:
        return successful_ids

    expanded_paths: list[Path] = []
    for path in summary_paths:
        if not path.exists():
            logger.warning(f"Summary path not found: {path}")
            continue
        if path.is_dir():
            summaries = list(path.rglob("summary.json"))
            if summaries:
                expanded_paths.extend(summaries)
            else:
                logger.warning(f"No summary.json found in directory: {path}")
        else:
            expanded_paths.append(path)

    for summary_file in expanded_paths:
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Failed to load summary {summary_file}: {exc}")
            continue
        before = len(successful_ids)
        successful_ids.update(_collect_successful_ids_from_dict(data))
        logger.info(f"Loaded {len(successful_ids) - before} successful IDs from {summary_file}")
    return successful_ids


def get_secb_docker_image_name(instance: dict[str, Any], config: dict[str, Any]) -> str:
    image_prefix = str(config.get("image_prefix", DEFAULT_SECB_IMAGE_PREFIX)).strip() or DEFAULT_SECB_IMAGE_PREFIX
    image_tag = str(config.get("image_tag", DEFAULT_SECB_IMAGE_TAG)).strip() or DEFAULT_SECB_IMAGE_TAG
    return f"{image_prefix}.{instance['instance_id']}:{image_tag}"


def get_secb_environment(config: dict[str, Any], instance: dict[str, Any]) -> Environment:
    base_env_config = copy.deepcopy(config.get("environment", {}))
    work_dir = str(instance.get("work_dir", "")).strip() or str(base_env_config.get("cwd", "/"))

    env_config = {
        **base_env_config,
        "environment_class": "docker",
        "timeout": config.get("docker_timeout", 600),
        "container_timeout": config.get("container_timeout", "24h"),
        "cwd": work_dir,
        "image": get_secb_docker_image_name(instance, config),
    }
    env = get_environment(env_config)

    cleanup_output = env.execute(
        "rm -f fix.patch /testcase/model_patch.diff /testcase/repo_changes.diff /testcase/gold_patch.diff"
    )
    if int(cleanup_output.get("returncode", -1)) != 0:
        logger.warning(
            f"Failed to cleanup stale patch files for {instance['instance_id']}: rc={cleanup_output.get('returncode')}"
        )
    return env


def _build_vul_location(instance: dict[str, Any], source_dir: str) -> str:
    vul_func = instance.get("vul_func", [])
    if not isinstance(vul_func, list) or not vul_func:
        return "No precise vulnerable location is provided. You need to locate the vulnerability from the reports."

    locations: list[str] = []
    for entry in vul_func:
        if not isinstance(entry, dict):
            continue
        file_path = entry.get("file_path", "")
        start_line = entry.get("start_line", "")
        end_line = entry.get("end_line", "")
        if file_path:
            locations.append(f"- {source_dir}/{file_path} (lines {start_line}-{end_line})")
    if not locations:
        return "No precise vulnerable location is provided. You need to locate the vulnerability from the reports."
    return "\n".join(locations)


def process_instance(
    instance: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    progress_manager: RunBatchProgressManager,
    retriever: ExperienceRetriever | None = None,
    run_mode: RunMode = RunMode.BASELINE,
) -> None:
    """Process one SEC-bench instance using baseline or retrieve-only patch generation."""
    instance_id = str(instance["instance_id"])
    cve_id = str(instance.get("cve_id", _extract_cve_id(instance_id)))
    instance_dir = output_dir / "raw_traj" / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    tool_config = EvolveToolsManagerConfig(
        BUG_LOCATION_OUTPUT_DIR=output_dir / "fl",
        PATCH_OUTPUT_DIR=output_dir / "patch",
        BUG_REPORT_OUTPUT_DIR=output_dir / "report",
        VUL_ID=instance_id,
        MODEL_CONFIG=config.get("model", {}),
    )
    tool_manager = EvolveToolsManager(tool_config)
    model_config = copy.deepcopy(config.get("model", {}))
    model_class = str(model_config.get("model_class", "")).strip()
    if not _model_class_supports_tools(model_class):
        raise ValueError(
            "secb_exp requires a tool-capable model class. "
            f"Current model_class='{model_class or '<empty>'}'. "
            "Please set --model-class litellm_tool or configure model.model_class=litellm_tool."
        )
    model_config["tools"] = tool_manager.get_all_schemas()
    model = get_model(config=model_config)

    cve_description = str(instance.get("cve_description", "")).strip()
    bug_description = str(instance.get("bug_description", "")).strip()
    cwe_info = instance.get("cwe_info", {})
    repo = str(instance.get("repo", "")).strip()
    work_dir = str(instance.get("work_dir", "")).strip() or "/"
    source_dir = repo.rstrip("/").split("/")[-1] if repo else Path(work_dir).name
    vul_location = _build_vul_location(instance, source_dir)
    live = bool(config.get("live", True))

    # ========== EXPERIENCE RETRIEVAL ==========
    experience_context = ""
    if run_mode == RunMode.RETRIEVE_ONLY and retriever is not None:
        try:
            # Build query text for embedding (concatenate CVE description with CWE details)
            query_text = cve_description
            if isinstance(cwe_info, dict):
                for cwe_id_key, cwe_data in cwe_info.items():
                    if isinstance(cwe_data, dict):
                        cwe_name = cwe_data.get("name", "")
                        cwe_desc = cwe_data.get("description", "")
                        if cwe_name or cwe_desc:
                            query_text += f"\n{cwe_id_key}"
                            if cwe_name:
                                query_text += f" ({cwe_name})"
                            if cwe_desc:
                                query_text += f": {cwe_desc}"

            exclude_current_cve = config.get("exclude_current_cve", True)
            temporal_filter = config.get("temporal_filter", False)
            retrieve_k = config.get("exp_retrieve_k", 3)
            return_n = config.get("exp_return_n", 3)
            sort_by_quality = config.get("exp_sort_by_quality", True)
            final_return_n, candidate_top_k = _normalize_experience_retrieval_counts(retrieve_k, return_n)

            logger.info(f"[{instance_id}] Retrieving experiences...")
            logger.info(f"   Query length: {len(query_text)} chars")
            logger.info(
                f"   Retrieval config: exclude_current_cve={exclude_current_cve}, "
                f"temporal_filter={temporal_filter}, return_n={final_return_n}, "
                f"retrieve_top_k={candidate_top_k}, "
                f"sort_by_quality={sort_by_quality}"
            )

            experiences: list[dict[str, Any]] = []
            if final_return_n > 0:
                experiences = retriever.retrieve(
                    cve_description=query_text,
                    exclude_cve_id=cve_id if exclude_current_cve else None,
                    k=final_return_n,
                    min_quality_score=0,
                    retrieve_top_k=candidate_top_k if candidate_top_k > 0 else None,
                    sort_by_quality=sort_by_quality,
                )
            else:
                logger.info(f"[{instance_id}] Skipping retrieval because exp_return_n <= 0")

            if temporal_filter and experiences:
                current_cve_tuple = _parse_cve_id_for_temporal_filter(cve_id)
                filtered_experiences = [
                    exp
                    for exp in experiences
                    if _parse_cve_id_for_temporal_filter(exp.get("cve_id", "")) < current_cve_tuple
                ]
                logger.info(
                    f"[{instance_id}] Temporal filter applied: {len(experiences)} -> "
                    f"{len(filtered_experiences)} experiences before {cve_id}"
                )
                experiences = filtered_experiences

            if experiences:
                logger.info(f"[{instance_id}] Retrieved {len(experiences)} experiences")

                # Format experiences for prompt injection
                formatted_experiences = []
                for exp in experiences:
                    exp_content = exp.get("experience", "")
                    header = f"### Experience from {exp.get('cve_id', 'Unknown')}\n"
                    header += f"**CWE Types**: {', '.join(exp.get('cwe_ids', []))}\n"
                    header += f"**Quality Score**: {exp.get('quality_score', 0)}/5\n"
                    header += f"**Similarity**: {exp.get('similarity_score', 0):.2f}\n"
                    header += f"**Repair Result**: {exp.get('repair_result', 'Unknown')}\n\n"

                    formatted_text = header + str(exp_content) + "\n\n"
                    formatted_experiences.append(formatted_text)

                experience_context = "".join(formatted_experiences)
                logger.info(f"[{instance_id}] Formatted experience context: {len(experience_context)} chars")
            else:
                logger.info(f"[{instance_id}] No experiences retrieved")

        except Exception as exc:
            logger.error(f"[{instance_id}] Error retrieving experiences: {exc}", exc_info=True)
            experience_context = ""

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent: ProgressTrackingAgent | None = None
    env: Environment | None = None
    extra_info: dict[str, Any] | None = None
    exit_status: str | None = None
    result: str | None = None

    try:
        env = get_secb_environment(config, instance)
        agent = ProgressTrackingAgent(
            model,
            env,
            tool_manager=tool_manager,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )

        exit_status, result = agent.run(
            task="",
            cve_id=cve_id,
            instance_id=instance_id,
            cve_description=cve_description,
            bug_description=bug_description,
            cwe_info=cwe_info,
            repo=repo,
            source_dir=source_dir,
            work_dir=work_dir,
            vul_location=vul_location,
            experience_context=experience_context,
            live=live,
        )
    except Exception as exc:
        logger.error(f"Error processing {instance_id}: {exc}", exc_info=True)
        exit_status, result = "Error", str(exc)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception as cleanup_exc:
                logger.warning(f"Error cleaning up environment for {instance_id}: {cleanup_exc}")

        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result or "")
        progress_manager.on_instance_end(instance_id, exit_status)


def validate_config(
    dataset: str,
    output_path: Path,
    run_mode: RunMode,
    auto_eval: bool,
    summary_paths: list[Path],
    config_path: Path,
    exp_db_path: str | None = None,
    exp_model_path: str | None = None,
) -> None:
    """Validate baseline run configuration."""
    errors: list[str] = []
    warnings: list[str] = []

    dataset_path = Path(dataset)
    if not dataset:
        errors.append("Dataset file not specified")
    elif not dataset_path.exists():
        errors.append(f"Dataset file not found: {dataset_path}")
    elif not dataset_path.is_file():
        errors.append(f"Dataset path is not a file: {dataset_path}")
    else:
        try:
            instances = read_secb_dataset(str(dataset_path))
            logger.info(f"✓ Dataset file: {dataset_path} ({len(instances)} instances)")
        except Exception as exc:
            errors.append(f"Failed to read dataset {dataset_path}: {exc}")

    if not config_path.exists():
        errors.append(f"Config file not found: {config_path}")
    else:
        try:
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
            logger.info(f"✓ Agent Config: {config_path} (valid YAML)")
        except Exception as exc:
            errors.append(f"Invalid config file {config_path}: {exc}")

    if run_mode != RunMode.BASELINE:
        logger.info(f"✓ Run mode: {run_mode.value}")

        # Validate ExpBank paths for retrieve-only mode
        exp_db = exp_db_path if exp_db_path else str(EXPBANK_ROOT / "vec_DB")
        exp_model = exp_model_path if exp_model_path else str(EXPBANK_ROOT / "models" / "bge-large-en-v1.5")

        if not Path(exp_db).exists():
            warnings.append(f"ExpBank database not found: {exp_db} (will use empty DB)")
        else:
            logger.info(f"✓ ExpBank database: {exp_db}")

        if not Path(exp_model).exists():
            errors.append(f"BGE model not found: {exp_model}")
        else:
            logger.info(f"✓ BGE model: {exp_model}")
    else:
        logger.info("✓ Run mode: baseline")

    if summary_paths:
        successful_ids = load_successful_ids(summary_paths)
        logger.info(f"✓ Summary filter loaded, total successful IDs: {len(successful_ids)}")

    logger.info(f"✓ Output directory: {output_path} (will be created)")

    if auto_eval and not SEC_BENCH_ROOT.exists():
        errors.append(f"SEC-bench directory not found: {SEC_BENCH_ROOT}")

    if warnings:
        logger.warning("-" * 60)
        logger.warning("WARNINGS:")
        for warning in warnings:
            logger.warning(f"  ⚠ {warning}")

    if errors:
        logger.error("-" * 60)
        logger.error("ERRORS:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        raise typer.Exit(code=1)


def looks_like_git_patch(model_patch: Any) -> tuple[bool, str]:
    """Validate that a prediction resembles a git diff patch before evaluation."""
    if not isinstance(model_patch, str):
        return False, "model_patch is not a string"

    patch_text = model_patch.strip()
    if not patch_text:
        return False, "model_patch is empty"
    if "diff --git " not in patch_text:
        return False, "model_patch does not contain 'diff --git '"
    if "\n--- " not in f"\n{patch_text}" or "\n+++ " not in f"\n{patch_text}":
        return False, "model_patch is missing unified diff file headers"
    return True, ""


def git_apply_precheck_patch(
    instance: dict[str, Any],
    config: dict[str, Any],
    model_patch: str,
) -> tuple[bool | None, str]:
    """Try git apply --check in a fresh SEC container.

    Returns:
        (True, detail): precheck passed
        (False, detail): patch is structurally patch-like but cannot be applied
        (None, detail): precheck could not run reliably; caller should keep the patch
    """
    env: Environment | None = None
    work_dir = str(instance.get("work_dir", "")).strip() or "/"
    try:
        env = get_secb_environment(config, instance)
    except Exception as exc:
        return None, f"precheck setup failed: {exc}"

    patch_b64 = base64.b64encode(model_patch.encode("utf-8")).decode("ascii")
    command = (
        "cat >/tmp/secb_prefilter.patch.b64 <<'EOF'\n"
        f"{patch_b64}\n"
        "EOF\n"
        "base64 -d /tmp/secb_prefilter.patch.b64 >/tmp/secb_prefilter.patch\n"
        f"cd {shlex.quote(work_dir)} && git apply --check --verbose /tmp/secb_prefilter.patch"
    )

    try:
        result = env.execute(command, timeout=min(int(config.get("docker_timeout", 600)), 120))
    except Exception as exc:
        return None, f"precheck execution failed: {exc}"
    finally:
        try:
            env.cleanup()
        except Exception as cleanup_exc:
            logger.warning(f"Failed to cleanup precheck environment for {instance['instance_id']}: {cleanup_exc}")

    returncode = int(result.get("returncode", 1))
    output = str(result.get("output", "")).strip()
    if returncode == 0:
        return True, output

    lowered = output.lower()
    if any(marker in lowered for marker in _PATCH_APPLY_FAILURE_MARKERS):
        return False, output or "git apply --check failed"

    return None, output or f"git apply --check returned {returncode}"


def write_prefilter_rejections(evaluation_dir: Path, rejections: list[dict[str, Any]]) -> None:
    """Persist prefilter rejection details for traceability."""
    rejections_path = evaluation_dir / "prefilter_rejections.jsonl"
    summary_path = evaluation_dir / "prefilter_summary.json"

    if rejections:
        with jsonlines.open(rejections_path, "w") as writer:
            for row in rejections:
                writer.write(row)
    else:
        rejections_path.unlink(missing_ok=True)

    rejection_breakdown: dict[str, int] = {}
    for row in rejections:
        key = str(row.get("rejection_type", "unknown"))
        rejection_breakdown[key] = rejection_breakdown.get(key, 0) + 1

    summary_payload = {
        "total_rejected": len(rejections),
        "rejection_breakdown": rejection_breakdown,
        "rejected_instances": [row.get("instance_id", "") for row in rejections],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prefilter_eval_predictions(
    preds_path: Path,
    instances: list[dict[str, Any]],
    config: dict[str, Any],
    evaluation_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Drop obviously invalid patch predictions before SEC-bench evaluation."""
    raw_preds = json.loads(preds_path.read_text(encoding="utf-8"))
    instance_map = {str(instance["instance_id"]): instance for instance in instances}
    enable_git_apply_check = bool(config.get("prefilter_git_apply_check", False))

    filtered_preds: dict[str, Any] = {}
    rejections: list[dict[str, Any]] = []

    for instance_id, pred in raw_preds.items():
        model_patch = pred.get("model_patch", "")
        is_patch_like, reason = looks_like_git_patch(model_patch)
        if not is_patch_like:
            rejections.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": pred.get("model_name_or_path", "unknown_model"),
                    "rejection_type": "invalid_patch_format",
                    "reason": reason,
                    "model_patch_preview": str(model_patch)[:500],
                }
            )
            continue

        instance = instance_map.get(instance_id)
        if instance is None:
            logger.warning(f"Missing dataset metadata for {instance_id}; skipping git apply precheck and keeping patch")
            filtered_preds[instance_id] = pred
            continue

        if not enable_git_apply_check:
            filtered_preds[instance_id] = pred
            continue

        apply_ok, detail = git_apply_precheck_patch(instance, config, str(model_patch))
        if apply_ok is False:
            rejections.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": pred.get("model_name_or_path", "unknown_model"),
                    "rejection_type": "git_apply_check_failed",
                    "reason": detail[:2000],
                    "model_patch_preview": str(model_patch)[:500],
                }
            )
            continue
        if apply_ok is None:
            logger.warning(
                "Could not reliably precheck patch for %s; keeping it for evaluator. Detail: %s",
                instance_id,
                detail,
            )

        filtered_preds[instance_id] = pred

    write_prefilter_rejections(evaluation_dir, rejections)
    logger.info(
        "Patch prefilter finished: kept %d / %d, rejected %d",
        len(filtered_preds),
        len(raw_preds),
        len(rejections),
    )
    return raw_preds, filtered_preds, rejections


def build_prediction_summary(
    instances: list[dict[str, Any]],
    raw_preds: dict[str, Any],
    filtered_preds: dict[str, Any],
    prefilter_rejections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize submitted, valid, invalid, and missing patch counts."""
    instance_ids = {str(instance["instance_id"]) for instance in instances}
    submitted_patch_ids = sorted(instance_ids & set(raw_preds))
    valid_patch_ids = sorted(instance_ids & set(filtered_preds))
    invalid_patch_ids = sorted(
        {
            str(row.get("instance_id", "")).strip()
            for row in prefilter_rejections
            if str(row.get("instance_id", "")).strip() in instance_ids
        }
    )
    missing_patch_ids = sorted(instance_ids - set(submitted_patch_ids))
    unexpected_prediction_ids = sorted(set(raw_preds) - instance_ids)

    rejection_breakdown: dict[str, int] = {}
    invalid_patch_reasons: dict[str, str] = {}
    for row in prefilter_rejections:
        rejection_type = str(row.get("rejection_type", "unknown")).strip() or "unknown"
        rejection_breakdown[rejection_type] = rejection_breakdown.get(rejection_type, 0) + 1
        instance_id = str(row.get("instance_id", "")).strip()
        reason = str(row.get("reason", "")).strip()
        if instance_id in instance_ids and reason:
            invalid_patch_reasons[instance_id] = reason

    total_instances = len(instances)
    submitted_patch_count = len(submitted_patch_ids)
    valid_patch_count = len(valid_patch_ids)
    invalid_patch_count = len(invalid_patch_ids)
    missing_patch_count = len(missing_patch_ids)

    return {
        "total_instances": total_instances,
        "total_patch_count": total_instances,
        "submitted_patch_count": submitted_patch_count,
        "valid_patch_count": valid_patch_count,
        "invalid_patch_count": invalid_patch_count,
        "missing_patch_count": missing_patch_count,
        "unexpected_prediction_count": len(unexpected_prediction_ids),
        "submitted_patch_instances": submitted_patch_ids,
        "valid_patch_instances": valid_patch_ids,
        "invalid_patch_instances": invalid_patch_ids,
        "missing_patch_instances": missing_patch_ids,
        "unexpected_prediction_instances": unexpected_prediction_ids,
        "prefilter_rejection_breakdown": dict(sorted(rejection_breakdown.items())),
        "invalid_patch_reasons": dict(sorted(invalid_patch_reasons.items())),
    }


def build_augmented_summary_section(
    title: str,
    instances: list[dict[str, Any]],
    raw_preds: dict[str, Any],
    filtered_preds: dict[str, Any],
    prefilter_rejections: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a summary section using total instances as the denominator."""
    instance_map = {str(instance["instance_id"]): instance for instance in instances}
    instance_ids = set(instance_map)
    prediction_summary = build_prediction_summary(instances, raw_preds, filtered_preds, prefilter_rejections)

    valid_patch_ids = set(prediction_summary["valid_patch_instances"])
    invalid_patch_ids = set(prediction_summary["invalid_patch_instances"])
    missing_patch_ids = set(prediction_summary["missing_patch_instances"])

    success_ids: set[str] = set()
    evaluated_patch_ids: set[str] = set()
    evaluator_failed_ids: set[str] = set()
    for row in report_rows:
        instance_id = str(row.get("instance_id", "")).strip()
        if instance_id not in instance_ids:
            continue
        evaluated_patch_ids.add(instance_id)
        if row.get("success") is True:
            success_ids.add(instance_id)
        else:
            evaluator_failed_ids.add(instance_id)

    unevaluated_valid_patch_ids = sorted(valid_patch_ids - evaluated_patch_ids)
    successful_instances = sorted(success_ids)
    failed_instances = sorted(instance_ids - success_ids)

    success_breakdown: dict[str, int] = {}
    failure_breakdown: dict[str, int] = {}
    successful_cves: dict[str, list[str]] = {}
    failed_cves: dict[str, list[str]] = {}

    for instance_id in sorted(instance_ids):
        instance = instance_map.get(instance_id, {})
        language = _extract_summary_language(instance)
        cve_id = str(instance.get("cve_id", "")).strip().upper() or _extract_cve_id(instance_id)
        if instance_id in success_ids:
            success_breakdown[language] = success_breakdown.get(language, 0) + 1
            _append_grouped_summary_value(successful_cves, language, cve_id)
        else:
            failure_breakdown[language] = failure_breakdown.get(language, 0) + 1
            _append_grouped_summary_value(failed_cves, language, cve_id)

    total_instances = prediction_summary["total_instances"]
    total_success = len(successful_instances)
    total_failed = total_instances - total_success

    return {
        "title": title,
        "total_cases": total_instances,
        "total_instances": total_instances,
        "total_patch_count": prediction_summary["total_patch_count"],
        "submitted_patch_count": prediction_summary["submitted_patch_count"],
        "valid_patch_count": prediction_summary["valid_patch_count"],
        "invalid_patch_count": prediction_summary["invalid_patch_count"],
        "missing_patch_count": prediction_summary["missing_patch_count"],
        "evaluated_patch_count": len(evaluated_patch_ids),
        "unevaluated_valid_patch_count": len(unevaluated_valid_patch_ids),
        "total_success": total_success,
        "total_failed": total_failed,
        "pass_rate": _format_percent(total_success, total_instances),
        "valid_patch_pass_rate": _format_percent(total_success, prediction_summary["valid_patch_count"]),
        "evaluated_patch_pass_rate": _format_percent(total_success, len(evaluated_patch_ids)),
        "success_breakdown": dict(sorted(success_breakdown.items())),
        "failure_breakdown": dict(sorted(failure_breakdown.items())),
        "successful_instances": successful_instances,
        "failed_instances": failed_instances,
        "evaluator_failed_instances": sorted(evaluator_failed_ids),
        "invalid_patch_instances": sorted(invalid_patch_ids),
        "missing_patch_instances": sorted(missing_patch_ids),
        "unevaluated_valid_patch_instances": unevaluated_valid_patch_ids,
        "successful_cves": _finalize_grouped_summary_values(successful_cves),
        "failed_cves": _finalize_grouped_summary_values(failed_cves),
        "prefilter_rejection_breakdown": prediction_summary["prefilter_rejection_breakdown"],
        "invalid_patch_reasons": prediction_summary["invalid_patch_reasons"],
    }


def _extract_execution_errors_from_rows(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    execution_errors: list[dict[str, Any]] = []
    for row in report_rows:
        try:
            exit_code = int(row.get("exit_code", 0))
        except (TypeError, ValueError):
            exit_code = 0
        if exit_code != -1:
            continue
        execution_errors.append(
            {
                "instance_id": str(row.get("instance_id", "")).strip(),
                "reason": str(row.get("reason", "")).strip(),
                "exit_code": exit_code,
                "model_name": str(row.get("model_name", "unknown_model")).strip() or "unknown_model",
            }
        )
    return execution_errors


def write_augmented_evaluation_summary(
    evaluation_dir: Path,
    instances: list[dict[str, Any]],
    raw_preds: dict[str, Any],
    filtered_preds: dict[str, Any],
    prefilter_rejections: list[dict[str, Any]],
    eval_mode: str,
) -> None:
    """Overwrite evaluator summary.json with total-instance-based statistics."""
    summary_path = evaluation_dir / "summary.json"
    existing_summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded_summary, dict):
                existing_summary = loaded_summary
        except Exception as exc:
            logger.warning(f"Failed to read existing evaluation summary {summary_path}: {exc}")

    if eval_mode == "all":
        section_names = ["strict", "medium", "generous"]
    else:
        section_names = [eval_mode]

    summary_data: dict[str, Any] = {
        "prediction_summary": build_prediction_summary(instances, raw_preds, filtered_preds, prefilter_rejections),
        "execution_errors": [],
    }

    execution_errors: list[dict[str, Any]] = []
    for section_name in section_names:
        report_path = evaluation_dir / f"report_{section_name}.jsonl"
        report_rows = _read_jsonl_rows(report_path)
        summary_key = f"{section_name}_evaluation"
        summary_title = f"{section_name.capitalize()} Evaluation Summary"
        summary_data[summary_key] = build_augmented_summary_section(
            title=summary_title,
            instances=instances,
            raw_preds=raw_preds,
            filtered_preds=filtered_preds,
            prefilter_rejections=prefilter_rejections,
            report_rows=report_rows,
        )
        execution_errors.extend(_extract_execution_errors_from_rows(report_rows))

    if not execution_errors:
        raw_errors = existing_summary.get("execution_errors", [])
        if isinstance(raw_errors, list):
            execution_errors = [row for row in raw_errors if isinstance(row, dict)]
    summary_data["execution_errors"] = execution_errors
    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_post_evaluation(
    output_path: Path,
    instances: list[dict[str, Any]],
    config: dict[str, Any],
    eval_workers: int = 16,
    eval_split: str = "eval",
    eval_mode: str = "medium",
    eval_dataset: str = "SEC-bench/SEC-bench",
) -> None:
    """Run SEC-bench evaluator on generated preds.json."""
    if not SEC_BENCH_ROOT.exists():
        logger.error(f"SEC-bench root not found: {SEC_BENCH_ROOT}")
        return

    preds_path = output_path / "preds.json"
    if not preds_path.exists():
        logger.warning(f"preds.json not found in {output_path}, skipping evaluation")
        return

    eval_output_dir = output_path / "evaluation"
    if eval_output_dir.exists():
        shutil.rmtree(eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Use a dedicated temporary input directory to avoid recursive copy behavior
    # in SEC-bench evaluator when output-dir is inside input-dir.
    eval_input_dir = output_path / "_eval_input"
    if eval_input_dir.exists():
        shutil.rmtree(eval_input_dir)
    eval_input_dir.mkdir(parents=True, exist_ok=True)

    raw_preds, filtered_preds, prefilter_rejections = prefilter_eval_predictions(
        preds_path=preds_path,
        instances=instances,
        config=config,
        evaluation_dir=eval_output_dir,
    )
    (eval_input_dir / "preds.json").write_text(
        json.dumps(filtered_preds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if prefilter_rejections:
        logger.warning(
            "Prefilter rejected %d predictions before SEC-bench evaluation. Details: %s",
            len(prefilter_rejections),
            eval_output_dir / "prefilter_rejections.jsonl",
        )

    if not filtered_preds:
        logger.warning("No valid patch predictions remained after prefilter; skipping SEC-bench evaluation")
        write_augmented_evaluation_summary(
            evaluation_dir=eval_output_dir,
            instances=instances,
            raw_preds=raw_preds,
            filtered_preds=filtered_preds,
            prefilter_rejections=prefilter_rejections,
            eval_mode=eval_mode,
        )
        _run_localization_coverage(output_path, eval_output_dir, config)
        return

    cmd = [
        sys.executable,
        "-m",
        "secb.evaluator.eval_instances",
        "--input-dir",
        str(eval_input_dir),
        "--type",
        "patch",
        "--split",
        eval_split,
        "--dataset",
        eval_dataset,
        "--mode",
        eval_mode,
        "--agent",
        "swea",
        "--num-workers",
        str(eval_workers),
        "--output-dir",
        str(eval_output_dir),
    ]

    logger.info(f"Running SEC-bench evaluation: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("HF_DATASETS_CACHE", "/tmp/secb_hf_datasets_cache")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SEC_BENCH_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )
    except Exception as exc:
        logger.error(f"Failed to run SEC-bench evaluation: {exc}")
        return

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

    write_augmented_evaluation_summary(
        evaluation_dir=eval_output_dir,
        instances=instances,
        raw_preds=raw_preds,
        filtered_preds=filtered_preds,
        prefilter_rejections=prefilter_rejections,
        eval_mode=eval_mode,
    )
    _run_localization_coverage(output_path, eval_output_dir, config)

    if result.returncode != 0:
        logger.error(f"SEC-bench evaluation failed with code {result.returncode}")
        return

    logger.info(f"SEC-bench evaluation completed. Results at: {eval_output_dir}")

    summary_path = eval_output_dir / "summary.json"
    if not summary_path.exists():
        logger.warning(f"SEC-bench evaluation completed but summary.json was not generated: {summary_path}")
        return

    try:
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to read evaluation summary {summary_path}: {exc}")
        return

    if eval_mode == "all":
        section_names = ["strict", "medium", "generous"]
    else:
        section_names = [eval_mode]

    for section_name in section_names:
        section = summary_data.get(f"{section_name}_evaluation", {})
        if not isinstance(section, dict) or not section:
            continue
        logger.info(
            f"SEC-bench {section_name} summary: "
            f"total={section.get('total_cases', 0)}, "
            f"success={section.get('total_success', 0)}, "
            f"failed={section.get('total_failed', 0)}, "
            f"pass_rate={section.get('pass_rate', '0.00%')}, "
            f"valid={section.get('valid_patch_count', 0)}, "
            f"invalid={section.get('invalid_patch_count', 0)}"
        )


def _run_localization_coverage(output_path: Path, eval_output_dir: Path, config: dict[str, Any]) -> None:
    gold_dir = config.get("localization_gold_dir")
    if gold_dir:
        gold_path = Path(str(gold_dir)).expanduser().resolve()
    else:
        gold_path = PROJECT_ROOT / "minisweagent" / "data" / "secb_gold_patches"
    if not gold_path.exists():
        logger.warning(f"Localization gold patch dir not found: {gold_path}")
        return

    preds_path = output_path / "preds.json"
    if not preds_path.exists():
        logger.warning(f"Localization skipped: preds.json not found at {preds_path}")
        return

    try:
        from minisweagent.run.extra.secb_localization_eval import compute_localization_coverage

        summary = compute_localization_coverage(
            preds_path=preds_path,
            eval_dir=eval_output_dir,
            gold_dir=gold_path,
            output_dir=eval_output_dir,
        )
        logger.info(
            "Localization coverage: file=%.2f%% (%d/%d), method=%.2f%% (%d/%d)",
            100.0 * summary.get("file_coverage", 0.0),
            summary.get("file_total_covered", 0),
            summary.get("file_total_required", 0),
            100.0 * summary.get("method_coverage", 0.0),
            summary.get("method_total_covered", 0),
            summary.get("method_total_required", 0),
        )
    except Exception as exc:
        logger.warning(f"Localization coverage computation failed: {exc}")


@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option(
        "minisweagent/data/secb_official_cve201_dataset.jsonl",
        "-d",
        "--dataset",
        help="Dataset JSONL file",
        rich_help_panel="Basic",
    ),
    output: str = typer.Option(
        "minisweagent/output/sec/test",
        "-o",
        "--output",
        help="Base output directory",
        rich_help_panel="Basic",
    ),
    workers: int = typer.Option(
        1,
        "-w",
        "--workers",
        help="Number of worker threads for parallel processing",
        rich_help_panel="Basic",
    ),
    model: str | None = typer.Option(
        "dashscope/qwen3.5-plus",
        "-m",
        "--model",
        help="Model name",
        rich_help_panel="Basic",
    ),
    model_class: str | None = typer.Option(
        None,
        "--model-class",
        help="Model class to use (e.g., litellm_tool)",
        rich_help_panel="Advanced",
    ),
    redo_existing: bool = typer.Option(
        True,
        "--redo-existing",
        help="Redo existing instances",
        rich_help_panel="Data selection",
    ),
    config_spec: Path = typer.Option(
        builtin_config_dir / "extra" / "secb_exp.yaml",
        "-c",
        "--config",
        help="Path to config file",
        rich_help_panel="Basic",
    ),
    summary_path: list[Path] = typer.Option(
        [],
        "--summary-path",
        help="Path(s) to summary.json from previous turns",
        rich_help_panel="Data selection",
    ),
    run_mode: RunMode = typer.Option(
        RunMode.BASELINE,
        "--mode",
        help="Run mode: baseline (no exp) or retrieve-only (inject experiences)",
        rich_help_panel="Experience Enhancement",
    ),
    exp_db_path: str | None = typer.Option(
        None,
        "--exp-db",
        help="ExpBank vector database path",
        rich_help_panel="Experience Enhancement",
    ),
    exp_model_path: str | None = typer.Option(
        None,
        "--exp-model",
        help="BGE model path for experience retrieval",
        rich_help_panel="Experience Enhancement",
    ),
    exclude_current_cve: bool = typer.Option(
        True,
        "--exclude-current-cve/--no-exclude-current-cve",
        help="Whether to exclude current CVE's experiences from retrieval",
        rich_help_panel="Experience Enhancement",
    ),
    max_instances: int | None = typer.Option(
        None,
        "--max-instances",
        help="Maximum number of instances to run",
        rich_help_panel="Data selection",
    ),
    auto_eval: bool = typer.Option(
        True,
        "--auto-eval/--no-auto-eval",
        help="Run SEC-bench evaluator after repair",
        rich_help_panel="Post-processing",
    ),
    eval_workers: int = typer.Option(
        16,
        "--eval-workers",
        help="Number of workers for evaluation",
        rich_help_panel="Post-processing",
    ),
    eval_split: str = typer.Option(
        "eval",
        "--eval-split",
        help="SEC-bench split for evaluation",
        rich_help_panel="Post-processing",
    ),
    eval_mode: str = typer.Option(
        "medium",
        "--eval-mode",
        help="Evaluation mode: strict/medium/generous/all",
        rich_help_panel="Post-processing",
    ),
    eval_dataset: str = typer.Option(
        "SEC-bench/SEC-bench",
        "--eval-dataset",
        help="SEC-bench HF dataset name",
        rich_help_panel="Post-processing",
    ),
    docker_timeout: int = typer.Option(
        600,
        "--docker-timeout",
        help="Docker command timeout in seconds",
        rich_help_panel="Basic",
    ),
    live: bool = typer.Option(
        True,
        "--live/--no-live",
        help="Include live tool-creation guidance in the prompt",
        rich_help_panel="Advanced",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration and exit",
        rich_help_panel="Advanced",
    ),
    temporal_filter: bool = typer.Option(
        False,
        "--temporal-filter/--no-temporal-filter",
        help="Only use experiences from earlier CVEs (by CVE ID year/number)",
        rich_help_panel="Experience Enhancement",
    ),
    exp_retrieve_k: int = typer.Option(
        3,
        "--exp-retrieve-k",
        help="Number of top similar experiences to retrieve first",
        rich_help_panel="Experience Enhancement",
    ),
    exp_return_n: int = typer.Option(
        3,
        "--exp-return-n",
        help="Number of experiences to return after quality sorting",
        rich_help_panel="Experience Enhancement",
    ),
    exp_sort_by_quality: bool = typer.Option(
        True,
        "--exp-sort-by-quality/--no-exp-sort-by-quality",
        help="Sort retrieved experiences by quality score",
        rich_help_panel="Experience Enhancement",
    ),
    turn_output: int | None = typer.Option(
        None,
        "--turn-output",
        help="Wrap output in turn_N subdirectory (e.g., --turn-output 1 creates output/turn_1/)",
        rich_help_panel="Advanced",
    ),
) -> None:
    if not output:
        raise typer.BadParameter("`--output` is required", param_hint="--output")

    # Handle turn_output parameter for multi-turn runner compatibility
    # If --turn-output N is specified, wrap output in turn_N subdirectory
    output_path = Path(output).expanduser().resolve()
    if turn_output is not None:
        output_path = output_path / f"turn_{turn_output}"
        logger.info(f"Multi-turn mode: wrapping output in {output_path}")

    logger.info(f"Target output directory: {output_path}")

    config_path = get_config_path(config_spec)
    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE: Validating Configuration")
        logger.info("=" * 60)

        # Resolve ExpBank paths for validation
        resolved_exp_db = exp_db_path if exp_db_path else str(EXPBANK_ROOT / "vec_DB")
        resolved_exp_model = exp_model_path if exp_model_path else str(EXPBANK_ROOT / "models" / "bge-large-en-v1.5")

        validate_config(
            dataset=dataset,
            output_path=output_path,
            run_mode=run_mode,
            auto_eval=auto_eval,
            summary_paths=summary_path,
            config_path=config_path,
            exp_db_path=resolved_exp_db,
            exp_model_path=resolved_exp_model,
        )
        logger.info("=" * 60)
        logger.info("Configuration validation passed")
        logger.info("=" * 60)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")

    log_path = output_path / "minisweagent_exp.log"
    if log_path.exists():
        log_path.unlink()
    add_file_handler(log_path)

    # Resolve ExpBank paths
    if exp_db_path is None:
        exp_db_path = str(EXPBANK_ROOT / "vec_DB")
    else:
        exp_db_path = str(Path(exp_db_path).expanduser().resolve())

    if exp_model_path is None:
        exp_model_path = str(EXPBANK_ROOT / "models" / "bge-large-en-v1.5")
    else:
        exp_model_path = str(Path(exp_model_path).expanduser().resolve())

    # Initialize ExperienceRetriever for retrieve-only mode
    retriever = None
    if run_mode == RunMode.RETRIEVE_ONLY:
        logger.info("Initializing ExpBank retriever for retrieve-only mode...")
        logger.info(f"  ExpBank database: {exp_db_path}")
        logger.info(f"  BGE model: {exp_model_path}")
        retriever = ExperienceRetriever(
            db_path=exp_db_path,
            model_path=exp_model_path,
        )
        logger.info("ExpBank retriever initialized successfully")
    else:
        logger.info("Run Mode: BASELINE")
        logger.info(f"Ignoring experience-related options in baseline mode: exp_db_path={exp_db_path}")

    logger.info(f"Loading dataset {dataset}")
    instances = read_secb_dataset(dataset)
    logger.info(f"Dataset loaded: {len(instances)} instances")

    if not redo_existing and (output_path / "preds.json").exists():
        existing_ids = set(json.loads((output_path / "preds.json").read_text(encoding="utf-8")).keys())
        logger.info(f"Skipping {len(existing_ids)} existing instances from preds.json")
        instances = [item for item in instances if item["instance_id"] not in existing_ids]

    if summary_path:
        successful_ids = load_successful_ids(summary_path)
        if successful_ids:
            before = len(instances)
            instances = [
                item
                for item in instances
                if item["instance_id"] not in successful_ids and item["cve_id"] not in successful_ids
            ]
            logger.info(f"Filtered {before - len(instances)} instances via summary successful IDs")

    if max_instances is not None and max_instances > 0:
        logger.info(f"Limiting to {max_instances} instances")
        instances = instances[:max_instances]

    logger.info(f"Running on {len(instances)} instances")

    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    current_model_class = str(config.setdefault("model", {}).get("model_class", "")).strip()
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    elif not current_model_class:
        config.setdefault("model", {})["model_class"] = DEFAULT_TOOL_MODEL_CLASS
        logger.info(f"Defaulting model_class to tool-capable '{DEFAULT_TOOL_MODEL_CLASS}'")

    config["docker_timeout"] = docker_timeout
    config["live"] = live
    logger.info(f"Docker timeout set to: {docker_timeout}s")

    # Store experience retrieval parameters in config
    config["exclude_current_cve"] = exclude_current_cve
    config["temporal_filter"] = temporal_filter
    config["exp_retrieve_k"] = exp_retrieve_k
    config["exp_return_n"] = exp_return_n
    config["exp_sort_by_quality"] = exp_sort_by_quality

    run_config = {
        "basic": {
            "dataset": str(Path(dataset).resolve()) if dataset else "",
            "output": str(output_path),
            "workers": workers,
            "model": model or config.get("model", {}).get("model_name", ""),
            "model_class": model_class,
            "config_file": str(config_path),
            "docker_timeout": docker_timeout,
            "live": live,
            "live_mode": "live" if live else "no-live",
        },
        "data_selection": {
            "redo_existing": redo_existing,
            "max_instances": max_instances,
            "summary_paths": [str(path) for path in summary_path] if summary_path else [],
        },
        "experience_enhancement": {
            "run_mode": run_mode.value,
            "exp_db_path": exp_db_path,
            "exp_model_path": exp_model_path,
            "exclude_current_cve": exclude_current_cve,
            "temporal_filter": temporal_filter,
            "exp_retrieve_k": exp_retrieve_k,
            "exp_return_n": exp_return_n,
            "exp_sort_by_quality": exp_sort_by_quality,
        },
        "post_processing": {
            "auto_eval": auto_eval,
            "eval_workers": eval_workers,
            "eval_split": eval_split,
            "eval_mode": eval_mode,
            "eval_dataset": eval_dataset,
        },
    }
    run_config_path = output_path / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Run config saved to: {run_config_path}")

    progress_manager = RunBatchProgressManager(len(instances), output_path / "exit_statuses.yaml")

    with Live(progress_manager.render_group, refresh_per_second=4):
        if workers <= 1:
            for instance in instances:
                try:
                    process_instance(instance, output_path, config, progress_manager, retriever, run_mode)
                except Exception as exc:
                    logger.error(f"Fatal error processing {instance.get('instance_id')}: {exc}", exc_info=True)
                    continue
        else:
            logger.info(f"Starting parallel processing with {workers} workers")

            def worker_fn(item: dict[str, Any]) -> tuple[str, str | None]:
                try:
                    process_instance(item, output_path, config, progress_manager, retriever, run_mode)
                    return item.get("instance_id", ""), None
                except Exception as exc:
                    logger.error(f"Fatal error processing {item.get('instance_id')}: {exc}", exc_info=True)
                    return item.get("instance_id", ""), str(exc)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(worker_fn, item): item for item in instances}
                for future in concurrent.futures.as_completed(futures):
                    instance_id, error = future.result()
                    if error:
                        logger.warning(f"Instance {instance_id} completed with error: {error}")

    if auto_eval:
        run_post_evaluation(
            output_path=output_path,
            instances=instances,
            config=config,
            eval_workers=eval_workers,
            eval_split=eval_split,
            eval_mode=eval_mode,
            eval_dataset=eval_dataset,
        )


if __name__ == "__main__":
    app()
