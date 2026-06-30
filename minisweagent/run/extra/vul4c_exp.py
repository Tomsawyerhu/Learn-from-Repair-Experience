#!/usr/bin/env python3
"""Run mini-SWE-agent on Vul4C instances (single-round, baseline/retrieve-only)."""

from __future__ import annotations

import base64
import concurrent.futures
import copy
import difflib
import json
import os
import re
import shlex
import shutil
import sys
import threading
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.live import Live

from minisweagent import logger
from minisweagent.agents.default import DefaultAgent, Submitted
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.extra.vul4c_evaluate import run_evaluation
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler

# ========== Paths ==========
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPBANK_ROOT = PROJECT_ROOT / "expbank"

# ExpBank import
sys.path.append(str(EXPBANK_ROOT))
try:
    from src.rag.experience_retriever import ExperienceRetriever

    EXPBANK_AVAILABLE = True
except ImportError:
    logger.warning("ExpBank not found, retrieve-only mode will run without experience injection.")
    ExperienceRetriever = None
    EXPBANK_AVAILABLE = False


class RunMode(str, Enum):
    """Run mode for experience enhancement."""

    BASELINE = "baseline"
    RETRIEVE_ONLY = "retrieve-only"


app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()
_SAFE_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DOCKER_MEMORY_LIMIT = "6g"
_MEMORY_LIMIT_PATTERNS = (
    "out of memory",
    "cannot allocate memory",
    "memory cgroup out of memory",
    "oom-kill",
    "oom killed",
    "oomkilled",
)


class MemoryLimitExceededError(RuntimeError):
    """Raised when command execution exceeds Docker memory limit."""


class ProgressTrackingAgent(DefaultAgent):
    """Default agent with progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()

    def has_finished(self, output: dict[str, str]) -> None:
        """Allow final marker to appear after shell startup noise (e.g., mesg tty warnings)."""
        lines = output.get("output", "").splitlines(keepends=True)
        final_markers = {"MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
        for idx, line in enumerate(lines):
            if line.strip() in final_markers:
                raise Submitted("".join(lines[idx + 1 :]))
        super().has_finished(output)

    def execute_action(self, action: dict, *, timeout: int | None = None) -> dict:
        output = super().execute_action(action, timeout=timeout)
        if _is_memory_limit_exceeded_output(output):
            msg = "Detected Docker memory limit exceeded during agent command execution."
            raise MemoryLimitExceededError(msg)
        return output


def parse_cve_id(cve_id: str) -> tuple[int, int]:
    """Parse CVE ID for temporal filtering."""
    try:
        parts = cve_id.upper().split("-")
        if len(parts) >= 3 and parts[0] == "CVE":
            return int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        pass
    return (9999, 999999)


def _sanitize_model_name(model_name: str | None) -> str:
    if not model_name:
        return "model"
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name)
    return cleaned or "model"


def _default_output_path(model_name: str | None, retrieve_k: int, return_n: int) -> Path:
    base = PROJECT_ROOT / "minisweagent" / "output" / "vul4c"
    return base / f"{_sanitize_model_name(model_name)}-k-{retrieve_k}-n-{return_n}"


def _default_tmp_mount_root(output_path: Path) -> Path:
    """Return a run-scoped mount root under the current output directory."""
    return output_path


def _default_vul4c_data_dir(output_path: Path) -> Path:
    """Return a run-scoped writable host directory for container /vul4c_data."""
    return output_path / "_vul4c_data_tmp"


def _run_args_has_mount_target(run_args: list[str], target: str) -> bool:
    for idx, arg in enumerate(run_args):
        if arg in {"-v", "--volume"} and idx + 1 < len(run_args):
            spec = str(run_args[idx + 1])
            parts = spec.split(":")
            if len(parts) >= 2 and parts[1] == target:
                return True
            continue
        if isinstance(arg, str) and arg.startswith("--mount="):
            spec = arg[len("--mount=") :]
            if f"target={target}" in spec or f"dst={target}" in spec or f"destination={target}" in spec:
                return True
    return False


def _enforce_docker_memory_limit(run_args: list[str], memory_limit: str) -> list[str]:
    """Force docker run args to use a fixed memory limit."""
    sanitized: list[str] = []
    skip_next = False
    for idx, arg in enumerate(run_args):
        if skip_next:
            skip_next = False
            continue
        if arg in {"--memory", "-m"}:
            if idx + 1 < len(run_args):
                skip_next = True
            continue
        if isinstance(arg, str) and (arg.startswith("--memory=") or arg.startswith("-m=")):
            continue
        sanitized.append(arg)
    sanitized.extend(["--memory", memory_limit])
    return sanitized


def _looks_like_memory_limit_signal(returncode: int, output: str) -> bool:
    lowered = (output or "").lower()
    if any(pattern in lowered for pattern in _MEMORY_LIMIT_PATTERNS):
        return True
    if returncode == 137 and ("killed" in lowered or "memory" in lowered or "oom" in lowered):
        return True
    return False


def _is_memory_limit_exceeded_output(output: dict[str, Any]) -> bool:
    rc = int(output.get("returncode", -1))
    text = str(output.get("output", ""))
    return _looks_like_memory_limit_signal(rc, text)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _update_json_entry(path: Path, instance_id: str, entry: dict[str, Any]) -> None:
    with _OUTPUT_FILE_LOCK:
        data = _load_json(path)
        data[instance_id] = entry
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _remove_json_entry(path: Path, instance_id: str) -> None:
    if not path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        data = _load_json(path)
        if instance_id in data:
            del data[instance_id]
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_remove_tree(path: Path) -> None:
    """Best-effort recursive cleanup for run-scoped working directories."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        logger.info(f"Cleaned working directory: {path}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to clean working directory {path}: {exc}")


def _validate_instance_id(instance_id: str) -> str:
    if not _SAFE_INSTANCE_ID_RE.fullmatch(instance_id):
        msg = f"Unsafe instance id for workspace path usage: {instance_id!r}"
        raise ValueError(msg)
    return instance_id


def _cleanup_instance_resources(
    env: Any | None,
    *,
    container_instance_dir: str,
    host_instance_dir: Path,
    cleanup_host_workspace: bool,
) -> None:
    """Best-effort instance-level cleanup to avoid resource accumulation in parallel runs."""
    if env is not None:
        cleanup_cmd = f"rm -rf -- {shlex.quote(container_instance_dir)}"
        try:
            output = env.execute(cleanup_cmd, cwd="/", timeout=60)
            if int(output.get("returncode", -1)) != 0:
                logger.warning(
                    f"Container workspace cleanup returned non-zero for {container_instance_dir}: "
                    f"rc={output.get('returncode')}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Container workspace cleanup failed for {container_instance_dir}: {exc}")
        try:
            env.cleanup()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Environment cleanup failed for {container_instance_dir}: {exc}")

    if cleanup_host_workspace:
        _safe_remove_tree(host_instance_dir)


def _has_completed_traj(output_dir: Path, instance_id: str) -> bool:
    """Whether this CVE already has a saved trajectory JSON in output_dir."""
    if not instance_id:
        return False
    traj_path = output_dir / "raw_traj" / instance_id / f"{instance_id}.traj.json"
    return traj_path.is_file()


def _load_vul4c_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Vul4C dataset must be a JSON object keyed by CVE ID.")

    instances: list[dict[str, Any]] = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item["cve_id"] = item.get("cve_id") or key
        instances.append(item)

    instances.sort(key=lambda x: x.get("cve_id", ""))
    return instances


def _build_cwe_text(instance: dict[str, Any]) -> tuple[str, str]:
    cwe_ids = instance.get("cwe", [])
    if isinstance(cwe_ids, str):
        cwe_id_text = cwe_ids
    elif isinstance(cwe_ids, list):
        cwe_id_text = ", ".join(str(x) for x in cwe_ids if x)
    else:
        cwe_id_text = ""

    details = instance.get("cwe_details", [])
    desc_parts: list[str] = []
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            item_desc = str(item.get("description", "")).strip()
            if item_id and item_desc:
                desc_parts.append(f"{item_id}: {item_desc}")
            elif item_id:
                desc_parts.append(item_id)
            elif item_desc:
                desc_parts.append(item_desc)
    cwe_desc_text = "\n".join(desc_parts)
    return cwe_id_text, cwe_desc_text


def _prepare_sok_runtime_snapshot(sok_source: Path, runtime_root: Path, sanitize_benchmark: bool) -> Path:
    """Create a runtime SoK snapshot to prevent truth-patch leakage."""
    marker = runtime_root / ".snapshot_ready"
    mode_marker = "sanitized" if sanitize_benchmark else "full"

    if marker.exists():
        current = marker.read_text(encoding="utf-8").strip()
        if current == mode_marker:
            return runtime_root

    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    src_vul4c = sok_source / "Vul4C_Src"
    src_benchmark = sok_source / "Vul4C-Benchmark"
    if not src_vul4c.exists():
        raise FileNotFoundError(f"Missing Vul4C_Src under {sok_source}")
    if not src_benchmark.exists():
        raise FileNotFoundError(f"Missing Vul4C-Benchmark under {sok_source}")

    shutil.copytree(src_vul4c, runtime_root / "Vul4C_Src")

    dst_benchmark = runtime_root / "Vul4C-Benchmark"
    if not sanitize_benchmark:
        shutil.copytree(src_benchmark, dst_benchmark)
    else:
        # Keep only exploit files required by vul4c checkout; exclude *_OLD/_NEW/.diff truth artifacts.
        for project_dir in sorted(src_benchmark.iterdir()):
            if not project_dir.is_dir():
                continue
            for cve_dir in sorted(project_dir.iterdir()):
                if not cve_dir.is_dir():
                    continue
                copied = False
                for file_path in sorted(cve_dir.iterdir()):
                    if not file_path.is_file():
                        continue
                    if not file_path.name.startswith("exploit"):
                        continue
                    dst_dir = dst_benchmark / project_dir.name / cve_dir.name
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dst_dir / file_path.name)
                    copied = True
                if not copied:
                    logger.warning(f"No exploit file found in {cve_dir}")

    marker.write_text(mode_marker, encoding="utf-8")
    return runtime_root


def _retrieve_experience_context(
    instance: dict[str, Any],
    retriever: Any,
    run_mode: RunMode,
    config: dict[str, Any],
    temporal_filter: bool,
) -> str:
    if run_mode == RunMode.BASELINE or retriever is None:
        return ""

    cve_id = str(instance.get("cve_id", ""))
    cve_desc = str(instance.get("cve_description", "") or instance.get("nvd_info", ""))
    cwe_id, cwe_desc = _build_cwe_text(instance)
    query_text = cve_desc
    if cwe_id:
        query_text += f"\n{cwe_id}"
    if cwe_desc:
        query_text += f"\n{cwe_desc}"

    exclude_current_cve = config.get("exclude_current_cve", True)
    retrieve_k = int(config.get("exp_retrieve_k", 3))
    return_n = int(config.get("exp_return_n", 3))
    sort_by_quality = bool(config.get("exp_sort_by_quality", True))

    experiences = retriever.retrieve(
        cve_description=query_text,
        exclude_cve_id=cve_id if exclude_current_cve else None,
        k=return_n,
        min_quality_score=0,
        retrieve_top_k=retrieve_k if not exclude_current_cve else max(retrieve_k, return_n * 3),
        sort_by_quality=sort_by_quality,
    )

    if temporal_filter:
        current_tuple = parse_cve_id(cve_id)
        experiences = [exp for exp in experiences if parse_cve_id(str(exp.get("cve_id", ""))) < current_tuple]

    if not experiences:
        return ""

    formatted: list[str] = []
    for exp in experiences:
        cwe_ids = exp.get("cwe_ids", [])
        if not isinstance(cwe_ids, list):
            cwe_ids = [str(cwe_ids)]
        header = (
            f"### Experience from {exp.get('cve_id', 'Unknown')}\n"
            f"**CWE Types**: {', '.join(str(x) for x in cwe_ids)}\n"
            f"**Quality Score**: {exp.get('quality_score', 0)}/5\n"
            f"**Similarity**: {exp.get('similarity_score', 0):.2f}\n"
            f"**Repair Result**: {exp.get('repair_result', 'Unknown')}\n\n"
        )
        formatted.append(header + str(exp.get("experience", "")) + "\n\n")
    return "".join(formatted)


def _run_or_raise(
    env: Any,
    command: str,
    *,
    cwd: str,
    timeout: int,
    step_name: str,
) -> dict[str, Any]:
    output = env.execute(command, cwd=cwd, timeout=timeout)
    if _is_memory_limit_exceeded_output(output):
        msg = f"{step_name} failed due to Docker memory limit exceeded."
        raise MemoryLimitExceededError(msg)
    if int(output.get("returncode", -1)) != 0:
        msg = f"{step_name} failed (rc={output.get('returncode')})\n{output.get('output', '')}"
        raise RuntimeError(msg)
    return output


def _ensure_vul4c_data_config(env: Any, *, timeout: int = 60) -> None:
    """Ensure /vul4c_data exists and contains a valid vul4c.ini for container paths."""
    ini_content = "[VUL4C]\nvul4c_path = /home/SoK-main\nlog_to_file = 1\nfile_log_level = INFO\n"
    py_code = (
        "import pathlib;"
        "root=pathlib.Path('/vul4c_data');"
        "(root/'logs').mkdir(parents=True,exist_ok=True);"
        "ini=root/'vul4c.ini';"
        f"ini.write_text({ini_content!r}, encoding='utf-8');"
    )
    _run_or_raise(
        env,
        f"python3 -c {shlex.quote(py_code)}",
        cwd="/",
        timeout=timeout,
        step_name="prepare_vul4c_data",
    )


def _rewrite_test_command_exploit_path(
    env: Any,
    *,
    cwd: str,
    timeout: int,
) -> tuple[bool, str]:
    """
    Best-effort fix: replace ../VUL4C/exploit in test_command when checkout
    only provides exploit.* (e.g., exploit.tif / exploit.bmp).
    """
    py_code = (
        "import json,pathlib,sys;"
        "p=pathlib.Path('VUL4C/vulnerability_info.json');"
        "placeholder='../VUL4C/exploit';"
        "p.exists() or (print('vulnerability_info.json missing') or sys.exit(0));"
        "info=json.loads(p.read_text(encoding='utf-8'));"
        "cmds=info.get('test_command');"
        "isinstance(cmds,list) or (print('test_command is not list') or sys.exit(0));"
        "vdir=p.parent;"
        "(vdir/'exploit').exists() and (print('plain exploit exists, no rewrite needed') or sys.exit(0));"
        "cands=sorted(x.name for x in vdir.iterdir() if x.is_file() and x.name.startswith('exploit') and x.name!='exploit');"
        "cands or (print('no exploit.* candidate found') or sys.exit(0));"
        "replacement=f'../VUL4C/{cands[0]}';"
        "changed=any(isinstance(cmd,str) and placeholder in cmd for cmd in cmds);"
        "new_cmds=[cmd.replace(placeholder,replacement) if isinstance(cmd,str) else cmd for cmd in cmds];"
        "changed or (print('placeholder not used in test_command') or sys.exit(0));"
        "info['test_command']=new_cmds;"
        "p.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8');"
        "print('rewritten test_command exploit path');"
        "print(f'replacement={replacement}');"
    )
    out = env.execute(f"python3 -c {shlex.quote(py_code)}", cwd=cwd, timeout=timeout)
    text = str(out.get("output", ""))
    changed = "rewritten test_command exploit path" in text
    tail = "\n".join(text.splitlines()[-20:])
    return changed, tail


def _remove_vulnerability_info_file(
    env: Any,
    *,
    cwd: str,
    timeout: int,
) -> None:
    """Hide vulnerability metadata from the model workspace before repair."""
    _run_or_raise(
        env,
        "rm -f VUL4C/vulnerability_info.json",
        cwd=cwd,
        timeout=timeout,
        step_name="remove_vulnerability_info",
    )


def _looks_like_diff_patch(text: str) -> bool:
    normalized = text.strip()
    return bool(normalized) and ("@@ " in normalized or "\n@@" in normalized) and ("--- " in normalized) and ("+++ " in normalized)


def _capture_hidden_original(
    env: Any,
    *,
    container_instance_dir: str,
    target_rel_path: str,
    hidden_original_path: Path,
) -> None:
    """Capture original vulnerable file to host-side hidden path (not visible to the model)."""
    py_code = (
        "import base64,pathlib,sys;"
        f"p=pathlib.Path({target_rel_path!r});"
        "p.exists() or (print('missing target file') or sys.exit(2));"
        "sys.stdout.write('MSWEA_ORIG_BEGIN\\n');"
        "sys.stdout.write(base64.b64encode(p.read_bytes()).decode('ascii'));"
        "sys.stdout.write('\\nMSWEA_ORIG_END\\n');"
    )
    output = _run_or_raise(
        env,
        f"python3 -c {shlex.quote(py_code)}",
        cwd=container_instance_dir,
        timeout=120,
        step_name="capture_original",
    )
    raw = str(output.get("output", ""))
    begin_marker = "MSWEA_ORIG_BEGIN\n"
    end_marker = "\nMSWEA_ORIG_END"
    begin = raw.find(begin_marker)
    end = raw.find(end_marker, begin + len(begin_marker))
    if begin < 0 or end < 0:
        msg = "capture_original marker not found"
        raise RuntimeError(msg)
    encoded = raw[begin + len(begin_marker) : end].strip()
    if not encoded:
        raise RuntimeError("capture_original payload is empty")
    try:
        decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001
        msg = "capture_original payload is not valid base64"
        raise RuntimeError(msg) from exc

    hidden_original_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_original_path.write_bytes(decoded)


def _build_fallback_patch(current_path: Path, hidden_original_path: Path, file_name: str) -> str:
    """Build unified diff from hidden original and final source state."""
    if not current_path.exists() or not hidden_original_path.exists():
        return ""

    old_text = hidden_original_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    new_text = current_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    if old_text == new_text:
        return ""

    diff_lines = difflib.unified_diff(
        old_text,
        new_text,
        fromfile=f"source/{file_name}",
        tofile=f"source/{file_name}",
        lineterm="",
    )
    return "\n".join(diff_lines) + "\n"


def _patch_mentions_target_file(patch_text: str, file_name: str) -> bool:
    target = f"source/{file_name}"
    return target in patch_text


def process_instance(
    instance: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    progress_manager: RunBatchProgressManager,
    *,
    retriever: Any,
    run_mode: RunMode,
    runtime_sok_dir: Path,
    workspace_root: Path,
    vul4c_data_dir: Path,
    docker_image: str,
    docker_timeout: int,
    container_timeout: str,
    checkout_timeout: int,
    temporal_filter: bool,
    cleanup_instance_workspace: bool,
) -> None:
    """Process one Vul4C instance."""
    instance_id = str(instance.get("cve_id", ""))
    if not instance_id:
        raise ValueError("Missing cve_id in instance")
    instance_id = _validate_instance_id(instance_id)

    raw_traj_dir = output_dir / "raw_traj" / instance_id
    host_instance_dir = workspace_root / instance_id
    raw_traj_dir.mkdir(parents=True, exist_ok=True)

    _remove_json_entry(output_dir / "preds.json", instance_id)
    _remove_json_entry(output_dir / "patches.json", instance_id)
    (raw_traj_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    progress_manager.on_instance_start(instance_id)

    cve_desc = str(instance.get("cve_description", "") or instance.get("nvd_info", ""))
    cwe_id, cwe_desc = _build_cwe_text(instance)
    file_name = str(instance.get("file_name", "")).strip().lstrip("/")
    project = str(instance.get("project", ""))
    test_info = str(instance.get("test_info", ""))

    experience_context = ""
    if run_mode != RunMode.BASELINE and retriever is not None:
        try:
            experience_context = _retrieve_experience_context(instance, retriever, run_mode, config, temporal_filter)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Experience retrieval failed for {instance_id}: {exc}")

    container_instance_dir = f"/work/{instance_id}"
    target_rel_path = f"source/{file_name}"
    orig_rel_path = f"VUL4C/originals/{Path(file_name).name}.orig"
    hidden_original_path = output_dir / "_hidden_originals" / instance_id / target_rel_path

    agent = None
    env = None
    extra_info = None
    exit_status = None
    result = None

    try:
        base_env_config = copy.deepcopy(config.get("environment", {}))
        run_args = list(base_env_config.get("run_args", ["--rm"]))
        if "--rm" not in run_args:
            run_args.insert(0, "--rm")
        if "--user" not in run_args and "-u" not in run_args:
            run_args.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        run_args = run_args + [
            "-v",
            f"{runtime_sok_dir.resolve()}:/home/SoK-main:ro",
            "-v",
            f"{workspace_root.resolve()}:/work",
        ]
        run_args = _enforce_docker_memory_limit(run_args, _DOCKER_MEMORY_LIMIT)
        if not _run_args_has_mount_target(run_args, "/vul4c_data"):
            run_args = run_args + [
                "-v",
                f"{vul4c_data_dir.resolve()}:/vul4c_data",
            ]

        env_config = {
            **base_env_config,
            "environment_class": "docker",
            "image": docker_image,
            "cwd": container_instance_dir,
            "timeout": docker_timeout,
            "container_timeout": container_timeout,
            "run_args": run_args,
        }
        env = get_environment(env_config)

        progress_manager.update_instance_status(instance_id, "Preparing workspace")
        _run_or_raise(env, "mkdir -p /work", cwd="/", timeout=60, step_name="mkdir")
        _ensure_vul4c_data_config(env, timeout=60)
        _run_or_raise(
            env,
            f"rm -rf -- {shlex.quote(container_instance_dir)}",
            cwd="/",
            timeout=60,
            step_name="workspace_cleanup",
        )

        progress_manager.update_instance_status(instance_id, "Checkout")
        _run_or_raise(
            env,
            f"vul4c checkout -i {shlex.quote(instance_id)} -d {shlex.quote(container_instance_dir)}",
            cwd="/",
            timeout=checkout_timeout,
            step_name="checkout",
        )

        changed, rewrite_log = _rewrite_test_command_exploit_path(
            env,
            cwd=container_instance_dir,
            timeout=120,
        )
        if rewrite_log:
            logger.info(f"[{instance_id}] test_command exploit path check: {rewrite_log}")
        if changed:
            if extra_info is None:
                extra_info = {}
            extra_info["test_command_exploit_path_rewritten"] = True

        progress_manager.update_instance_status(instance_id, "Hide vulnerability metadata")
        _remove_vulnerability_info_file(
            env,
            cwd=container_instance_dir,
            timeout=60,
        )

        progress_manager.update_instance_status(instance_id, "Capture hidden baseline")
        _capture_hidden_original(
            env,
            container_instance_dir=container_instance_dir,
            target_rel_path=target_rel_path,
            hidden_original_path=hidden_original_path,
        )

        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(
            task="",
            CVE_ID=instance_id,
            PROJECT=project,
            FILE_NAME=file_name,
            ORIG_FILE_PATH=orig_rel_path,
            CVE_DESCRIPTION=cve_desc,
            CWE_ID=cwe_id,
            CWE_DESCRIPTION=cwe_desc,
            TEST_INFO=test_info,
            experience_context=experience_context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error processing {instance_id}: {exc}", exc_info=True)
        exit_status, result = type(exc).__name__, str(exc)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        try:
            raw_result = result or ""
            patch_text = raw_result
            current_source_path = host_instance_dir / "source" / file_name
            fallback_patch = _build_fallback_patch(current_source_path, hidden_original_path, file_name)
            need_fallback = (not _looks_like_diff_patch(patch_text)) or (
                _looks_like_diff_patch(patch_text) and not _patch_mentions_target_file(patch_text, file_name)
            )
            if need_fallback:
                if fallback_patch:
                    patch_text = fallback_patch
                    if extra_info is None:
                        extra_info = {}
                    extra_info["fallback_patch_generated"] = True
                    extra_info["fallback_reason"] = f"exit_status={exit_status}"
            if patch_text != raw_result:
                if extra_info is None:
                    extra_info = {}
                extra_info["raw_submission"] = raw_result

            save_traj(
                agent,
                raw_traj_dir / f"{instance_id}.traj.json",
                exit_status=exit_status,
                result=patch_text,
                extra_info=extra_info,
                instance_id=instance_id,
                print_fct=logger.info,
            )

            model_name = model.config.model_name
            _update_json_entry(
                output_dir / "preds.json",
                instance_id,
                {
                    "model_name_or_path": model_name,
                    "instance_id": instance_id,
                    "model_patch": patch_text,
                },
            )
            _update_json_entry(
                output_dir / "patches.json",
                instance_id,
                {
                    "instance_id": instance_id,
                    "cve_id": instance_id,
                    "project": project,
                    "file_name": file_name,
                    "model_name_or_path": model_name,
                    "model_patch": patch_text,
                    "exit_status": exit_status,
                },
            )
            progress_manager.on_instance_end(instance_id, exit_status)
        finally:
            _cleanup_instance_resources(
                env,
                container_instance_dir=container_instance_dir,
                host_instance_dir=host_instance_dir,
                cleanup_host_workspace=cleanup_instance_workspace,
            )


@app.command(help="Run mini-SWE-agent on Vul4C instances.")
def main(
    dataset: str = typer.Option("SoK/Vul4C_Src/Vul4C.json", "-d", "--dataset", help="Vul4C json dataset path."),
    output: str = typer.Option("", "-o", "--output", help="Output directory."),
    workers: int = typer.Option(16, "-w", "--workers", help="Number of worker threads."),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use."),
    model_class: str | None = typer.Option(
        None,
        "--model-class",
        help="Model class to use (e.g., litellm_tool).",
    ),
    redo_existing: bool = typer.Option(
        False,
        "--redo-existing/--no-redo-existing",
        help="Redo instances even if raw_traj/<CVE>/<CVE>.traj.json already exists.",
    ),
    config_spec: Path = typer.Option(
        builtin_config_dir / "extra" / "vul4c.yaml",
        "--config",
        "-C",
        help="Path to config file.",
    ),
    filter_cve_ids: str = typer.Option("", "--filter", "-f", help="Comma-separated CVE IDs to run."),
    max_instances: int = typer.Option(0, "--max-instances", help="Run at most N instances (0 = all)."),
    run_mode: RunMode = typer.Option(
        RunMode.RETRIEVE_ONLY,
        "--mode",
        help="baseline or retrieve-only.",
    ),
    exp_db_path: str | None = typer.Option(None, "--exp-db", help="ExpBank vector DB path."),
    exp_model_path: str | None = typer.Option(None, "--exp-model", help="ExpBank embedding model path."),
    exclude_current_cve: bool = typer.Option(
        True,
        "--exclude-current-cve/--no-exclude-current-cve",
        help="Exclude current CVE from retrieval.",
    ),
    exp_retrieve_k: int = typer.Option(10, "--exp-retrieve-k", help="Top-k retrieved experiences."),
    exp_return_n: int = typer.Option(5, "--exp-return-n", help="Returned experiences after ranking."),
    exp_sort_by_quality: bool = typer.Option(
        True,
        "--exp-sort-by-quality/--no-exp-sort-by-quality",
        help="Sort retrieved experiences by quality score.",
    ),
    temporal_filter: bool = typer.Option(
        False,
        "--temporal-filter/--no-temporal-filter",
        help="Only keep experiences from earlier CVEs.",
    ),
    docker_image: str = typer.Option("vul4c/vul4c:3.0", "--docker-image", help="Docker image for Vul4C."),
    docker_timeout: int = typer.Option(600, "--docker-timeout", help="Default command timeout in seconds."),
    container_timeout: str = typer.Option("6h", "--container-timeout", help="Container lifetime."),
    checkout_timeout: int = typer.Option(900, "--checkout-timeout", help="Checkout timeout in seconds."),
    sok_path: str = typer.Option("SoK", "--sok-path", help="Source SoK repository path."),
    runtime_sok_dir: str = typer.Option("", "--runtime-sok-dir", help="Prepared runtime SoK path."),
    workspace_dir: str = typer.Option("", "--workspace-dir", help="Host workspace mounted to /work."),
    vul4c_data_dir: str = typer.Option(
        "",
        "--vul4c-data-dir",
        help="Host directory mounted to /vul4c_data (default: output/_vul4c_data_tmp).",
    ),
    sanitize_benchmark: bool = typer.Option(
        True,
        "--sanitize-benchmark/--no-sanitize-benchmark",
        help="Copy only exploit files into runtime SoK snapshot.",
    ),
    auto_eval: bool = typer.Option(
        True,
        "--auto-eval/--no-auto-eval",
        help="Run automatic evaluation after patch generation.",
    ),
    eval_workers: int = typer.Option(16, "--eval-workers", help="Parallel workers for evaluation."),
    eval_stage_timeout: int = typer.Option(1200, "--eval-stage-timeout", help="Per-stage timeout for evaluation."),
    eval_progress: bool = typer.Option(
        True,
        "--eval-progress/--no-eval-progress",
        help="Show tqdm progress bar for automatic evaluation.",
    ),
    eval_skip_existing: bool = typer.Option(
        True,
        "--eval-skip-existing/--no-eval-skip-existing",
        help="Skip CVEs already present in evaluation/results.json.",
    ),
) -> None:
    config_path = get_config_path(config_spec)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    logger.info(f"Loading agent config from '{config_path}'")

    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    output_path = Path(output) if output else _default_output_path(
        config.get("model", {}).get("model_name"), exp_retrieve_k, exp_return_n
    )
    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "minisweagent.log")
    logger.info(f"Results will be saved to {output_path}")

    dataset_path = Path(dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    instances = _load_vul4c_dataset(dataset_path)
    logger.info(f"Loaded {len(instances)} instances from {dataset_path}")

    if filter_cve_ids:
        filter_set = {x.strip() for x in filter_cve_ids.split(",") if x.strip()}
        instances = [x for x in instances if str(x.get("cve_id", "")) in filter_set]
        logger.info(f"After filtering, {len(instances)} instances remain")

    if max_instances > 0:
        instances = instances[:max_instances]
        logger.info(f"Truncated to first {len(instances)} instances")

    if not redo_existing:
        before_count = len(instances)
        instances = [
            x for x in instances if not _has_completed_traj(output_path, str(x.get("cve_id", "")))
        ]
        skipped_count = before_count - len(instances)
        logger.info(
            "Skipping completed instances by trajectory JSON: "
            f"skipped={skipped_count}, remaining={len(instances)}"
        )

    if not instances:
        logger.warning("No instances to run.")
        return

    sok_source = Path(sok_path).resolve()
    if not sok_source.exists():
        raise FileNotFoundError(f"SoK path not found: {sok_source}")

    tmp_mount_root = _default_tmp_mount_root(output_path)
    user_provided_runtime_dir = bool(runtime_sok_dir)
    runtime_dir = Path(runtime_sok_dir).resolve() if runtime_sok_dir else tmp_mount_root / "_vul4c_runtime_sok"
    runtime_dir = _prepare_sok_runtime_snapshot(sok_source, runtime_dir, sanitize_benchmark=sanitize_benchmark)
    logger.info(f"Runtime SoK snapshot: {runtime_dir}")

    user_provided_workspace_dir = bool(workspace_dir)
    workspace_root = Path(workspace_dir).resolve() if workspace_dir else tmp_mount_root / "_vul4c_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Workspace mount root: {workspace_root}")

    user_provided_vul4c_data_dir = bool(vul4c_data_dir)
    resolved_vul4c_data_dir = (
        Path(vul4c_data_dir).resolve() if vul4c_data_dir else _default_vul4c_data_dir(output_path).resolve()
    )
    resolved_vul4c_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Vul4C data mount root (/vul4c_data): {resolved_vul4c_data_dir}")

    retriever = None
    resolved_exp_db = str(Path(exp_db_path).resolve()) if exp_db_path else str(EXPBANK_ROOT / "vec_DB")
    resolved_exp_model = (
        str(Path(exp_model_path).resolve()) if exp_model_path else str(EXPBANK_ROOT / "models" / "bge-large-en-v1.5")
    )
    if run_mode != RunMode.BASELINE:
        if not EXPBANK_AVAILABLE:
            logger.warning("ExpBank is unavailable, retrieve-only will continue without experience context.")
        else:
            logger.info(f"Initializing ExpBank retriever: db={resolved_exp_db}, model={resolved_exp_model}")
            retriever = ExperienceRetriever(db_path=resolved_exp_db, model_path=resolved_exp_model)

    config["exclude_current_cve"] = exclude_current_cve
    config["exp_retrieve_k"] = exp_retrieve_k
    config["exp_return_n"] = exp_return_n
    config["exp_sort_by_quality"] = exp_sort_by_quality

    run_config = {
        "basic": {
            "dataset": str(dataset_path),
            "output": str(output_path),
            "workers": workers,
            "model": config.get("model", {}).get("model_name", ""),
            "model_class": config.get("model", {}).get("model_class", ""),
            "config_file": str(config_path),
            "num_instances": len(instances),
        },
        "experience_enhancement": {
            "run_mode": run_mode.value,
            "exp_db_path": resolved_exp_db,
            "exp_model_path": resolved_exp_model,
            "exclude_current_cve": exclude_current_cve,
            "exp_retrieve_k": exp_retrieve_k,
            "exp_return_n": exp_return_n,
            "exp_sort_by_quality": exp_sort_by_quality,
            "temporal_filter": temporal_filter,
        },
        "docker": {
            "docker_image": docker_image,
            "docker_timeout": docker_timeout,
            "container_timeout": container_timeout,
            "memory_limit": _DOCKER_MEMORY_LIMIT,
            "checkout_timeout": checkout_timeout,
            "runtime_sok_dir": str(runtime_dir),
            "workspace_dir": str(workspace_root),
            "vul4c_data_dir": str(resolved_vul4c_data_dir),
            "sanitize_benchmark": sanitize_benchmark,
        },
        "evaluation": {
            "auto_eval": auto_eval,
            "eval_workers": eval_workers,
            "eval_stage_timeout": eval_stage_timeout,
            "eval_progress": eval_progress,
            "eval_skip_existing": eval_skip_existing,
            "evaluation_dir": str(output_path / "evaluation"),
        },
    }
    (output_path / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    (output_path / "raw_traj").mkdir(parents=True, exist_ok=True)
    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]) -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                cve_id = futures[future]
                logger.error(f"Error in future for {cve_id}: {exc}")
                progress_manager.on_uncaught_exception(cve_id, exc)

    try:
        with Live(progress_manager.render_group, refresh_per_second=4):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_instance,
                        instance,
                        output_path,
                        config,
                        progress_manager,
                        retriever=retriever,
                        run_mode=run_mode,
                        runtime_sok_dir=runtime_dir,
                        workspace_root=workspace_root,
                        vul4c_data_dir=resolved_vul4c_data_dir,
                        docker_image=docker_image,
                        docker_timeout=docker_timeout,
                        container_timeout=container_timeout,
                        checkout_timeout=checkout_timeout,
                        temporal_filter=temporal_filter,
                        cleanup_instance_workspace=not user_provided_workspace_dir,
                    ): str(instance.get("cve_id", ""))
                    for instance in instances
                }
                try:
                    process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("Cancelling pending jobs. Press Ctrl-C again to exit immediately.")
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    process_futures(futures)

        if auto_eval:
            preds_path = output_path / "preds.json"
            evaluation_dir = output_path / "evaluation"
            logger.info("Starting automatic evaluation...")
            try:
                eval_result = run_evaluation(
                    preds_path=preds_path,
                    output_dir=evaluation_dir,
                    workers=eval_workers,
                    docker_image=docker_image,
                    stage_timeout=eval_stage_timeout,
                    container_timeout=container_timeout,
                    filter_cve_ids=filter_cve_ids,
                    skip_existing=eval_skip_existing,
                    sok_path=str(sok_source),
                    runtime_sok_dir=runtime_dir,
                    workspace_root=workspace_root,
                    sanitize_benchmark=sanitize_benchmark,
                    show_progress=eval_progress,
                )
                logger.info(
                    "Automatic evaluation finished: "
                    f"evaluated_this_run={eval_result.get('evaluated_this_run', 0)}, "
                    f"total={eval_result.get('total', 0)}, "
                    f"summary={eval_result.get('summary_path', '')}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Automatic evaluation failed: {exc}", exc_info=True)
    finally:
        # Always clean run-scoped working directories after completion.
        # For user-provided paths, keep ownership with the caller and do not remove.
        if not user_provided_workspace_dir:
            _safe_remove_tree(workspace_root)
        if not user_provided_runtime_dir:
            _safe_remove_tree(runtime_dir)
        if not user_provided_vul4c_data_dir:
            _safe_remove_tree(resolved_vul4c_data_dir)

    logger.info("Vul4C single-round run finished.")


if __name__ == "__main__":
    app()
