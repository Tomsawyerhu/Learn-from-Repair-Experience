#!/usr/bin/env python3
"""Evaluate Vul4C patches in Docker with stage-based status classification."""

from __future__ import annotations

import concurrent.futures
import json
import shlex
import shutil
import subprocess
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

from minisweagent import logger

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_NON_TTY_SHELL_NOISE_PATTERNS = (
    "mesg: ttyname failed: Inappropriate ioctl for device",
)
_DOCKER_MEMORY_LIMIT = "6g"
_MEMORY_LIMIT_PATTERNS = (
    "out of memory",
    "cannot allocate memory",
    "memory cgroup out of memory",
    "oom-kill",
    "oom killed",
    "oomkilled",
)


class _NoopProgress:
    def __enter__(self) -> _NoopProgress:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def update(self, _: int = 1) -> None:
        return

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return


@dataclass
class EvalResult:
    cve_id: str
    status: str
    duration: float
    patch_apply_ok: bool = False
    compile_ok: bool = False
    old_reproduced: bool = False
    new_reproduced: bool = False
    tests_ok: bool = False
    error_message: str = ""
    stage_returncodes: dict[str, int] | None = None


def _run_cmd(args: list[str], timeout: int) -> tuple[int, str]:
    def _strip_known_shell_noise(text: str) -> str:
        if not text:
            return ""
        lines = text.splitlines()
        filtered = [
            line
            for line in lines
            if not any(pattern in line for pattern in _NON_TTY_SHELL_NOISE_PATTERNS)
        ]
        return "\n".join(filtered)

    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return result.returncode, _strip_known_shell_noise(result.stdout)
    except subprocess.TimeoutExpired as exc:
        raw_stdout = exc.stdout or ""
        if isinstance(raw_stdout, bytes):
            raw_stdout = raw_stdout.decode("utf-8", errors="replace")
        return 124, _strip_known_shell_noise(raw_stdout + "\n[TIMEOUT] Host-side command timeout")


def _run_docker_exec(container_name: str, command: str, timeout_s: int) -> tuple[int, str]:
    # timeout inside container gives stage-level timeout signal (rc=124)
    wrapped = f"timeout {timeout_s}s bash -c {json.dumps(command)}"
    return _run_cmd(["docker", "exec", container_name, "bash", "-c", wrapped], timeout=timeout_s + 30)


def _looks_like_memory_limit_signal(returncode: int, output: str) -> bool:
    lowered = (output or "").lower()
    if any(pattern in lowered for pattern in _MEMORY_LIMIT_PATTERNS):
        return True
    if returncode == 137 and ("killed" in lowered or "memory" in lowered or "oom" in lowered):
        return True
    return False


@contextmanager
def _optional_semaphore(semaphore: threading.Semaphore | None):
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _start_container(
    container_name: str,
    docker_image: str,
    runtime_sok_dir: Path,
    workspace_root: Path,
    container_timeout: str,
) -> tuple[bool, str]:
    args = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "--memory",
        _DOCKER_MEMORY_LIMIT,
        "-v",
        f"{runtime_sok_dir.resolve()}:/home/SoK-main:ro",
        "-v",
        f"{workspace_root.resolve()}:/work",
        docker_image,
        "sleep",
        container_timeout,
    ]
    rc, out = _run_cmd(args, timeout=120)
    return rc == 0, out


def _stop_container(container_name: str) -> None:
    _run_cmd(["docker", "rm", "-f", container_name], timeout=30)


def _looks_reproduced(text: str) -> bool:
    if not text:
        return False
    patterns = [
        "AddressSanitizer",
        "heap-buffer-overflow",
        "stack-buffer-overflow",
        "Segmentation fault",
        "runtime error",
        "buffer over-read",
        "invalid_data",
    ]
    lowered = text.lower()
    return any(p.lower() in lowered for p in patterns)


def _looks_test_failed(text: str) -> bool:
    if not text:
        return False
    bad_patterns = [
        "fatal error:",
        "no such file or directory",
        "error:",
        "# FAIL:",
        "# ERROR:",
    ]
    lowered = text.lower()
    return any(p.lower() in lowered for p in bad_patterns)


def _is_diff_patch(patch_text: str) -> bool:
    normalized = patch_text.strip()
    if not normalized:
        return False
    return ("@@" in normalized) and ("--- " in normalized) and ("+++ " in normalized)


def _normalize_patch_path(raw_path: str) -> str | None:
    """Normalize patch header path to comparable repository-style relative path."""
    path = raw_path.strip()
    if not path:
        return None
    path = path.split("\t", 1)[0].split(" ", 1)[0]
    if path in {"/dev/null", "dev/null"}:
        return None
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("source/"):
        return path
    marker = "/source/"
    if marker in path:
        return "source/" + path.split(marker, 1)[1]
    return path


def _extract_patch_target_files(patch_text: str) -> set[str]:
    """Extract normalized target files from patch `+++` headers."""
    files: set[str] = set()
    for line in patch_text.splitlines():
        if not line.startswith("+++ "):
            continue
        normalized = _normalize_patch_path(line[4:])
        if normalized:
            files.add(normalized)
    return files


def _get_expected_target_file(cve_host_dir: Path) -> str | None:
    """Read expected vulnerable source path from vulnerability metadata."""
    vuln_info_path = cve_host_dir / "VUL4C" / "vulnerability_info.json"
    if not vuln_info_path.exists():
        return None
    try:
        info = json.loads(vuln_info_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    file_name = info.get("file_name")
    if not isinstance(file_name, str):
        return None
    normalized = file_name.strip().lstrip("/").replace("\\", "/")
    if not normalized:
        return None
    return f"source/{normalized}"


def _build_localization_error_message(
    *,
    expected_file: str | None,
    patch_files: set[str],
    patch_text: str,
    cve_host_dir: Path,
) -> str:
    parts = ["[stage]\nlocalization_check"]
    if expected_file is None:
        parts.append("[expected_target]\n<missing from vulnerability_info.json:file_name>")
    else:
        parts.append(f"[expected_target]\n{expected_file}")

    if patch_files:
        parts.append("[patch_targets]\n" + "\n".join(sorted(patch_files)))
    else:
        parts.append("[patch_targets]\n<none>")

    header_lines = [line for line in patch_text.splitlines() if line.startswith(("--- ", "+++ ", "@@"))]
    if header_lines:
        parts.append("[patch_headers]\n" + _tail_text("\n".join(header_lines), max_lines=80, max_chars=8000))

    vuln_info_path = cve_host_dir / "VUL4C" / "vulnerability_info.json"
    if vuln_info_path.exists():
        parts.append("[vulnerability_info_path]\n" + str(vuln_info_path))
    return "\n\n".join(parts)


def _prepare_sok_runtime_snapshot(sok_source: Path, runtime_root: Path, sanitize_benchmark: bool) -> Path:
    marker = runtime_root / ".snapshot_ready"
    mode_marker = "sanitized" if sanitize_benchmark else "full"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == mode_marker:
        return runtime_root

    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    src_vul4c = sok_source / "Vul4C_Src"
    src_benchmark = sok_source / "Vul4C-Benchmark"
    if not src_vul4c.exists() or not src_benchmark.exists():
        raise FileNotFoundError(f"Invalid SoK path for runtime snapshot: {sok_source}")

    shutil.copytree(src_vul4c, runtime_root / "Vul4C_Src")
    dst_benchmark = runtime_root / "Vul4C-Benchmark"
    if not sanitize_benchmark:
        shutil.copytree(src_benchmark, dst_benchmark)
    else:
        for project_dir in sorted(src_benchmark.iterdir()):
            if not project_dir.is_dir():
                continue
            for cve_dir in sorted(project_dir.iterdir()):
                if not cve_dir.is_dir():
                    continue
                for file_path in sorted(cve_dir.iterdir()):
                    if file_path.is_file() and file_path.name.startswith("exploit"):
                        dst_dir = dst_benchmark / project_dir.name / cve_dir.name
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dst_dir / file_path.name)

    marker.write_text(mode_marker, encoding="utf-8")
    return runtime_root


def _read_file_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _tail_text(text: str, *, max_lines: int = 120, max_chars: int = 20000) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    merged = "\n".join(lines)
    if len(merged) > max_chars:
        merged = merged[-max_chars:]
    return merged


def _collect_compile_context(cve_host_dir: Path) -> str:
    parts: list[str] = []
    vuln_info_path = cve_host_dir / "VUL4C" / "vulnerability_info.json"
    if vuln_info_path.exists():
        try:
            info = json.loads(vuln_info_path.read_text(encoding="utf-8"))
            compile_cmd = info.get("compile_command", [])
            if isinstance(compile_cmd, list):
                cmd_preview = "\n".join(f"{idx + 1}. {cmd}" for idx, cmd in enumerate(compile_cmd[:20]))
                parts.append("[compile_command]\n" + cmd_preview)
        except Exception as exc:  # noqa: BLE001
            parts.append(f"[compile_command]\nFailed to parse vulnerability_info.json: {exc}")

    source_dir = cve_host_dir / "source"
    if source_dir.exists():
        key_files = [
            "Makefile",
            "makefile",
            "CMakeLists.txt",
            "configure",
            "configure.ac",
            "autogen.sh",
            "meson.build",
        ]
        present = [name for name in key_files if (source_dir / name).exists()]
        parts.append(f"[source_hints]\nsource_dir={source_dir}\nkey_files_present={present}")
    else:
        parts.append(f"[source_hints]\nsource_dir_missing={source_dir}")

    return "\n\n".join(parts)


def _build_stage_error_message(stage: str, docker_output: str, cve_host_dir: Path) -> str:
    parts = [f"[stage]\n{stage}"]
    docker_tail = _tail_text(docker_output)
    if docker_tail:
        parts.append("[docker_exec_output_tail]\n" + docker_tail)

    stage_logs: dict[str, list[Path]] = {
        "compile_old": [cve_host_dir / "VUL4C" / "compile.log"],
        "compile_new": [cve_host_dir / "VUL4C" / "compile.log"],
        "compile_test": [cve_host_dir / "VUL4C" / "compile.log"],
        "reproduce_old": [cve_host_dir / "VUL4C" / "reproducing.log"],
        "reproduce_new": [cve_host_dir / "VUL4C" / "reproducing.log"],
        "test_new": [cve_host_dir / "VUL4C" / "testing.log"],
    }
    for log_path in stage_logs.get(stage, []):
        if not log_path.exists():
            continue
        log_tail = _tail_text(log_path.read_text(encoding="utf-8", errors="replace"))
        if log_tail:
            parts.append(f"[{log_path.name}_tail]\n{log_tail}")

    if stage.startswith("compile"):
        compile_context = _collect_compile_context(cve_host_dir)
        if compile_context:
            parts.append(compile_context)

    return "\n\n".join(parts)


def _build_memory_limit_error_message(stage: str, docker_output: str, cve_host_dir: Path) -> str:
    base = _build_stage_error_message(stage, docker_output, cve_host_dir)
    marker = f"[memory_limit]\nexceeded (limit={_DOCKER_MEMORY_LIMIT})"
    if not base:
        return marker
    return base + "\n\n" + marker


def _remove_leading_make_clean(
    *,
    container_name: str,
    cve_container_dir: str,
    stage_timeout: int,
) -> tuple[bool, str]:
    vuln_info_file = f"{cve_container_dir}/VUL4C/vulnerability_info.json"
    py_code = (
        "import json,pathlib,sys;"
        f"p=pathlib.Path({vuln_info_file!r});"
        "p.exists() or (print('vulnerability_info.json missing') or sys.exit(2));"
        "info=json.loads(p.read_text(encoding='utf-8'));"
        "compile_cmd=info.get('compile_command');"
        "(isinstance(compile_cmd,list) and bool(compile_cmd)) or (print('compile_command missing or empty') or sys.exit(4));"
        "first=compile_cmd[0];"
        "isinstance(first,str) or (print('compile_command[0] is not string') or sys.exit(5));"
        "first.strip().startswith('make clean') or (print('compile_command does not start with make clean') or sys.exit(6));"
        "info['compile_command']=compile_cmd[1:];"
        "p.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8');"
        "print('removed leading make clean from compile_command');"
    )
    fix_cmd = f"python3 -c {shlex.quote(py_code)}"
    rc, out = _run_docker_exec(container_name, fix_cmd, stage_timeout)
    return rc == 0, _tail_text(out, max_lines=40, max_chars=4000)


def _rewrite_test_command_exploit_path(
    *,
    container_name: str,
    cve_container_dir: str,
    stage_timeout: int,
) -> tuple[bool, str]:
    """
    Best-effort fix for datasets where test_command uses ../VUL4C/exploit
    but checkout provides exploit with suffix (e.g., exploit.tif).
    """
    info_path = f"{cve_container_dir}/VUL4C/vulnerability_info.json"
    py_code = (
        "import json,pathlib,sys;"
        f"p=pathlib.Path({info_path!r});"
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
    rc, out = _run_docker_exec(container_name, f"python3 -c {shlex.quote(py_code)}", stage_timeout)
    if rc != 0:
        return False, _tail_text(out, max_lines=40, max_chars=4000)

    output_tail = _tail_text(out, max_lines=40, max_chars=4000)
    changed = "rewritten test_command exploit path" in output_tail
    return changed, output_tail


def _run_compile_with_tolerance(
    *,
    cve_id: str,
    stage_key: str,
    compile_cmd: str,
    container_name: str,
    cve_container_dir: str,
    stage_timeout: int,
    stage_rc: dict[str, int],
) -> tuple[int, str]:
    rc, out = _run_docker_exec(container_name, f"vul4c compile -d {cve_container_dir} {compile_cmd}".strip(), stage_timeout)
    stage_rc[f"{stage_key}_initial"] = rc
    stage_rc[stage_key] = rc
    if rc == 0:
        return rc, out

    stage_rc[f"{stage_key}_tolerance_attempted"] = 1
    changed, reason = _remove_leading_make_clean(
        container_name=container_name,
        cve_container_dir=cve_container_dir,
        stage_timeout=stage_timeout,
    )
    stage_rc[f"{stage_key}_tolerance_removed"] = int(changed)
    if not changed:
        logger.info(f"[{cve_id}] {stage_key} tolerance skipped: {reason}")
        return rc, out + "\n\n[compile_tolerance]\n" + reason

    logger.warning(f"[{cve_id}] {stage_key} retry without `make clean`: {reason}")
    retry_rc, retry_out = _run_docker_exec(
        container_name, f"vul4c compile -d {cve_container_dir} {compile_cmd}".strip(), stage_timeout
    )
    stage_rc[f"{stage_key}_retry"] = retry_rc
    stage_rc[stage_key] = retry_rc

    merged_out = (
        out
        + "\n\n[compile_tolerance]\n"
        + reason
        + "\n\n[compile_retry_output]\n"
        + retry_out
    )
    return retry_rc, merged_out


def evaluate_one(
    cve_id: str,
    patch_text: str,
    *,
    docker_image: str,
    runtime_sok_dir: Path,
    workspace_root: Path,
    container_timeout: str,
    stage_timeout: int,
    old_stage_semaphore: threading.Semaphore | None = None,
) -> EvalResult:
    start = time.time()
    stage_rc: dict[str, int] = {}

    if not patch_text.strip():
        return EvalResult(cve_id=cve_id, status="empty_patch", duration=0.0, stage_returncodes=stage_rc)
    if not _is_diff_patch(patch_text):
        return EvalResult(cve_id=cve_id, status="not_a_diff_patch", duration=0.0, stage_returncodes=stage_rc)

    # One workspace directory per CVE to avoid collision in parallel workers.
    cve_host_dir = workspace_root / cve_id
    if cve_host_dir.exists():
        try:
            shutil.rmtree(cve_host_dir)
        except Exception as exc:  # noqa: BLE001
            # Old runs may leave root-owned artifacts in bind-mounted paths.
            # Fall back to a unique per-run directory instead of hard failing.
            fallback = workspace_root / f"{cve_id}-{int(time.time() * 1000)}"
            logger.warning(f"Failed to clean {cve_host_dir}, using fallback {fallback}: {exc}")
            cve_host_dir = fallback
    cve_host_dir.mkdir(parents=True, exist_ok=True)
    cve_work_dir_name = cve_host_dir.name
    patch_host_dir = workspace_root / "_eval_patches"
    patch_host_dir.mkdir(parents=True, exist_ok=True)
    patch_host_path = patch_host_dir / f"{cve_work_dir_name}.patch"
    patch_host_path.write_text(patch_text, encoding="utf-8")

    container_name = f"vul4c-eval-{cve_id.lower()}-{int(time.time() * 1000) % 1_000_000}"
    started, start_out = _start_container(
        container_name,
        docker_image=docker_image,
        runtime_sok_dir=runtime_sok_dir,
        workspace_root=workspace_root,
        container_timeout=container_timeout,
    )
    if not started and _looks_like_memory_limit_signal(1, start_out):
        return EvalResult(
            cve_id=cve_id,
            status="memory_limit_exceeded",
            duration=time.time() - start,
            error_message=f"[stage]\nstart_container\n\n[memory_limit]\nexceeded (limit={_DOCKER_MEMORY_LIMIT})\n\n"
            f"[docker_run_output]\n{_tail_text(start_out)}",
            stage_returncodes=stage_rc,
        )
    if not started:
        return EvalResult(
            cve_id=cve_id,
            status="infra_error",
            duration=time.time() - start,
            error_message=f"Failed to start container: {start_out}",
            stage_returncodes=stage_rc,
        )

    try:
        cve_container_dir = f"/work/{cve_work_dir_name}"
        patch_container_path = f"/work/_eval_patches/{cve_work_dir_name}.patch"

        rc, out = _run_docker_exec(
            container_name,
            f"rm -rf {cve_container_dir} && vul4c checkout -i {cve_id} -d {cve_container_dir}",
            stage_timeout,
        )
        stage_rc["checkout"] = rc
        if _looks_like_memory_limit_signal(rc, out):
            return EvalResult(
                cve_id=cve_id,
                status="memory_limit_exceeded",
                duration=time.time() - start,
                error_message=_build_memory_limit_error_message("checkout", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        if rc != 0:
            return EvalResult(
                cve_id=cve_id,
                status="checkout_failed",
                duration=time.time() - start,
                error_message=_build_stage_error_message("checkout", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )

        cmd_rewritten, rewrite_message = _rewrite_test_command_exploit_path(
            container_name=container_name,
            cve_container_dir=cve_container_dir,
            stage_timeout=min(stage_timeout, 120),
        )
        stage_rc["test_command_exploit_path_rewritten"] = int(cmd_rewritten)
        if rewrite_message:
            logger.info(f"[{cve_id}] test_command exploit path check: {rewrite_message}")

        expected_target_file = _get_expected_target_file(cve_host_dir)
        patch_target_files = _extract_patch_target_files(patch_text)
        if expected_target_file is None:
            stage_rc["localization_check"] = 2
            return EvalResult(
                cve_id=cve_id,
                status="infra_error",
                duration=time.time() - start,
                error_message=_build_localization_error_message(
                    expected_file=None,
                    patch_files=patch_target_files,
                    patch_text=patch_text,
                    cve_host_dir=cve_host_dir,
                ),
                stage_returncodes=stage_rc,
            )
        if expected_target_file not in patch_target_files:
            stage_rc["localization_check"] = 1
            return EvalResult(
                cve_id=cve_id,
                status="localization_failed",
                duration=time.time() - start,
                error_message=_build_localization_error_message(
                    expected_file=expected_target_file,
                    patch_files=patch_target_files,
                    patch_text=patch_text,
                    cve_host_dir=cve_host_dir,
                ),
                stage_returncodes=stage_rc,
            )
        stage_rc["localization_check"] = 0

        with _optional_semaphore(old_stage_semaphore):
            rc, out = _run_compile_with_tolerance(
                cve_id=cve_id,
                stage_key="compile_old",
                compile_cmd="",
                container_name=container_name,
                cve_container_dir=cve_container_dir,
                stage_timeout=stage_timeout,
                stage_rc=stage_rc,
            )
        if _looks_like_memory_limit_signal(rc, out):
            return EvalResult(
                cve_id=cve_id,
                status="memory_limit_exceeded",
                duration=time.time() - start,
                error_message=_build_memory_limit_error_message("compile_old", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        if rc != 0:
            return EvalResult(
                cve_id=cve_id,
                status="compile_failed",
                duration=time.time() - start,
                error_message=_build_stage_error_message("compile_old", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )

        with _optional_semaphore(old_stage_semaphore):
            rc, out = _run_docker_exec(container_name, f"vul4c reproduce -d {cve_container_dir}", stage_timeout)
        stage_rc["reproduce_old"] = rc
        if _looks_like_memory_limit_signal(rc, out):
            return EvalResult(
                cve_id=cve_id,
                status="memory_limit_exceeded",
                duration=time.time() - start,
                error_message=_build_memory_limit_error_message("reproduce_old", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        if rc == 124:
            return EvalResult(
                cve_id=cve_id,
                status="reproduce_timeout",
                duration=time.time() - start,
                error_message=_build_stage_error_message("reproduce_old", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        old_reproduce_log = _read_file_if_exists(cve_host_dir / "VUL4C" / "reproducing.log")
        old_reproduced = _looks_reproduced(out + "\n" + old_reproduce_log)

        rc, out = _run_docker_exec(
            container_name,
            f"vul4c apply -d {cve_container_dir} -p {patch_container_path}",
            stage_timeout,
        )
        stage_rc["apply"] = rc
        if _looks_like_memory_limit_signal(rc, out):
            return EvalResult(
                cve_id=cve_id,
                status="memory_limit_exceeded",
                duration=time.time() - start,
                old_reproduced=old_reproduced,
                error_message=_build_memory_limit_error_message("apply", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        if rc != 0:
            return EvalResult(
                cve_id=cve_id,
                status="patch_apply_failed",
                duration=time.time() - start,
                old_reproduced=old_reproduced,
                error_message=_build_stage_error_message("apply", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )

        rc, out = _run_compile_with_tolerance(
            cve_id=cve_id,
            stage_key="compile_new",
            compile_cmd="",
            container_name=container_name,
            cve_container_dir=cve_container_dir,
            stage_timeout=stage_timeout,
            stage_rc=stage_rc,
        )
        if _looks_like_memory_limit_signal(rc, out):
            return EvalResult(
                cve_id=cve_id,
                status="memory_limit_exceeded",
                duration=time.time() - start,
                patch_apply_ok=True,
                old_reproduced=old_reproduced,
                error_message=_build_memory_limit_error_message("compile_new", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        if rc != 0:
            return EvalResult(
                cve_id=cve_id,
                status="compile_failed",
                duration=time.time() - start,
                patch_apply_ok=True,
                old_reproduced=old_reproduced,
                error_message=_build_stage_error_message("compile_new", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )

        rc, out = _run_docker_exec(container_name, f"vul4c reproduce -d {cve_container_dir}", stage_timeout)
        stage_rc["reproduce_new"] = rc
        if _looks_like_memory_limit_signal(rc, out):
            return EvalResult(
                cve_id=cve_id,
                status="memory_limit_exceeded",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=old_reproduced,
                error_message=_build_memory_limit_error_message("reproduce_new", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        if rc == 124:
            return EvalResult(
                cve_id=cve_id,
                status="reproduce_timeout",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=old_reproduced,
                error_message=_build_stage_error_message("reproduce_new", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        new_reproduce_log = _read_file_if_exists(cve_host_dir / "VUL4C" / "reproducing.log")
        new_reproduced = _looks_reproduced(out + "\n" + new_reproduce_log)

        rc, out = _run_compile_with_tolerance(
            cve_id=cve_id,
            stage_key="compile_test",
            compile_cmd="",
            container_name=container_name,
            cve_container_dir=cve_container_dir,
            stage_timeout=stage_timeout,
            stage_rc=stage_rc,
        )
        if _looks_like_memory_limit_signal(rc, out):
            return EvalResult(
                cve_id=cve_id,
                status="memory_limit_exceeded",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=old_reproduced,
                new_reproduced=new_reproduced,
                error_message=_build_memory_limit_error_message("compile_test", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        if rc != 0:
            return EvalResult(
                cve_id=cve_id,
                status="compile_failed",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=old_reproduced,
                new_reproduced=new_reproduced,
                error_message=_build_stage_error_message("compile_test", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )

        rc, out = _run_docker_exec(container_name, f"vul4c test -d {cve_container_dir}", stage_timeout)
        stage_rc["test_new"] = rc
        if _looks_like_memory_limit_signal(rc, out):
            return EvalResult(
                cve_id=cve_id,
                status="memory_limit_exceeded",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=old_reproduced,
                new_reproduced=new_reproduced,
                error_message=_build_memory_limit_error_message("test_new", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        if rc == 124:
            return EvalResult(
                cve_id=cve_id,
                status="test_timeout",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=old_reproduced,
                new_reproduced=new_reproduced,
                error_message=_build_stage_error_message("test_new", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )
        test_log = _read_file_if_exists(cve_host_dir / "VUL4C" / "testing.log")
        test_failed = (rc != 0) or _looks_test_failed(out + "\n" + test_log)

        if not old_reproduced:
            return EvalResult(
                cve_id=cve_id,
                status="reproduce_failed",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=False,
                new_reproduced=new_reproduced,
                tests_ok=not test_failed,
                error_message="Old vulnerability was not reproduced, cannot confirm fix effectiveness.",
                stage_returncodes=stage_rc,
            )
        if new_reproduced:
            return EvalResult(
                cve_id=cve_id,
                status="reproduce_failed",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=True,
                new_reproduced=True,
                tests_ok=not test_failed,
                error_message="Vulnerability still reproduced after patch.",
                stage_returncodes=stage_rc,
            )
        if test_failed:
            return EvalResult(
                cve_id=cve_id,
                status="regression_failed",
                duration=time.time() - start,
                patch_apply_ok=True,
                compile_ok=True,
                old_reproduced=True,
                new_reproduced=False,
                tests_ok=False,
                error_message=_build_stage_error_message("test_new", out, cve_host_dir),
                stage_returncodes=stage_rc,
            )

        return EvalResult(
            cve_id=cve_id,
            status="success",
            duration=time.time() - start,
            patch_apply_ok=True,
            compile_ok=True,
            old_reproduced=True,
            new_reproduced=False,
            tests_ok=True,
            stage_returncodes=stage_rc,
        )
    except Exception as exc:  # noqa: BLE001
        return EvalResult(
            cve_id=cve_id,
            status="error",
            duration=time.time() - start,
            error_message=str(exc),
            stage_returncodes=stage_rc,
        )
    finally:
        _stop_container(container_name)


def _load_predictions(preds_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(preds_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("preds.json must be a dict keyed by CVE ID.")
    return data


def run_evaluation(
    *,
    preds_path: Path,
    output_dir: Path | None = None,
    workers: int = 16,
    docker_image: str = "vul4c/vul4c:3.0",
    stage_timeout: int = 1200,
    container_timeout: str = "6h",
    old_stage_workers: int = 8,
    filter_cve_ids: str = "",
    skip_existing: bool = True,
    sok_path: str = "SoK",
    runtime_sok_dir: Path | None = None,
    workspace_root: Path | None = None,
    sanitize_benchmark: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run Vul4C evaluation and persist results/summary/errors JSON files."""
    preds_path = preds_path.resolve()
    if not preds_path.exists():
        raise FileNotFoundError(f"preds.json not found: {preds_path}")

    predictions = _load_predictions(preds_path)
    all_items: list[tuple[str, dict[str, Any]]] = sorted(predictions.items(), key=lambda x: x[0])

    if filter_cve_ids:
        filter_set = {x.strip() for x in filter_cve_ids.split(",") if x.strip()}
        all_items = [(cve, obj) for cve, obj in all_items if cve in filter_set]

    evaluation_dir = output_dir.resolve() if output_dir else preds_path.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    results_path = evaluation_dir / "results.json"
    summary_path = evaluation_dir / "summary.json"
    errors_path = evaluation_dir / "errors.json"

    existing_results: dict[str, Any] = {}
    if skip_existing and results_path.exists():
        existing_results = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(existing_results, dict):
            all_items = [(cve, obj) for cve, obj in all_items if cve not in existing_results]
        else:
            existing_results = {}

    if not all_items:
        logger.warning("No CVEs to evaluate.")
        return {
            "total": len(existing_results),
            "evaluated_this_run": 0,
            "results_path": str(results_path),
            "summary_path": str(summary_path),
            "errors_path": str(errors_path),
        }

    sok_source = Path(sok_path).resolve()
    runtime_dir = runtime_sok_dir.resolve() if runtime_sok_dir else (evaluation_dir.parent / "_vul4c_runtime_sok")
    runtime_dir = _prepare_sok_runtime_snapshot(sok_source, runtime_dir, sanitize_benchmark=sanitize_benchmark)

    eval_workspace_root = workspace_root.resolve() if workspace_root else (evaluation_dir.parent / "_vul4c_eval_workspace")
    eval_workspace_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating {len(all_items)} CVEs with {workers} workers")
    old_stage_workers = max(1, int(old_stage_workers))
    logger.info(f"compile_old/reproduce_old limited to {old_stage_workers} concurrent workers")
    logger.info(f"Runtime SoK: {runtime_dir}")
    logger.info(f"Workspace: {eval_workspace_root}")

    results: dict[str, dict[str, Any]] = {}
    start = time.time()
    old_stage_semaphore = threading.Semaphore(old_stage_workers)
    run_counter: Counter[str] = Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map: dict[concurrent.futures.Future, str] = {}
        for cve_id, item in all_items:
            patch_text = str(item.get("model_patch", ""))
            future = executor.submit(
                evaluate_one,
                cve_id,
                patch_text,
                docker_image=docker_image,
                runtime_sok_dir=runtime_dir,
                workspace_root=eval_workspace_root,
                container_timeout=container_timeout,
                stage_timeout=stage_timeout,
                old_stage_semaphore=old_stage_semaphore,
            )
            future_map[future] = cve_id

        progress_context: Any
        if show_progress and tqdm is not None:
            progress_context = tqdm(
                total=len(future_map),
                desc="Vul4C Eval",
                unit="cve",
                dynamic_ncols=True,
            )
        else:
            if show_progress and tqdm is None:
                logger.warning("tqdm is unavailable; falling back to non-progress logging mode.")
            progress_context = _NoopProgress()

        with progress_context as pbar:
            for future in concurrent.futures.as_completed(future_map):
                cve_id = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = EvalResult(
                        cve_id=cve_id,
                        status="error",
                        duration=0.0,
                        error_message=f"Unhandled evaluator exception: {exc}",
                    )
                results[cve_id] = asdict(result)
                run_counter[result.status] += 1
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "success": run_counter.get("success", 0),
                        "failed": sum(run_counter.values()) - run_counter.get("success", 0),
                    },
                    refresh=False,
                )

    merged_results = dict(existing_results)
    merged_results.update(results)
    results_path.write_text(json.dumps(merged_results, indent=2, ensure_ascii=False), encoding="utf-8")

    counter = Counter(item["status"] for item in merged_results.values())
    successful_cves = sorted(
        cve_id for cve_id, item in merged_results.items() if str(item.get("status", "")).lower() == "success"
    )
    failed_cves = sorted(cve_id for cve_id in merged_results.keys() if cve_id not in set(successful_cves))
    summary = {
        "total": len(merged_results),
        "evaluated_this_run": len(results),
        "duration_seconds": round(time.time() - start, 2),
        "status_counts": dict(counter),
        "success_rate": round(counter.get("success", 0) * 100.0 / max(len(merged_results), 1), 2),
        "successful_cves": successful_cves,
        "failed_cves": failed_cves,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    errors = [
        {"cve_id": cve_id, "status": item["status"], "error_message": item.get("error_message", "")}
        for cve_id, item in merged_results.items()
        if item.get("status") != "success"
    ]
    errors_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Saved results: {results_path}")
    logger.info(f"Saved summary: {summary_path}")
    logger.info(f"Saved errors: {errors_path}")
    logger.info(
        "Evaluation run status counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(Counter(item['status'] for item in results.values()).items()))
    )
    return {
        "total": len(merged_results),
        "evaluated_this_run": len(results),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "errors_path": str(errors_path),
    }


@app.command(help="Evaluate Vul4C patches from preds.json.")
def main(
    preds: str = typer.Option(..., "--preds", help="Path to preds.json generated by vul4c_exp.py."),
    output_dir: str = typer.Option("", "--output-dir", help="Directory to store evaluation results."),
    workers: int = typer.Option(16, "--workers", help="Parallel workers for evaluation."),
    docker_image: str = typer.Option("vul4c/vul4c:3.0", "--docker-image", help="Vul4C docker image."),
    stage_timeout: int = typer.Option(1200, "--stage-timeout", help="Per-stage timeout in seconds."),
    container_timeout: str = typer.Option("6h", "--container-timeout", help="Container lifetime."),
    old_stage_workers: int = typer.Option(
        8,
        "--old-stage-workers",
        help="Concurrency limit for compile_old/reproduce_old stages.",
    ),
    filter_cve_ids: str = typer.Option("", "--filter", help="Comma-separated CVE IDs to evaluate."),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip CVEs already present in results.json.",
    ),
    sok_path: str = typer.Option("SoK", "--sok-path", help="Source SoK repository path."),
    runtime_sok_dir: str = typer.Option("", "--runtime-sok-dir", help="Prepared runtime SoK path."),
    workspace_dir: str = typer.Option("", "--workspace-dir", help="Host workspace root for evaluation."),
    sanitize_benchmark: bool = typer.Option(
        True,
        "--sanitize-benchmark/--no-sanitize-benchmark",
        help="Copy only exploit files into runtime SoK snapshot.",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show tqdm progress bar during evaluation.",
    ),
) -> None:
    run_evaluation(
        preds_path=Path(preds),
        output_dir=Path(output_dir).resolve() if output_dir else None,
        workers=workers,
        docker_image=docker_image,
        stage_timeout=stage_timeout,
        container_timeout=container_timeout,
        old_stage_workers=old_stage_workers,
        filter_cve_ids=filter_cve_ids,
        skip_existing=skip_existing,
        sok_path=sok_path,
        runtime_sok_dir=Path(runtime_sok_dir).resolve() if runtime_sok_dir else None,
        workspace_root=Path(workspace_dir).resolve() if workspace_dir else None,
        sanitize_benchmark=sanitize_benchmark,
        show_progress=progress,
    )


if __name__ == "__main__":
    app()
