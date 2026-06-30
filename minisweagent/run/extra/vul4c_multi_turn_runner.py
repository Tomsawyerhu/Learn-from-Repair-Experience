#!/usr/bin/env python3
"""Multi-turn Vul4C runner (repair -> evaluation -> experience pipeline)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import typer
import yaml

from minisweagent import logger
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.run.extra.vul4c_evaluate import run_evaluation
from minisweagent.run.extra.vul4c_exp import RunMode
from minisweagent.run.extra.vul4c_exp import main as run_single_round

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPBANK_ROOT = PROJECT_ROOT / "expbank"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "minisweagent" / "output" / "vul4c"
DEFAULT_PIPELINE_CONFIG = EXPBANK_ROOT / "src" / "config" / "run" / "pipeline_config_vul4c.yaml"
STAGE_LOGGER_NAME = "minisweagent.vul4c_multi_turn.stage"


def _setup_stage_logger(log_path: Path) -> logging.Logger:
    stage_logger = logging.getLogger(STAGE_LOGGER_NAME)
    stage_logger.setLevel(logging.INFO)
    stage_logger.propagate = False

    for handler in list(stage_logger.handlers):
        stage_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stage_logger.addHandler(file_handler)
    print(f"Stage logging to '{log_path}'")
    return stage_logger


def _stage_log(stage_logger: logging.Logger, message: str, *, level: int = logging.INFO) -> None:
    logger.log(level, message)
    stage_logger.log(level, message)


def _sanitize_name(name: str | None) -> str:
    if not name:
        return "model"
    mapped = [ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip()]
    out = "".join(mapped).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "model"


def _resolve_model_name(model: str | None, config: dict[str, Any]) -> str:
    if model:
        return model
    configured = str(config.get("model", {}).get("model_name", "")).strip()
    if configured:
        return configured
    return "model"


def _default_output_dir(run_mode: RunMode, model_name: str, exp_retrieve_k: int, exp_return_n: int) -> Path:
    safe_model = _sanitize_name(model_name)
    if run_mode == RunMode.BASELINE:
        subdir = f"{safe_model}-baseline"
    else:
        subdir = f"{safe_model}-retrieve-k-{exp_retrieve_k}-{exp_return_n}"
    return DEFAULT_OUTPUT_BASE / subdir


def _load_dataset_cves(dataset_path: Path) -> list[str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cves: list[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                cve_id = str(value.get("cve_id") or key).strip()
            else:
                cve_id = str(key).strip()
            if cve_id:
                cves.append(cve_id)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            cve_id = str(item.get("cve_id", "")).strip()
            if cve_id:
                cves.append(cve_id)
    else:
        msg = f"Unsupported dataset format: {dataset_path}"
        raise ValueError(msg)

    return sorted(set(cves))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_summary_sets(summary_path: Path, fallback_input: list[str]) -> tuple[set[str], set[str]]:
    summary = _read_json(summary_path)

    success_raw = summary.get("successful_cves", [])
    failed_raw = summary.get("failed_cves", [])

    success_set = {str(x).strip() for x in success_raw if str(x).strip()}
    failed_set = {str(x).strip() for x in failed_raw if str(x).strip()}

    if not failed_set and fallback_input:
        failed_set = set(fallback_input) - success_set

    return success_set, failed_set


def _copy_db_for_retrieval(src_db: Path, dst_db: Path) -> Path:
    if not src_db.exists():
        msg = f"Experience DB does not exist: {src_db}"
        raise FileNotFoundError(msg)
    if dst_db.exists():
        shutil.rmtree(dst_db)
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_db, dst_db)
    return dst_db


def _run_pipeline_vul4c(
    *,
    pipeline_config: Path,
    repair_dir: Path,
    summary_path: Path,
    expbank_dir: Path,
    workers: int,
    max_files: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "src.pipeline_vul4c",
        "--config",
        str(pipeline_config),
        "--input-dir",
        str(repair_dir),
        "--summary-path",
        str(summary_path),
        "--output-dir",
        str(expbank_dir),
        "--stages",
        "extract,experience,index",
        "--workers",
        str(workers),
    ]
    if max_files > 0:
        cmd.extend(["--max-files", str(max_files)])

    proc = subprocess.run(
        cmd,
        cwd=str(EXPBANK_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = proc.stdout or ""
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-120:])
        msg = f"pipeline_vul4c failed (rc={proc.returncode})\n{tail}"
        raise RuntimeError(msg)

    stats: dict[str, Any] = {"raw_output_tail": "\n".join(output.splitlines()[-80:])}
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    stats.update(parsed)
            except Exception:  # noqa: BLE001
                pass
            break
    return stats


def _build_turn_meta(
    *,
    turn: int,
    run_mode: RunMode,
    input_cves: list[str],
    retrieval_db_src: Path | None,
    retrieval_db_used: Path | None,
    repair_dir: Path,
    evaluation_dir: Path,
    expbank_dir: Path,
    evaluation_summary: dict[str, Any],
    pipeline_stats: dict[str, Any],
    duration_seconds: float,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    success_list = evaluation_summary.get("successful_cves", [])
    failed_list = evaluation_summary.get("failed_cves", [])
    return {
        "turn": turn,
        "status": status,
        "run_mode": run_mode.value,
        "input_cves": input_cves,
        "input_count": len(input_cves),
        "retrieval_db_source": str(retrieval_db_src) if retrieval_db_src else "",
        "retrieval_db_used": str(retrieval_db_used) if retrieval_db_used else "",
        "paths": {
            "repair_dir": str(repair_dir),
            "evaluation_dir": str(evaluation_dir),
            "expbank_dir": str(expbank_dir),
        },
        "evaluation": {
            "total": int(evaluation_summary.get("total", 0)),
            "success_rate": float(evaluation_summary.get("success_rate", 0.0)),
            "successful_cves": success_list,
            "failed_cves": failed_list,
            "status_counts": evaluation_summary.get("status_counts", {}),
        },
        "pipeline": pipeline_stats,
        "duration_seconds": round(duration_seconds, 2),
        "error_message": error_message,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _resolve_resume_turn(
    output_root: Path,
    turns: int,
) -> int:
    for turn in range(1, turns + 1):
        meta_path = output_root / f"turn_{turn}" / "turn_meta.json"
        if not meta_path.exists():
            return turn
        meta = _read_json(meta_path)
        if str(meta.get("status", "")).lower() != "completed":
            return turn
    return turns + 1


@app.command(help="Run Vul4C multi-turn workflow: repair -> evaluation -> experience.")
def main(
    dataset: str = typer.Option("SoK/Vul4C_Src/Vul4C.json", "-d", "--dataset", help="Vul4C dataset path."),
    output: str = typer.Option("", "-o", "--output", help="Output root directory."),
    turns: int = typer.Option(3, "-n", "--turns", min=1, help="Number of turns."),
    run_mode: RunMode = typer.Option(
        RunMode.BASELINE,
        "--run-mode",
        help="baseline or retrieve-only (no hybrid mode).",
    ),
    config_spec: Path = typer.Option(
        builtin_config_dir / "extra" / "vul4c.yaml",
        "--config",
        "-C",
        help="Agent config path used by vul4c_exp.py.",
    ),
    model: str | None = typer.Option(None, "-m", "--model", help="Override model name."),
    model_class: str | None = typer.Option(None, "--model-class", help="Override model class."),
    workers: int = typer.Option(16, "--workers", help="Repair workers per turn."),
    eval_workers: int = typer.Option(8, "--eval-workers", help="Evaluation workers per turn."),
    eval_old_stage_workers: int = typer.Option(
        8,
        "--eval-old-stage-workers",
        help="Concurrency limit for compile_old/reproduce_old in evaluation.",
    ),
    eval_progress: bool = typer.Option(
        True,
        "--eval-progress/--no-eval-progress",
        help="Show tqdm progress bar in evaluation stage.",
    ),
    pipeline_workers: int = typer.Option(1, "--pipeline-workers", help="Workers for pipeline_vul4c."),
    pipeline_max_files: int = typer.Option(0, "--pipeline-max-files", help="Limit files for pipeline (0 = all)."),
    filter_cve_ids: str = typer.Option("", "--filter", "-f", help="Comma-separated CVE IDs to run."),
    max_instances: int = typer.Option(0, "--max-instances", help="Limit CVE count after filter (0 = all)."),
    carry_failed_only: bool = typer.Option(
        True,
        "--carry-failed-only/--no-carry-failed-only",
        help="Next turn runs only previous failed CVEs.",
    ),
    stop_when_no_failures: bool = typer.Option(
        True,
        "--stop-when-no-failures/--no-stop-when-no-failures",
        help="Stop early when failed set is empty.",
    ),
    redo_existing: bool = typer.Option(
        False,
        "--redo-existing/--no-redo-existing",
        help="Redo completed CVEs inside each single-turn run.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume from first incomplete turn if output already exists.",
    ),
    # Experience retrieval params (repair stage)
    exp_db_path: str = typer.Option(
        "",
        "--exp-db",
        help="Seed experience DB path for turn 1 in retrieve-only mode.",
    ),
    exp_model_path: str | None = typer.Option(None, "--exp-model", help="Embedding model path for retrieval."),
    exclude_current_cve: bool = typer.Option(
        True,
        "--exclude-current-cve/--no-exclude-current-cve",
        help="Exclude current CVE in retrieval.",
    ),
    exp_retrieve_k: int = typer.Option(10, "--exp-retrieve-k", help="Top-k candidates retrieved."),
    exp_return_n: int = typer.Option(5, "--exp-return-n", help="Returned experiences."),
    exp_sort_by_quality: bool = typer.Option(
        True,
        "--exp-sort-by-quality/--no-exp-sort-by-quality",
        help="Sort by quality score in retrieval.",
    ),
    temporal_filter: bool = typer.Option(
        False,
        "--temporal-filter/--no-temporal-filter",
        help="Only retrieve experiences from earlier CVEs.",
    ),
    # Docker/runtime params
    docker_image: str = typer.Option("vul4c/vul4c:3.0", "--docker-image", help="Docker image."),
    docker_timeout: int = typer.Option(600, "--docker-timeout", help="Single command timeout for repair."),
    checkout_timeout: int = typer.Option(900, "--checkout-timeout", help="Checkout timeout for repair."),
    stage_timeout: int = typer.Option(1200, "--stage-timeout", help="Per-stage timeout for evaluation."),
    container_timeout: str = typer.Option("6h", "--container-timeout", help="Container lifetime."),
    sok_path: str = typer.Option("SoK", "--sok-path", help="SoK source path."),
    sanitize_benchmark: bool = typer.Option(
        True,
        "--sanitize-benchmark/--no-sanitize-benchmark",
        help="Copy only exploit files into runtime SoK snapshot.",
    ),
    pipeline_config: Path = typer.Option(
        DEFAULT_PIPELINE_CONFIG,
        "--pipeline-config",
        help="pipeline_vul4c config path.",
    ),
) -> None:
    config_path = get_config_path(config_spec)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_name = _resolve_model_name(model, config)
    output_root = Path(output).resolve() if output else _default_output_dir(run_mode, model_name, exp_retrieve_k, exp_return_n)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_logger = _setup_stage_logger(output_root / "multi_turn_vul4c.log")
    _stage_log(stage_logger, f"Multi-turn output root: {output_root}")

    dataset_path = Path(dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    pipeline_config = Path(pipeline_config).resolve()
    if not pipeline_config.exists():
        raise FileNotFoundError(f"Pipeline config not found: {pipeline_config}")

    target_cves = _load_dataset_cves(dataset_path)
    if filter_cve_ids:
        selected = {x.strip() for x in filter_cve_ids.split(",") if x.strip()}
        target_cves = [cve for cve in target_cves if cve in selected]
    if max_instances > 0:
        target_cves = target_cves[:max_instances]
    if not target_cves:
        raise ValueError("No CVEs selected after filtering.")

    seed_exp_db = Path(exp_db_path).resolve() if exp_db_path else None
    if run_mode == RunMode.RETRIEVE_ONLY and not seed_exp_db and not resume:
        raise ValueError("--exp-db is required for retrieve-only when resume is disabled.")

    run_meta_path = output_root / "run_meta.json"
    run_meta = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(dataset_path),
        "run_mode": run_mode.value,
        "turns_requested": turns,
        "target_cves": target_cves,
        "config": {
            "model_name": model_name,
            "model_class": model_class or config.get("model", {}).get("model_class", ""),
            "workers": workers,
            "eval_workers": eval_workers,
            "eval_old_stage_workers": eval_old_stage_workers,
            "eval_progress": eval_progress,
            "pipeline_workers": pipeline_workers,
            "pipeline_max_files": pipeline_max_files,
            "carry_failed_only": carry_failed_only,
            "stop_when_no_failures": stop_when_no_failures,
            "docker_image": docker_image,
            "docker_timeout": docker_timeout,
            "checkout_timeout": checkout_timeout,
            "stage_timeout": stage_timeout,
            "container_timeout": container_timeout,
            "sok_path": str(Path(sok_path).resolve()),
            "sanitize_benchmark": sanitize_benchmark,
            "pipeline_config": str(pipeline_config),
            "exp_db_seed": str(seed_exp_db) if seed_exp_db else "",
            "exp_model_path": exp_model_path or "",
            "exp_retrieve_k": exp_retrieve_k,
            "exp_return_n": exp_return_n,
            "exclude_current_cve": exclude_current_cve,
            "exp_sort_by_quality": exp_sort_by_quality,
            "temporal_filter": temporal_filter,
        },
    }
    _write_json(run_meta_path, run_meta)

    turn_start = 1
    if resume:
        turn_start = _resolve_resume_turn(output_root, turns)
        _stage_log(stage_logger, f"Resume enabled, starting from turn {turn_start}")

    cumulative_success: set[str] = set()
    first_success_turn: dict[str, int] = {}
    prev_failed: set[str] = set(target_cves)

    # Recover status from completed turns if resume.
    if turn_start > 1:
        for turn in range(1, turn_start):
            summary_path = output_root / f"turn_{turn}" / "evaluation" / "summary.json"
            success_set, failed_set = _load_summary_sets(summary_path, sorted(prev_failed))
            for cve in sorted(success_set):
                if cve not in first_success_turn:
                    first_success_turn[cve] = turn
            cumulative_success.update(success_set)
            prev_failed = set(target_cves) - cumulative_success if not carry_failed_only else failed_set

    actual_turns = 0
    for turn in range(turn_start, turns + 1):
        turn_dir = output_root / f"turn_{turn}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        turn_meta_path = turn_dir / "turn_meta.json"

        input_cves = sorted(prev_failed if carry_failed_only else set(target_cves))
        if not input_cves:
            _stage_log(stage_logger, f"Turn {turn}: no CVEs to run, stop.")
            break

        _stage_log(stage_logger, f"========== Turn {turn}/{turns} ==========")
        _stage_log(stage_logger, f"Input CVEs: {len(input_cves)}")
        turn_start_time = time.time()

        repair_dir = turn_dir / "repair"
        eval_dir = turn_dir / "evaluation"
        expbank_dir = turn_dir / "expbank"
        retrieval_db_used: Path | None = None
        retrieval_db_source: Path | None = None

        try:
            if run_mode == RunMode.RETRIEVE_ONLY:
                if turn == 1:
                    retrieval_db_source = seed_exp_db
                else:
                    prev_vec_db = output_root / f"turn_{turn - 1}" / "expbank" / "vec_DB"
                    retrieval_db_source = prev_vec_db if prev_vec_db.exists() else None
                    if retrieval_db_source is None and seed_exp_db:
                        retrieval_db_source = seed_exp_db

                if retrieval_db_source is None:
                    raise ValueError(
                        f"Turn {turn} retrieve-only requires experience DB, "
                        "but no seed DB and no previous turn vec_DB were found."
                    )

                retrieval_db_used = _copy_db_for_retrieval(
                    retrieval_db_source,
                    turn_dir / "retrieval_db" / "vec_DB",
                )
                _stage_log(stage_logger, f"Turn {turn} retrieval DB: {retrieval_db_used}")

            # Stage 1: repair
            _stage_log(stage_logger, f"Turn {turn} stage repair started.")
            run_single_round(
                dataset=str(dataset_path),
                output=str(repair_dir),
                workers=workers,
                model=model,
                model_class=model_class,
                redo_existing=redo_existing,
                config_spec=config_spec,
                filter_cve_ids=",".join(input_cves),
                max_instances=0,
                run_mode=run_mode,
                exp_db_path=str(retrieval_db_used) if retrieval_db_used else None,
                exp_model_path=exp_model_path,
                exclude_current_cve=exclude_current_cve,
                exp_retrieve_k=exp_retrieve_k,
                exp_return_n=exp_return_n,
                exp_sort_by_quality=exp_sort_by_quality,
                temporal_filter=temporal_filter,
                docker_image=docker_image,
                docker_timeout=docker_timeout,
                container_timeout=container_timeout,
                checkout_timeout=checkout_timeout,
                sok_path=sok_path,
                runtime_sok_dir="",
                workspace_dir="",
                vul4c_data_dir="",
                sanitize_benchmark=sanitize_benchmark,
                auto_eval=False,
                eval_workers=eval_workers,
                eval_stage_timeout=stage_timeout,
                eval_progress=eval_progress,
                eval_skip_existing=False,
            )
            _stage_log(stage_logger, f"Turn {turn} stage repair completed.")

            preds_path = repair_dir / "preds.json"
            if not preds_path.exists():
                raise FileNotFoundError(f"preds.json missing after repair stage: {preds_path}")

            # Stage 2: evaluation
            _stage_log(stage_logger, f"Turn {turn} stage evaluation started.")
            eval_stats = run_evaluation(
                preds_path=preds_path,
                output_dir=eval_dir,
                workers=eval_workers,
                docker_image=docker_image,
                stage_timeout=stage_timeout,
                container_timeout=container_timeout,
                old_stage_workers=eval_old_stage_workers,
                filter_cve_ids=",".join(input_cves),
                skip_existing=False,
                sok_path=sok_path,
                runtime_sok_dir=None,
                workspace_root=None,
                sanitize_benchmark=sanitize_benchmark,
                show_progress=eval_progress,
            )
            _ = eval_stats
            _stage_log(stage_logger, f"Turn {turn} stage evaluation completed.")

            summary_path = eval_dir / "summary.json"
            evaluation_summary = _read_json(summary_path)

            # Stage 3: experience pipeline
            _stage_log(stage_logger, f"Turn {turn} stage experience pipeline started.")
            pipeline_stats = _run_pipeline_vul4c(
                pipeline_config=pipeline_config,
                repair_dir=repair_dir,
                summary_path=summary_path,
                expbank_dir=expbank_dir,
                workers=pipeline_workers,
                max_files=pipeline_max_files,
            )
            _stage_log(stage_logger, f"Turn {turn} stage experience pipeline completed.")

            success_set, failed_set = _load_summary_sets(summary_path, input_cves)
            for cve in sorted(success_set):
                if cve not in first_success_turn:
                    first_success_turn[cve] = turn
            cumulative_success.update(success_set)
            prev_failed = failed_set if carry_failed_only else set(target_cves) - cumulative_success

            turn_meta = _build_turn_meta(
                turn=turn,
                run_mode=run_mode,
                input_cves=input_cves,
                retrieval_db_src=retrieval_db_source,
                retrieval_db_used=retrieval_db_used,
                repair_dir=repair_dir,
                evaluation_dir=eval_dir,
                expbank_dir=expbank_dir,
                evaluation_summary=evaluation_summary,
                pipeline_stats=pipeline_stats,
                duration_seconds=time.time() - turn_start_time,
                status="completed",
            )
            _write_json(turn_meta_path, turn_meta)
            actual_turns += 1

            _stage_log(
                stage_logger,
                f"Turn {turn} completed: success={len(success_set)}, fail={len(failed_set)}, "
                f"cumulative_success={len(cumulative_success)}",
            )
            if stop_when_no_failures and not failed_set:
                _stage_log(stage_logger, f"Turn {turn}: failed set is empty, stop early.")
                break

        except Exception as exc:  # noqa: BLE001
            turn_meta = _build_turn_meta(
                turn=turn,
                run_mode=run_mode,
                input_cves=input_cves,
                retrieval_db_src=retrieval_db_source,
                retrieval_db_used=retrieval_db_used,
                repair_dir=repair_dir,
                evaluation_dir=eval_dir,
                expbank_dir=expbank_dir,
                evaluation_summary={},
                pipeline_stats={},
                duration_seconds=time.time() - turn_start_time,
                status="failed",
                error_message=str(exc),
            )
            _write_json(turn_meta_path, turn_meta)
            _stage_log(stage_logger, f"Turn {turn} failed: {exc}", level=logging.ERROR)
            raise

    completed_turns = 0
    for turn in range(1, turns + 1):
        meta_path = output_root / f"turn_{turn}" / "turn_meta.json"
        meta = _read_json(meta_path)
        if str(meta.get("status", "")).lower() == "completed":
            completed_turns += 1

    final_cve_status: dict[str, dict[str, Any]] = {}
    for cve in target_cves:
        if cve in cumulative_success:
            final_cve_status[cve] = {
                "status": "success",
                "first_success_turn": first_success_turn.get(cve),
            }
        else:
            final_cve_status[cve] = {
                "status": "failed",
                "first_success_turn": None,
            }

    final_summary = {
        "run_mode": run_mode.value,
        "turns_requested": turns,
        "turns_completed": completed_turns,
        "turns_completed_this_run": actual_turns,
        "target_total": len(target_cves),
        "final_success_total": len(cumulative_success),
        "final_failure_total": len(target_cves) - len(cumulative_success),
        "final_success_rate": round(100.0 * len(cumulative_success) / max(len(target_cves), 1), 2),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(output_root / "final_cve_status.json", final_cve_status)
    _write_json(output_root / "final_summary.json", final_summary)

    run_meta["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_meta["turns_completed"] = completed_turns
    run_meta["turns_completed_this_run"] = actual_turns
    run_meta["final_summary"] = final_summary
    _write_json(run_meta_path, run_meta)

    _stage_log(stage_logger, "Multi-turn Vul4C run completed.")
    _stage_log(
        stage_logger,
        "Final summary: "
        f"success={final_summary['final_success_total']}/{final_summary['target_total']} "
        f"({final_summary['final_success_rate']}%)",
    )


if __name__ == "__main__":
    app()
