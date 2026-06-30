#!/usr/bin/env python3

"""Redo error CVEs from patcheval_exp output directory.

This script rescans an existing output directory (e.g., turn_1) to find CVEs with error status,
re-processes them, and updates the output files accordingly.
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from jsonlines import jsonlines
from rich.live import Live

# Reuse components from patcheval_exp
from minisweagent.run.extra.patcheval_exp import (
    PROJECT_ROOT,
    EXPBANK_ROOT,
    PATCHEVAL_ROOT,
    RunMode,
    ProgressTrackingAgent,
    read_patcheval_dataset,
    process_instance,
    run_post_evaluation,
    validate_config,
    update_patches_file,
    update_preds_file,
    remove_from_preds_file,
    load_successful_cves,
    escape_rich_markup,
    RunStatistics,  # Statistics collection
)
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.utils.log import add_file_handler, logger

# ExpBank imports
sys.path.append(str(EXPBANK_ROOT))
from src.rag.experience_retriever import ExperienceRetriever

_HELP_TEXT = """Redo error CVEs from an existing patcheval_exp output directory.

This script:
1. Scans {target_dir}/exit_statuses.yaml to find error CVEs
2. Re-processes those CVEs, overwriting existing trajectories
3. Updates exit_statuses.yaml with merged results (overwrites original)
4. Updates patches.json and preds.json
5. Optionally re-runs evaluation with --auto-eval

Usage:
  python patcheval_exp_redo.py -d dataset.jsonl -o turn_1 [options]

The -o parameter specifies the target directory directly (not a parent directory).
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def load_error_cves(exit_statuses_path: Path) -> list[str]:
    """Load CVE IDs with error status from exit_statuses.yaml.
    
    Args:
        exit_statuses_path: Path to exit_statuses.yaml
        
    Returns:
        List of CVE IDs that have error status
    """
    if not exit_statuses_path.exists():
        logger.error(f"exit_statuses.yaml not found: {exit_statuses_path}")
        return []
    
    data = yaml.safe_load(exit_statuses_path.read_text())
    instances_by_status = data.get("instances_by_exit_status", {})
    
    error_cves = []
    for status, cve_list in instances_by_status.items():
        # Error or Uncaught* status are considered errors
        if status == "Error" or status.startswith("Uncaught"):
            error_cves.extend(cve_list)
            logger.info(f"Found {len(cve_list)} CVEs with status '{status}'")
    
    return error_cves


def merge_exit_statuses(original_path: Path, redo_results: dict) -> dict:
    """Merge original exit statuses with redo results.
    
    Args:
        original_path: Path to original exit_statuses.yaml
        redo_results: Dict of {status: [cve_ids]} from redo run
        
    Returns:
        Merged dict with all CVEs and their latest status
    """
    # Load original data
    original_data = {}
    if original_path.exists():
        original_data = yaml.safe_load(original_path.read_text()) or {}
    
    original_statuses = original_data.get("instances_by_exit_status", {})
    
    # Get all CVEs that were redone
    redo_cves = set()
    for cve_list in redo_results.values():
        redo_cves.update(cve_list)
    
    # Create merged result
    merged = {"instances_by_exit_status": {}}
    
    # Copy original statuses, excluding redone CVEs
    for status, cve_list in original_statuses.items():
        remaining = [cve for cve in cve_list if cve not in redo_cves]
        if remaining:
            merged["instances_by_exit_status"][status] = remaining
    
    # Add redo results
    for status, cve_list in redo_results.items():
        if status in merged["instances_by_exit_status"]:
            merged["instances_by_exit_status"][status].extend(cve_list)
        else:
            merged["instances_by_exit_status"][status] = list(cve_list)
    
    return merged


def load_run_config(output_path: Path) -> dict:
    """Load run configuration from run_config.json if it exists.
    
    Args:
        output_path: Path to the turn directory
        
    Returns:
        Dict with configuration values, or empty dict if file doesn't exist
    """
    run_config_file = output_path / "run_config.json"
    if not run_config_file.exists():
        return {}
    
    try:
        config_data = json.loads(run_config_file.read_text())
        logger.info(f"Loaded run configuration from {run_config_file}")
        return config_data
    except Exception as e:
        logger.warning(f"Failed to load run_config.json: {e}")
        return {}


# Default sentinel values for detecting user-provided arguments
_DEFAULT_WORKERS = 1
_DEFAULT_DOCKER_TIMEOUT = 600
_DEFAULT_EXP_RETRIEVE_K = 3
_DEFAULT_EXP_RETURN_N = 3
_DEFAULT_EVAL_WORKERS = 4
_DEFAULT_CONFIG_SPEC = builtin_config_dir / "extra" / "patcheval_exp.yaml"


@app.command(help=_HELP_TEXT)
def main(
        dataset: str = typer.Option("", "-d", "--dataset", help="Dataset File", rich_help_panel="Basic"),
        output: str = typer.Option("", "-o", "--output", help="Target output directory (e.g., turn_1) to redo errors from", rich_help_panel="Basic"),
        workers: int = typer.Option(_DEFAULT_WORKERS, "-w", "--workers", help="Number of worker threads for parallel processing",
                                    rich_help_panel="Basic"),
        model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
        model_class: str | None = typer.Option(None, "--model-class",
                                               help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')",
                                               rich_help_panel="Advanced"),
        redo_existing: bool = typer.Option(True, "--redo-existing", help="Redo existing instances",
                                           rich_help_panel="Data selection"),
        config_spec: Path = typer.Option(_DEFAULT_CONFIG_SPEC, "-c", "--config",
                                         help="Path to a config file", rich_help_panel="Basic"),
        summary_path: list[Path] = typer.Option([], "--summary-path", help="Path(s) to summary.json from previous turns to filter out successful instances",
                                         rich_help_panel="Data selection"),
        run_mode: RunMode | None = typer.Option(None, "--mode", help="Run mode: baseline (no exp), retrieve-only (inject but no update)",
                                        rich_help_panel="Experience Enhancement"),
        exp_db_path: str = typer.Option(None, "--exp-db", help="ExpBank vector database path (default: <PROJECT_ROOT>/expbank/vec_DB)",
                                       rich_help_panel="Experience Enhancement"),
        exp_model_path: str = typer.Option(None, "--exp-model", help="BGE model path (default: <PROJECT_ROOT>/expbank/models/bge-large-en-v1.5)",
                                          rich_help_panel="Experience Enhancement"),
        exclude_current_cve: bool | None = typer.Option(None, "--exclude-current-cve/--no-exclude-current-cve",
                                                 help="Whether to exclude current CVE's experiences from retrieval (default: True)",
                                                 rich_help_panel="Experience Enhancement"),
        max_instances: int = typer.Option(None, "--max-instances", help="Maximum number of error instances to redo (default: all)", rich_help_panel="Data selection"),
        auto_eval: bool | None = typer.Option(None, "--auto-eval/--no-auto-eval",
                                       help="Automatically run patch collection and evaluation after processing (default: False)",
                                       rich_help_panel="Post-processing"),
        eval_workers: int = typer.Option(_DEFAULT_EVAL_WORKERS, "--eval-workers", help="Number of workers for evaluation (default: 4)",
                                        rich_help_panel="Post-processing"),
        docker_timeout: int = typer.Option(_DEFAULT_DOCKER_TIMEOUT, "--docker-timeout", help="Docker command timeout in seconds (default: 600)",
                                           rich_help_panel="Basic"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration and show error CVEs without running (default: False)",
                                    rich_help_panel="Advanced"),
        temporal_filter: bool | None = typer.Option(None, "--temporal-filter/--no-temporal-filter",
                                             help="Only use experiences from earlier CVEs (by CVE ID year/number, default: False)",
                                             rich_help_panel="Experience Enhancement"),
        exp_retrieve_k: int = typer.Option(_DEFAULT_EXP_RETRIEVE_K, "--exp-retrieve-k",
                                           help="Number of top similar experiences to retrieve first (default: 3)",
                                           rich_help_panel="Experience Enhancement"),
        exp_return_n: int = typer.Option(_DEFAULT_EXP_RETURN_N, "--exp-return-n",
                                         help="Number of experiences to return after quality sorting (default: 3)",
                                         rich_help_panel="Experience Enhancement"),
        exp_sort_by_quality: bool | None = typer.Option(None, "--exp-sort-by-quality/--no-exp-sort-by-quality",
                                                 help="Sort retrieved experiences by quality score (default: True)",
                                                 rich_help_panel="Experience Enhancement"),
) -> None:
    """Redo error CVEs from an existing output directory."""

    # Target directory is directly the -o path (no turn_N subdirectory creation)
    output_path = Path(output)

    # ========== AUTO-LOAD RUN CONFIG ==========
    # Try to load configuration from existing run_config.json
    saved_config = load_run_config(output_path)

    # Use saved config as defaults if command-line args are not provided
    # For optional parameters (None means not set), we can safely use saved config
    # For parameters with default values, we check against their sentinel defaults
    if saved_config:
        basic = saved_config.get("basic", {})
        exp_enh = saved_config.get("experience_enhancement", {})
        post_proc = saved_config.get("post_processing", {})
        data_sel = saved_config.get("data_selection", {})

        # ===== BASIC SETTINGS =====
        if dataset == "" and basic.get("dataset"):
            dataset = basic["dataset"]
            logger.info(f"Using dataset from config: {dataset}")

        if model is None and basic.get("model"):
            model = basic["model"]
            logger.info(f"Using model from config: {model}")

        if model_class is None and basic.get("model_class"):
            model_class = basic["model_class"]
            logger.info(f"Using model_class from config: {model_class}")

        if workers == _DEFAULT_WORKERS and basic.get("workers"):
            workers = basic["workers"]
            logger.info(f"Using workers from config: {workers}")

        if docker_timeout == _DEFAULT_DOCKER_TIMEOUT and basic.get("docker_timeout"):
            docker_timeout = basic["docker_timeout"]
            logger.info(f"Using docker_timeout from config: {docker_timeout}")

        if config_spec == _DEFAULT_CONFIG_SPEC and basic.get("config_file"):
            config_spec = Path(basic["config_file"])
            logger.info(f"Using config_file from config: {config_spec}")

        # ===== EXPERIENCE ENHANCEMENT SETTINGS =====
        # Use None as sentinel for optional bool/enum parameters
        if run_mode is None and exp_enh.get("run_mode"):
            run_mode = RunMode(exp_enh["run_mode"])
            logger.info(f"Using run_mode from config: {run_mode.value}")

        if exp_db_path is None and exp_enh.get("exp_db_path"):
            exp_db_path = exp_enh["exp_db_path"]
            logger.info(f"Using exp_db_path from config: {exp_db_path}")

        if exp_model_path is None and exp_enh.get("exp_model_path"):
            exp_model_path = exp_enh["exp_model_path"]
            logger.info(f"Using exp_model_path from config: {exp_model_path}")

        if exclude_current_cve is None and "exclude_current_cve" in exp_enh:
            exclude_current_cve = exp_enh["exclude_current_cve"]
            logger.info(f"Using exclude_current_cve from config: {exclude_current_cve}")

        if temporal_filter is None and "temporal_filter" in exp_enh:
            temporal_filter = exp_enh["temporal_filter"]
            logger.info(f"Using temporal_filter from config: {temporal_filter}")

        if exp_retrieve_k == _DEFAULT_EXP_RETRIEVE_K and exp_enh.get("exp_retrieve_k"):
            exp_retrieve_k = exp_enh["exp_retrieve_k"]
            logger.info(f"Using exp_retrieve_k from config: {exp_retrieve_k}")

        if exp_return_n == _DEFAULT_EXP_RETURN_N and exp_enh.get("exp_return_n"):
            exp_return_n = exp_enh["exp_return_n"]
            logger.info(f"Using exp_return_n from config: {exp_return_n}")

        if exp_sort_by_quality is None and "exp_sort_by_quality" in exp_enh:
            exp_sort_by_quality = exp_enh["exp_sort_by_quality"]
            logger.info(f"Using exp_sort_by_quality from config: {exp_sort_by_quality}")

        # ===== DATA SELECTION SETTINGS =====
        if not summary_path and data_sel.get("summary_paths"):
            summary_path = [Path(p) for p in data_sel["summary_paths"]]
            logger.info(f"Using summary_paths from config: {summary_path}")

        # ===== POST-PROCESSING SETTINGS =====
        if auto_eval is None and "auto_eval" in post_proc:
            auto_eval = post_proc["auto_eval"]
            logger.info(f"Using auto_eval from config: {auto_eval}")

        if eval_workers == _DEFAULT_EVAL_WORKERS and post_proc.get("eval_workers"):
            eval_workers = post_proc["eval_workers"]
            logger.info(f"Using eval_workers from config: {eval_workers}")

    # ========== APPLY DEFAULTS FOR UNSET OPTIONAL PARAMETERS ==========
    # Set defaults for parameters that were None (not provided by user or config)
    if run_mode is None:
        run_mode = RunMode.RETRIEVE_ONLY
    if exclude_current_cve is None:
        exclude_current_cve = True
    if temporal_filter is None:
        temporal_filter = False
    if exp_sort_by_quality is None:
        exp_sort_by_quality = True
    if auto_eval is None:
        auto_eval = False
    
    
    if not output_path.exists():
        logger.error(f"Target directory does not exist: {output_path}")
        logger.error("The -o parameter should specify an existing turn directory (e.g., turn_1)")
        raise typer.Exit(code=1)
    
    # Check for exit_statuses.yaml
    exit_statuses_file = output_path / "exit_statuses.yaml"
    if not exit_statuses_file.exists():
        logger.error(f"exit_statuses.yaml not found in {output_path}")
        raise typer.Exit(code=1)
    
    # Load error CVEs
    error_cves = load_error_cves(exit_statuses_file)
    if not error_cves:
        logger.info("No error CVEs found. Nothing to redo.")
        return
    
    logger.info(f"Found {len(error_cves)} error CVE(s) to redo:")
    for cve in error_cves[:10]:  # Show first 10
        logger.info(f"  - {cve}")
    if len(error_cves) > 10:
        logger.info(f"  ... and {len(error_cves) - 10} more")
    
    # Auto-default summary_path to parent directory (base experiment dir) if not specified
    if not summary_path and output_path.parent.exists():
        summary_path = [output_path.parent]
        logger.info(f"Auto-defaulting summary path to parent directory: {output_path.parent}")
    
    # Resolve config path
    config_path = get_config_path(config_spec)
    
    # DRY RUN: Show error CVEs and validate config
    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE: Validating Configuration for Redo")
        logger.info("=" * 60)
        logger.info(f"Target directory: {output_path}")
        logger.info(f"Error CVEs to redo: {len(error_cves)}")
        for cve in error_cves:
            logger.info(f"  - {cve}")
        validate_config(dataset, output_path, run_mode, exp_db_path, exp_model_path, auto_eval, summary_path, config_path)
        logger.info("=" * 60)
        logger.info("Configuration validation passed!")
        logger.info("=" * 60)
        return
    
    # Add file handler for logging
    log_file = output_path / "patcheval_exp_redo.log"
    if log_file.exists():
        os.remove(log_file)
    add_file_handler(log_file)
    logger.info(f"Redo logging to {log_file}")
    logger.info(f"Results will be saved/updated in {output_path}")
    
    # Log run mode
    logger.info(f"Run Mode: {run_mode.value.upper()}")
    if run_mode == RunMode.BASELINE:
        logger.info("  - Experience injection: DISABLED")
    elif run_mode == RunMode.RETRIEVE_ONLY:
        logger.info("  - Experience injection: ENABLED")
    
    # Resolve ExpBank paths
    if exp_db_path is None:
        exp_db_path = str(EXPBANK_ROOT / "vec_DB")
    else:
        exp_db_path = str(Path(exp_db_path).resolve())
    
    if exp_model_path is None:
        exp_model_path = str(EXPBANK_ROOT / "models" / "bge-large-en-v1.5")
    else:
        exp_model_path = str(Path(exp_model_path).resolve())
    
    if run_mode != RunMode.BASELINE:
        logger.info(f"ExpBank paths resolved:")
        logger.info(f"  - Database: {exp_db_path}")
        logger.info(f"  - BGE Model: {exp_model_path}")
    
    # Initialize retriever
    retriever = None
    if run_mode != RunMode.BASELINE:
        logger.info("Initializing ExpBank retriever...")
        retriever = ExperienceRetriever(
            db_path=exp_db_path,
            model_path=exp_model_path
        )
    
    # Load dataset and filter to error CVEs only
    logger.info(f"Loading dataset {dataset}")
    all_instances = read_patcheval_dataset(dataset)
    logger.info(f"Dataset contains {len(all_instances)} instances total")
    
    # Filter to only error CVEs
    error_cve_set = set(error_cves)
    instances = [inst for inst in all_instances if inst["cve_id"] in error_cve_set]
    logger.info(f"Filtered to {len(instances)} error instances for redo")
    
    # Apply max instances limit
    if max_instances is not None and max_instances > 0:
        logger.info(f"Limiting to {max_instances} instances")
        instances = instances[:max_instances]
    
    if not instances:
        logger.warning("No matching instances found in dataset for error CVEs")
        return
    
    logger.info(f"Will redo {len(instances)} instance(s)...")
    
    # Load config
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    
    config["exclude_current_cve"] = exclude_current_cve
    config["docker_timeout"] = docker_timeout
    logger.info(f"Exclude current CVE: {exclude_current_cve}")
    logger.info(f"Docker timeout: {docker_timeout} seconds")
    logger.info(f"Temporal filter: {temporal_filter}")
    
    # Store experience retrieval parameters
    config["exp_retrieve_k"] = exp_retrieve_k
    config["exp_return_n"] = exp_return_n
    config["exp_sort_by_quality"] = exp_sort_by_quality
    logger.info(f"Experience retrieval: retrieve_k={exp_retrieve_k}, return_n={exp_return_n}, sort_by_quality={exp_sort_by_quality}")
    
    # Create temporary exit status file for this redo run
    redo_exit_statuses_temp = output_path / "exit_statuses_redo_temp.yaml"
    progress_manager = RunBatchProgressManager(len(instances), redo_exit_statuses_temp)
    
    # Initialize run statistics collector
    run_statistics = RunStatistics() if run_mode != RunMode.BASELINE else None
    
    with Live(progress_manager.render_group, refresh_per_second=4):
        if workers <= 1:
            for instance in instances:
                try:
                    process_instance(instance, output_path, config, progress_manager, retriever, run_mode, run_statistics=run_statistics, temporal_filter=temporal_filter)
                except Exception as e:
                    logger.error(f"Fatal error processing {instance.get('cve_id')}: {e}", exc_info=True)
                    continue
        else:
            logger.info(f"Starting parallel processing with {workers} workers")
            
            def worker_fn(instance):
                try:
                    process_instance(instance, output_path, config, progress_manager, retriever, run_mode, run_statistics=run_statistics, temporal_filter=temporal_filter)
                    return instance.get('cve_id'), None
                except Exception as e:
                    logger.error(f"Fatal error processing {instance.get('cve_id')}: {e}", exc_info=True)
                    return instance.get('cve_id'), str(e)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(worker_fn, inst): inst for inst in instances}
                for future in concurrent.futures.as_completed(futures):
                    cve_id, error = future.result()
                    if error:
                        logger.warning(f"Instance {cve_id} completed with error: {error}")
    
    # Merge exit statuses and create exit_statuses_redo.yaml
    logger.info("Merging exit statuses...")
    
    # Load redo results from temp file
    redo_results = {}
    if redo_exit_statuses_temp.exists():
        redo_data = yaml.safe_load(redo_exit_statuses_temp.read_text()) or {}
        redo_results = redo_data.get("instances_by_exit_status", {})
    
    # Merge with original
    merged = merge_exit_statuses(exit_statuses_file, redo_results)
    
    # Save merged results back to original file
    exit_statuses_file.write_text(yaml.dump(merged, indent=4))
    logger.info(f"Updated exit statuses and merged into: {exit_statuses_file}")
    
    # Log summary
    logger.info("=" * 60)
    logger.info("REDO SUMMARY")
    logger.info("=" * 60)

    for status, cves in redo_results.items():
        logger.info(f"  {status}: {len(cves)}")
    
    # Cleanup temp file
    if redo_exit_statuses_temp.exists():
        redo_exit_statuses_temp.unlink()
    
    # ========== SAVE RUN STATISTICS ==========
    if run_statistics is not None and len(run_statistics) > 0:
        stats_file = run_statistics.save(output_path)
        logger.info(f"📊 Redo run statistics saved to: {stats_file}")
        logger.info(f"   Total CVEs with experiences: {len(run_statistics)}")
        summary = run_statistics.compute_summary()
        logger.info(f"   Avg success exp: {summary['avg_success_exp']:.2f}, Avg failure exp: {summary['avg_failure_exp']:.2f}")
        logger.info(f"   Avg quality score: {summary['avg_quality_score']:.2f}, Avg similarity: {summary['avg_similarity_score']:.4f}")
    
    # Post-processing: evaluation
    if auto_eval:
        run_post_evaluation(output_path, dataset, eval_workers)


if __name__ == "__main__":
    app()
