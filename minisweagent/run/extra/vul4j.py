import csv
import copy
import json
import sys
import threading
import time
from enum import Enum
from pathlib import Path

import typer
import traceback
import concurrent.futures
import yaml
from rich.live import Live

from minisweagent import logger
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler

from minisweagent.environments import get_environment

# ========== ExpBank experience repository configuration ==========
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # vulagent-main/
EXPBANK_ROOT = PROJECT_ROOT / "expbank"


def _sanitize_model_name(model_name: str | None) -> str:
    if not model_name:
        return "model"
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name)
    return cleaned or "model"


def _default_output_path(model_name: str | None, retrieve_k: int, return_n: int) -> Path:
    base = PROJECT_ROOT / "minisweagent" / "output" / "vul4j"
    return base / f"{_sanitize_model_name(model_name)}-k-{retrieve_k}-n-{return_n}"


# ExpBank imports
sys.path.append(str(EXPBANK_ROOT))
try:
    from src.rag.experience_retriever import ExperienceRetriever

    EXPBANK_AVAILABLE = True
except ImportError:
    logger.warning(f"ExpBank not found at {EXPBANK_ROOT}, experience enhancement will be disabled")
    ExperienceRetriever = None
    EXPBANK_AVAILABLE = False


class RunMode(str, Enum):
    """Run mode for experience enhancement."""

    BASELINE = "baseline"  # No experience injection
    RETRIEVE_ONLY = "retrieve-only"  # Inject but don't update


_HELP_TEXT = """Run mini-SWE-agent on VUL4J instances.

Run Modes:
  - baseline: No experience injection (same as original)
  - retrieve-only: Inject experiences but don't update experience bank
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    retriever=None,
    run_mode: RunMode = RunMode.BASELINE,
    temporal_filter: bool = False,
) -> None:
    """Process a single VUL4J instance with optional experience enhancement."""
    instance_id = instance["vul_id"]
    # Store trajectories under the raw_traj directory
    raw_traj_dir = output_dir / "raw_traj" / instance_id
    raw_traj_dir.mkdir(parents=True, exist_ok=True)

    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (raw_traj_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    CVE_ID = instance["cve_id"]
    CWE_ID = instance["cwe_id"]
    CVE_DESCRIPTION = instance["cve_description"]
    CWE_DESCRIPTION = instance["cwe_description_all"]

    progress_manager.on_instance_start(instance_id)

    # ========== EXPERIENCE RETRIEVAL ==========
    experience_context = ""
    if run_mode != RunMode.BASELINE and retriever is not None:
        try:
            # Build the query text: CVE description + CWE description
            query_text = CVE_DESCRIPTION
            if CWE_DESCRIPTION:
                query_text += f"\n{CWE_DESCRIPTION}"

            logger.info(f"🔍 [{run_mode.value.upper()}] EXPERIENCE RETRIEVAL for {instance_id}")

            exclude_current_cve = config.get("exclude_current_cve", True)
            retrieve_k = config.get("exp_retrieve_k", 3)
            return_n = config.get("exp_return_n", 3)
            sort_by_quality = config.get("exp_sort_by_quality", True)

            # Retrieve experiences
            experiences = retriever.retrieve(
                cve_description=query_text,
                exclude_cve_id=instance_id if exclude_current_cve else None,
                k=return_n,
                min_quality_score=0,
                retrieve_top_k=retrieve_k if not exclude_current_cve else max(retrieve_k, return_n * 3),
                sort_by_quality=sort_by_quality,
            )

            logger.info(f"   Retrieved {len(experiences)} experience(s) for {instance_id}")

            # Format experiences
            if experiences:
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
                logger.info(f"   Formatted experience context length: {len(experience_context)} chars")

        except Exception as e:
            logger.error(f"❌ Error retrieving experiences for {instance_id}: {e}")
            experience_context = ""
    else:
        logger.info(f"⏭️  [{run_mode.value.upper()}] Skipping experience retrieval")

    agent = None
    extra_info = None
    env = None
    exit_status = None
    result = None

    try:
        base_env_config = copy.deepcopy(config.get("environment", {}))
        image_name = "bqcuongas/vul4j"
        env_config = {
            **base_env_config,
            "environment_class": "docker",
            "image": image_name,
            "cwd": f"/tmp/vul4j/{instance_id}",
            "timeout": 240,
        }
        env = get_environment(env_config)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        # Clear possible leftovers
        env.execute(f"rm -rf /tmp/vul4j/{instance_id}", cwd="/")

        env.execute(f"vul4j checkout --id {instance_id} -d /tmp/vul4j/{instance_id}", cwd="/")

        # Remove human_patch to prevent patch leakage
        env.execute(f"rm -rf /tmp/vul4j/{instance_id}/VUL4J/human_patch", cwd="/")
        env.execute(f'find /tmp/vul4j/{instance_id}/VUL4J/vulnerable/ -name "*.java" -type f -delete', cwd="/")

        exit_status, result = agent.run(
            CVE_ID=CVE_ID,
            CWE_ID=CWE_ID,
            CVE_DESCRIPTION=CVE_DESCRIPTION,
            CWE_DESCRIPTION=CWE_DESCRIPTION,
            experience_context=experience_context,  # Inject experiences
            task="",
        )
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            raw_traj_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result or "")
        progress_manager.on_instance_end(instance_id, exit_status)


def read_vul4j_dataset(dataset: str):
    vuls = []
    with open(dataset, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["no"]) <= 79:
                vuls.append(row)
    return vuls


@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option(
        "minisweagent/data/vul4j_dataset.csv", "-d", "--dataset", help="Dataset File", rich_help_panel="Basic"
    ),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(
        1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"
    ),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(
        None,
        "--model-class",
        help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')",
        rich_help_panel="Advanced",
    ),
    redo_existing: bool = typer.Option(
        True, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"
    ),
    config_spec: Path = typer.Option(
        builtin_config_dir / "extra" / "vul4j.yaml",
        "--config",
        "-C",
        help="Path to a config file",
        rich_help_panel="Basic",
    ),
    # Evaluation-related options
    eval_only: bool = typer.Option(
        False, "--eval-only", help="Only run evaluation on existing predictions", rich_help_panel="Evaluation"
    ),
    generate_report: bool = typer.Option(
        False,
        "--generate-report",
        help="Generate report from existing evaluation results (no re-evaluation)",
        rich_help_panel="Evaluation",
    ),
    run_eval: bool = typer.Option(
        False, "--run-eval", help="Run evaluation after repair completes", rich_help_panel="Evaluation"
    ),
    eval_workers: int = typer.Option(
        12, "--eval-workers", help="Number of workers for evaluation", rich_help_panel="Evaluation"
    ),
    eval_timeout: int = typer.Option(
        600, "--eval-timeout", help="Timeout for each evaluation (seconds)", rich_help_panel="Evaluation"
    ),
    filter_vul_ids: str = typer.Option(
        "", "--filter", "-f", help="Filter specific vulnerability IDs (comma-separated)", rich_help_panel="Evaluation"
    ),
    skip_existing_eval: bool = typer.Option(
        True, "--skip-existing-eval", help="Skip already evaluated instances", rich_help_panel="Evaluation"
    ),
    # Experience repository-related options
    run_mode: RunMode = typer.Option(
        RunMode.RETRIEVE_ONLY,
        "--mode",
        help="Run mode: baseline (no exp), retrieve-only (inject but no update)",
        rich_help_panel="Experience Enhancement",
    ),
    exp_db_path: str = typer.Option(
        None,
        "--exp-db",
        help="ExpBank vector database path (default: <PROJECT_ROOT>/expbank/vec_DB)",
        rich_help_panel="Experience Enhancement",
    ),
    exp_model_path: str = typer.Option(
        None,
        "--exp-model",
        help="BGE model path (default: <PROJECT_ROOT>/expbank/models/bge-large-en-v1.5)",
        rich_help_panel="Experience Enhancement",
    ),
    exclude_current_cve: bool = typer.Option(
        True,
        "--exclude-current-cve/--no-exclude-current-cve",
        help="Whether to exclude current CVE's experiences from retrieval (default: True)",
        rich_help_panel="Experience Enhancement",
    ),
    exp_retrieve_k: int = typer.Option(
        3,
        "--exp-retrieve-k",
        help="Number of top similar experiences to retrieve first (default: 3)",
        rich_help_panel="Experience Enhancement",
    ),
    exp_return_n: int = typer.Option(
        3,
        "--exp-return-n",
        help="Number of experiences to return after quality sorting (default: 3)",
        rich_help_panel="Experience Enhancement",
    ),
    exp_sort_by_quality: bool = typer.Option(
        True,
        "--exp-sort-by-quality/--no-exp-sort-by-quality",
        help="Sort retrieved experiences by quality score (default: True)",
        rich_help_panel="Experience Enhancement",
    ),
    temporal_filter: bool = typer.Option(
        False,
        "--temporal-filter/--no-temporal-filter",
        help="Only use experiences from earlier CVEs (by CVE ID year/number, default: False)",
        rich_help_panel="Experience Enhancement",
    ),
) -> None:
    # fmt: on
    config_path = get_config_path(config_spec)
    config = yaml.safe_load(config_path.read_text())
    logger.info(f"Loading agent config from '{config_path}'")
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    if output:
        output_path = Path(output)
    else:
        output_path = _default_output_path(
            config.get("model", {}).get("model_name"),
            exp_retrieve_k,
            exp_return_n,
        )
        logger.info(f"No output directory provided, defaulting to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # ========== Evaluation mode ==========
    if eval_only:
        logger.info("Running in EVAL-ONLY mode")
        _run_evaluation_only(
            output_path=output_path,
            workers=eval_workers,
            timeout=eval_timeout,
            filter_vul_ids=filter_vul_ids,
            skip_existing=skip_existing_eval,
        )
        return

    # ========== Generate report from existing results mode ==========
    if generate_report:
        logger.info("Running in GENERATE-REPORT mode")
        from minisweagent.run.extra.vul4j_evaluate import generate_report_from_existing_results

        generate_report_from_existing_results(
            output_dir=output_path / "evaluation",
            preds_path=output_path / "preds.json",
        )
        return

    # ========== Repair mode ==========
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    # ========== Experience repository configuration ==========
    # Resolve the experience repository path
    if exp_db_path is None:
        exp_db_path = str(EXPBANK_ROOT / "vec_DB")
    else:
        exp_db_path = str(Path(exp_db_path).resolve())

    if exp_model_path is None:
        exp_model_path = str(EXPBANK_ROOT / "models" / "bge-large-en-v1.5")
    else:
        exp_model_path = str(Path(exp_model_path).resolve())

    # Log the run mode
    logger.info(f"Run Mode: {run_mode.value.upper()}")
    if run_mode == RunMode.BASELINE:
        logger.info("  - Experience injection: DISABLED")
    elif run_mode == RunMode.RETRIEVE_ONLY:
        logger.info("  - Experience injection: ENABLED")
        logger.info(f"  - ExpBank database: {exp_db_path}")
        logger.info(f"  - BGE model: {exp_model_path}")

    # Initialize the experience repository retriever
    retriever = None
    if run_mode != RunMode.BASELINE:
        if not EXPBANK_AVAILABLE:
            logger.warning(f"ExpBank not available at {EXPBANK_ROOT}, running without experience enhancement")
        else:
            try:
                logger.info("Initializing ExpBank retriever...")
                retriever = ExperienceRetriever(db_path=exp_db_path, model_path=exp_model_path)
            except Exception as e:
                logger.error(f"Failed to initialize ExpBank retriever: {e}")
                logger.warning("Continuing without experience enhancement")

    logger.info(f"Loading dataset {dataset}")
    instances = read_vul4j_dataset(dataset)
    logger.info(f"Loading dataset complete, {len(instances)} instances in total")
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["vul_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    # Store experience repository parameters in the config
    config["exclude_current_cve"] = exclude_current_cve
    config["exp_retrieve_k"] = exp_retrieve_k
    config["exp_return_n"] = exp_return_n
    config["exp_sort_by_quality"] = exp_sort_by_quality

    logger.info(
        f"Experience retrieval: retrieve_k={exp_retrieve_k}, return_n={exp_return_n}, sort_by_quality={exp_sort_by_quality}"
    )
    logger.info(f"Exclude current CVE: {exclude_current_cve}")

    # ========== Save run configuration ==========
    run_config = {
        "basic": {
            "dataset": str(Path(dataset).resolve()) if dataset else "",
            "output": str(output_path),
            "workers": workers,
            "model": model or config.get("model", {}).get("model_name", ""),
            "model_class": model_class,
            "config_file": str(config_path),
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
    }

    run_config_path = output_path / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
    logger.info(f"Run config saved to: {run_config_path}")

    # Create the raw_traj directory
    (output_path / "raw_traj").mkdir(parents=True, exist_ok=True)

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                vul_id = futures[future]
                logger.error(f"Error in future for instance {vul_id}: {e}")
                progress_manager.on_uncaught_exception(vul_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                    retriever,
                    run_mode,
                    temporal_filter,
                ): instance["vul_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    # ========== Automatically run evaluation after repair completes ==========
    if run_eval:
        logger.info("Repair completed, starting evaluation...")
        _run_evaluation_only(
            output_path=output_path,
            workers=eval_workers,
            timeout=eval_timeout,
            filter_vul_ids=filter_vul_ids,
            skip_existing=skip_existing_eval,
        )


def _run_evaluation_only(
    output_path: Path,
    workers: int = 16,
    timeout: int = 600,
    filter_vul_ids: str = "",
    skip_existing: bool = True,
) -> None:
    """Run evaluation only, without running repair."""
    from minisweagent.run.extra.vul4j_evaluate import (
        EvalConfig,
        ReportGenerator,
        Vul4jEvaluator,
        setup_logging,
    )

    preds_path = output_path / "preds.json"
    if not preds_path.exists():
        logger.error(f"Predictions file not found: {preds_path}")
        return

    # Parse the filtered vulnerability IDs
    filter_ids = []
    if filter_vul_ids:
        filter_ids = [x.strip() for x in filter_vul_ids.split(",") if x.strip()]

    # Create the evaluation configuration
    eval_config = EvalConfig(
        preds_path=preds_path,
        output_dir=output_path / "evaluation",
        workers=workers,
        timeout=timeout,
        filter_vul_ids=filter_ids,
        skip_existing=skip_existing,
        verbose=False,
    )

    # Set up logging
    eval_logger = setup_logging(verbose=False)
    eval_logger.info(f"Starting evaluation for predictions from: {preds_path}")
    eval_logger.info(f"Output directory: {eval_config.output_dir}")
    eval_logger.info(f"Workers: {workers}, Timeout: {timeout}s")

    # Run evaluation
    evaluator = Vul4jEvaluator(eval_config, eval_logger)
    predictions = evaluator.load_predictions()
    eval_logger.info(f"Loaded {len(predictions)} predictions")

    results = evaluator.evaluate(predictions)

    # Generate the report
    report_gen = ReportGenerator(results, eval_config.output_dir)
    summary = report_gen.generate_summary()

    report_gen.save_results_json()
    report_gen.save_summary_json(summary)
    report_gen.save_summary_text(summary)
    report_gen.save_errors_json()

    # Print the summary
    report_gen.print_summary_table(summary)

    eval_logger.info(f"Evaluation completed. Results saved to: {eval_config.output_dir}")


if __name__ == "__main__":
    app()

# python minisweagent/run/extra/vul4j.py -d /root/autodl-tmp/MultiLocationFix/data/vul_location.json -o /root/autodl-tmp/mini-swe-agent-main/output --model dashscope/qwen3-max-preview --model-class litellm
# python minisweagent/run/extra/vul4j.py -d /root/autodl-tmp/MultiLocationFix/data/vul_location.json -o /root/autodl-tmp/mini-swe-agent-main/output --model dashscope/qwen3-coder-plus-2025-07-22 --model-class litellm
