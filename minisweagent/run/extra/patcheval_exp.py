#!/usr/bin/env python3

"""Run mini-SWE-agent on PatchEval instances with ExpBank experience enhancement."""
# This is an enhanced version of patcheval.py with ExpBank experience retrieval integration

import concurrent.futures
import copy
import json
import os
import sys
import threading
import time
import traceback
from enum import Enum
from pathlib import Path

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
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.tools.base import ToolManager
from minisweagent.tools.evolve_tools import EvolveToolsManager, EvolveToolsManagerConfig

# Compute project root directory (VulAgentEvolver/)
# Current file: VulAgentEvolver/minisweagent/run/extra/patcheval_exp.py
# Project root: ../../../ (3 levels up)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPBANK_ROOT = PROJECT_ROOT / "expbank"
PATCHEVAL_ROOT = PROJECT_ROOT / "PatchEval"

# ExpBank imports
sys.path.append(str(EXPBANK_ROOT))
from src.rag.experience_retriever import ExperienceRetriever

# Run statistics module
from minisweagent.run.extra.run_statistics import RunStatistics, create_cve_statistics

_HELP_TEXT = """Run mini-SWE-agent on PatchEval instances with experience enhancement.

Run Modes:
  - baseline: No experience injection (same as original patcheval)
  - retrieve-only: Inject experiences but don't update experience bank
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


class RunMode(str, Enum):
    """Run mode for experience enhancement."""
    BASELINE = "baseline"          # No experience injection
    RETRIEVE_ONLY = "retrieve-only"  # Inject but don't update


def parse_cve_id(cve_id: str) -> tuple:
    """
    Parse CVE ID into (year, number) tuple for temporal comparison.
    
    Args:
        cve_id: CVE ID in format "CVE-YYYY-NNNNN"
    
    Returns:
        Tuple of (year: int, number: int), or (9999, 999999) if parsing fails
    """
    try:
        parts = cve_id.upper().split("-")
        if len(parts) >= 3 and parts[0] == "CVE":
            year = int(parts[1])
            number = int(parts[2])
            return (year, number)
    except (ValueError, IndexError):
        pass
    return (9999, 999999)  # Invalid CVE, place at end


def escape_rich_markup(text: str) -> str:
    """Escape square brackets in text to prevent Rich Markup parsing errors."""
    if not isinstance(text, str):
        text = str(text)
    return text.replace("[", "[[").replace("]", "]]")


class ProgressTrackingAgent(EnhancedAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, tool_manager: ToolManager, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, tool_manager=tool_manager, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


import subprocess


def check_docker_image(image_name):
    # Run the docker images command.
    result = subprocess.run(['docker', 'images'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check whether the image name appears in the output.
    if image_name in result.stdout:
        return True
    else:
        return False


def get_patcheval_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_prefix = 'ghcr.io/anonymous2578-data/'
    cve_id = instance.get("cve_id").lower()
    image_name = image_prefix + cve_id
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    # Use per-instance env config copy to avoid cross-thread mutation of shared config.
    base_env_config = copy.deepcopy(config.get("environment", {}))
    env_config = {
        **base_env_config,
        "environment_class": "docker",
        "timeout": config.get("docker_timeout", 600),  # Default: 600 seconds, configurable.
    }
    # container_timeout: maximum container lifetime (used by the sleep command).
    # Long-running batches need enough time so the container does not exit early.
    env_config["container_timeout"] = config.get("container_timeout", "24h")
    image_name = get_patcheval_docker_image_name(instance)
    env_config["image"] = image_name
    env = get_environment(env_config)
    # Remove any leaked patch artifact to avoid exposing the correct fix.
    cleanup_output = env.execute("rm -rf -- fix.patch")
    if int(cleanup_output.get("returncode", -1)) != 0:
        logger.warning(
            f"Failed to cleanup leaked patch artifact in container image {image_name}: "
            f"rc={cleanup_output.get('returncode')}"
        )
    return env


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



def load_successful_cves(summary_paths: list[Path]) -> set:
    """Load successful CVEs from multiple summary files or directories."""
    successful_cves = set()
    if not summary_paths:
        return successful_cves

    # First, expand any directories to find summary.json files
    expanded_paths = []
    for path in summary_paths:
        if not path.exists():
            logger.warning(f"Summary path not found: {path}")
            continue
            
        if path.is_dir():
            # Recursively find summary.json
            found_summaries = list(path.rglob("summary.json"))
            if found_summaries:
                logger.info(f"Found {len(found_summaries)} summary files in {path}:")
                for p in found_summaries:
                    logger.info(f"  - {p}")
                expanded_paths.extend(found_summaries)
            else:
                logger.warning(f"No summary.json files found in directory: {path}")
        else:
            expanded_paths.append(path)

    for path in expanded_paths:
        if path.exists():
            try:
                summary_data = json.loads(path.read_text())
                
                # Check strict_evaluation successful_cves
                strict_eval = summary_data.get("strict_evaluation", {})
                success_dict = strict_eval.get("successful_cves", {})
                
                count_before = len(successful_cves)
                for lang_cves in success_dict.values():
                    if isinstance(lang_cves, list):
                        successful_cves.update(lang_cves)
                
                logger.info(f"Loaded {len(successful_cves) - count_before} successful CVEs from {path}")
                    
            except Exception as e:
                logger.error(f"Error loading summary file {path}: {e}")
        else:
            logger.warning(f"Summary file not found: {path}")
            
    return successful_cves


def update_patches_file(output_dir: Path, instance_id: str, patch_content: str):
    """
    Update the patches.json file with the generated patch.
    Maintains a list of {"cve": ..., "fix_patch": ...} objects.
    """
    patches_file = output_dir / "patches.json"
    
    # Ensure parent directory exists
    patches_file.parent.mkdir(parents=True, exist_ok=True)
    
    with _OUTPUT_FILE_LOCK:
        patches_data = []
        if patches_file.exists():
            try:
                patches_data = json.loads(patches_file.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse existing patches file: {patches_file}. creating new one.")
                patches_data = []
        
        # Check if instance already exists in patches list
        existing_idx = -1
        for i, item in enumerate(patches_data):
            if item.get("cve") == instance_id:
                existing_idx = i
                break
        
        new_entry = {
            "cve": instance_id,
            "fix_patch": patch_content
        }
        
        if existing_idx >= 0:
            patches_data[existing_idx] = new_entry
        else:
            patches_data.append(new_entry)
            
        patches_file.write_text(json.dumps(patches_data, indent=4))


from tqdm import tqdm


def read_patcheval_dataset(file_path, language='all'):
    result = []
    docker_records = []  # Records that have a matching Docker image.

    # Read the file and process each record.
    with jsonlines.open(file_path, 'r') as f:
        for line in tqdm(f, desc="Processing dataset", unit="instance"):
            if language != 'all' and line['programing_language'] != language:
                continue
            docker_records.append(line)
    
    return docker_records


def process_instance(
        instance: dict,
        output_dir: Path,
        config: dict,
        progress_manager: RunBatchProgressManager,
        retriever: ExperienceRetriever,
        run_mode: RunMode,
        run_statistics: RunStatistics = None,
        perfect_location: bool = True,
        temporal_filter: bool = False,
) -> None:
    """Process a single PatchEval instance with experience enhancement."""
    cve_id = str(instance["cve_id"])
    instance_dir = output_dir / "raw_traj" / cve_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", cve_id)
    (instance_dir / f"{cve_id}.traj.json").unlink(missing_ok=True)

    _config = EvolveToolsManagerConfig(
        BUG_LOCATION_OUTPUT_DIR=output_dir / 'fl',
        PATCH_OUTPUT_DIR=output_dir / 'patch',
        BUG_REPORT_OUTPUT_DIR=output_dir / 'report',
        VUL_ID=cve_id,
        MODEL_CONFIG=config.get("model", {}),
    )

    tool_manager = EvolveToolsManager(_config)
    model_config = copy.deepcopy(config.get("model", {}))
    model_config["tools"] = tool_manager.get_all_schemas()
    model = get_model(config=model_config)

    cve_description = instance["cve_description"]
    cwe_info = instance["cwe_info"]
    
    # Build query text for embedding (Symmetric to ExperienceStore.add_experience)
    # Concatenate CVE description with CWE details
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
    
    # ========== EXPERIENCE RETRIEVAL ==========
    # Retrieval strategy:
    # 1. If exclude_current_cve=False AND current CVE has experiences:
    #    - Reserve 1 slot for best current CVE experience (priority: success > highest quality)
    #    - Retrieve (n-1) experiences from other CVEs
    # 2. Otherwise: retrieve n experiences (excluding current CVE if configured)
    experience_context = ""
    
    if run_mode != RunMode.BASELINE:
        try:
            logger.info(f"🔍 [{run_mode.value.upper()}] EXPERIENCE RETRIEVAL for {cve_id}")
            logger.info(f"   Query text length: {len(query_text)} chars (CVE+CWE)")
            logger.info(f"   Exclude current CVE: {config.get('exclude_current_cve', True)}")
            
            exclude_current_cve = config.get('exclude_current_cve', True)
            retrieve_k = config.get('exp_retrieve_k', 3)
            return_n = config.get('exp_return_n', 3)
            sort_by_quality = config.get('exp_sort_by_quality', True)
            
            current_cve_exp = None
            experiences_to_use = []
            used_self_experience = False
            
            # ========== STEP 1: Check current CVE experiences (if not excluded) ==========
            if not exclude_current_cve:
                logger.info(f"🔎 Checking for current CVE experiences...")
                current_cve_experiences = retriever.retrieve(
                    cve_description=query_text,
                    k=20,  # Get more to ensure we can find this CVE's experiences
                    min_quality_score=0
                )
                
                # Filter to only this CVE's experiences
                same_cve_exps = [exp for exp in current_cve_experiences if exp.get('cve_id') == cve_id]
                
                if same_cve_exps:
                    logger.info(f"✅ Found {len(same_cve_exps)} experience(s) from current CVE {cve_id}")
                    
                    # Select best experience: priority to success, then highest quality
                    success_exps = [exp for exp in same_cve_exps if exp.get('repair_result') == 'success']
                    if success_exps:
                        # Pick the highest quality success experience
                        current_cve_exp = max(success_exps, key=lambda x: (x.get('quality_score', 0), x.get('similarity_score', 0)))
                        logger.info(f"⭐ Selected SUCCESS experience from current CVE (quality={current_cve_exp.get('quality_score', 0)})")
                    else:
                        # Pick the highest quality failure experience
                        current_cve_exp = max(same_cve_exps, key=lambda x: (x.get('quality_score', 0), x.get('similarity_score', 0)))
                        logger.info(f"📌 Selected FAILURE experience from current CVE (quality={current_cve_exp.get('quality_score', 0)})")
                    
                    used_self_experience = True
                else:
                    logger.info(f"ℹ️  No experiences found for current CVE")
            
            # ========== STEP 2: Retrieve from other CVEs ==========
            # Adjust retrieval count based on whether we have current CVE experience
            if current_cve_exp:
                # Reserve 1 slot for current CVE exp, retrieve (n-1) from others
                other_cve_count = max(0, return_n - 1)
                logger.info(f"📚 Retrieving {other_cve_count} experience(s) from OTHER CVEs (reserved 1 slot for current CVE)")
            else:
                # Retrieve full n experiences
                other_cve_count = return_n
                logger.info(f"📚 Retrieving {other_cve_count} experience(s) from database")
            
            if other_cve_count > 0:
                other_experiences = retriever.retrieve(
                    cve_description=query_text,
                    exclude_cve_id=cve_id,  # Always exclude current CVE here
                    k=other_cve_count,
                    min_quality_score=0,
                    retrieve_top_k=retrieve_k if not current_cve_exp else max(retrieve_k, other_cve_count * 3),
                    sort_by_quality=sort_by_quality
                )
                logger.info(f"   Retrieved {len(other_experiences)} from other CVEs")
            else:
                other_experiences = []
                logger.info(f"   Skipping retrieval from other CVEs (only using current CVE experience)")
            
            # ========== STEP 3: Combine results ==========
            if current_cve_exp:
                # Put current CVE experience at the beginning
                experiences_to_use = [current_cve_exp] + other_experiences
                logger.info(f"✨ Final: 1 current CVE exp + {len(other_experiences)} other exp(s) = {len(experiences_to_use)} total")
            else:
                experiences_to_use = other_experiences
                logger.info(f"📊 Final: {len(experiences_to_use)} experience(s) from other CVEs")
            
            # ========== TEMPORAL FILTER ==========
            # Filter out experiences from CVEs newer than current CVE
            if temporal_filter and experiences_to_use:
                current_cve_tuple = parse_cve_id(cve_id)
                logger.info(f"⏱️  Applying temporal filter: only CVEs before {cve_id} ({current_cve_tuple})")
                
                filtered_experiences = []
                for exp in experiences_to_use:
                    exp_cve_id = exp.get('cve_id', '')
                    exp_cve_tuple = parse_cve_id(exp_cve_id)
                    
                    # Keep only experiences from earlier CVEs
                    if exp_cve_tuple < current_cve_tuple:
                        filtered_experiences.append(exp)
                        logger.debug(f"   ✓ Keeping {exp_cve_id} ({exp_cve_tuple})")
                    else:
                        logger.debug(f"   ✗ Filtering out {exp_cve_id} ({exp_cve_tuple}) - newer than current CVE")
                
                logger.info(f"⏱️  Temporal filter: {len(experiences_to_use)} -> {len(filtered_experiences)} experiences")
                temporal_filtered_count = len(experiences_to_use) - len(filtered_experiences)
                experiences_to_use = filtered_experiences
            else:
                temporal_filtered_count = 0
            
            # ========== COLLECT STATISTICS ==========
            if run_statistics is not None and experiences_to_use:
                programing_language = instance.get("programing_language", "")
                cve_statistics = create_cve_statistics(
                    cve_id=cve_id,
                    programing_language=programing_language,
                    experiences=experiences_to_use,
                    used_self_experience=used_self_experience,
                    temporal_filtered_count=temporal_filtered_count
                )
                run_statistics.add_cve_record(cve_statistics)
                logger.info(f"📊 Statistics collected: success={cve_statistics.success_exp_count}, failure={cve_statistics.failure_exp_count}, cross_lang={cve_statistics.cross_language_count}")
            
            # Format and log retrieved experiences
            if experiences_to_use:
                logger.info(f"✅ Using {len(experiences_to_use)} experience(s) total")
                
                for idx, exp in enumerate(experiences_to_use, 1):
                    logger.info(f"\n📚 Experience #{idx}:")
                    logger.info(f"   CVE: {exp.get('cve_id', 'Unknown')}")
                    logger.info(f"   CWE IDs: {exp.get('cwe_ids', [])}")
                    logger.info(f"   Tags: {exp.get('tags', [])}")
                    logger.info(f"   Similarity: {exp.get('similarity_score', 0):.4f}")
                    logger.info(f"   Quality: {exp.get('quality_score', 0)}/5")
                    logger.info(f"   Repair: {exp.get('repair_result', 'Unknown')}")
                    
                    # Preview experience content
                    exp_content = exp.get('experience', '')
                    if isinstance(exp_content, str):
                        preview = exp_content[:200] + "..." if len(exp_content) > 200 else exp_content
                        logger.info(f"   Preview: {escape_rich_markup(preview)}")
                
                # Format experiences for prompt injection
                formatted_experiences = []
                for exp in experiences_to_use:
                    exp_content = exp.get("experience", "")
                    
                    # Build header with metadata
                    header = f"### Experience from {exp.get('cve_id', 'Unknown')}\n"
                    header += f"**CWE Types**: {', '.join(exp.get('cwe_ids', []))}\n"
                    header += f"**Quality Score**: {exp.get('quality_score', 0)}/5\n"
                    header += f"**Similarity**: {exp.get('similarity_score', 0):.2f}\n"
                    header += f"**Repair Result**: {exp.get('repair_result', 'Unknown')}\n\n"
                    
                    formatted_text = header + str(exp_content) + "\n\n"
                    formatted_experiences.append(formatted_text)
                
                experience_context = "".join(formatted_experiences)
                logger.info(f"📝 Formatted experience context length: {len(experience_context)} chars")
            else:
                logger.warning(f"⚠️  No experiences retrieved for {cve_id}")
                logger.info(f"   CVE Description: {cve_description[:200]}...")
                logger.info(f"   Possible reasons:")
                logger.info(f"   - No high-quality experiences (score>=4) in database")
                logger.info(f"   - Low similarity with existing experiences")
                logger.info(f"   - Empty experience database")
                
        except Exception as e:
            logger.error(f"❌ Error retrieving experiences for {cve_id}: {e}")
            logger.error(f"Traceback:", exc_info=True)
            # Don't exit, continue without experiences
            experience_context = ""
    else:
        logger.info(f"⏭️  [{run_mode.value.upper()}] Skipping experience retrieval")

    
    progress_manager.on_instance_start(cve_id)
    progress_manager.update_instance_status(cve_id, "Pulling/starting docker")

    agent = None
    extra_info = None
    exit_status, result = None, None

    env = None
    try:
        env = get_sb_environment(config, instance)

        agent = ProgressTrackingAgent(
            model,
            env,
            tool_manager=tool_manager,
            progress_manager=progress_manager,
            instance_id=cve_id,
            **config.get("agent", {}),
        )

        source_dir = instance['repo'].split('/')[-1]

        if perfect_location:
            vul_location = "\n".join(
                [f"- {source_dir}/{x['file_path']} (lines {x['start_line']}-{x['end_line']})" for x in
                 instance['vul_func']])
        else:
            vul_location = "You need to locate the vulnerability yourself."

        # ========== INJECT EXPERIENCE ==========
        # Log experience context stats before injection
        logger.info("🔍 Experience context summary:")
        logger.info(f"   Length: {len(experience_context)} chars")
        logger.info(f"   Is empty: {not experience_context}")
        if experience_context:
            logger.info(f"   Preview (first 300 chars): {escape_rich_markup(experience_context[:300])}...")

        # Pass experience_context to agent.run via native API
        exit_status, result = agent.run(
            task='',
            cve_description=cve_description,
            cwe_info=cwe_info,
            source_dir=source_dir,
            vul_location=vul_location,
            experience_context=experience_context  # Experience injection!
        )
    except Exception as e:
        logger.error(f"Error processing instance {cve_id}: {e}", exc_info=True)
        exit_status, result = "Error", str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Clean up Docker environment immediately to prevent container accumulation
        if env is not None:
            try:
                env.cleanup()
                logger.debug(f"Container cleanup requested for {cve_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up Docker environment for {cve_id}: {e}")

        try:
            traj_path = instance_dir / f"{cve_id}.traj.json"
            save_traj(
                agent,
                traj_path,
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                instance_id=cve_id,
                print_fct=logger.info,
            )
            update_preds_file(output_dir / "preds.json", cve_id, model.config.model_name, result)

            # Update patches file directly if submitted
            if exit_status == "Submitted":
                if result and result.strip():
                    update_patches_file(output_dir, cve_id, result)
                else:
                    logger.warning(f"⚠️ Discarding empty patch for {cve_id} despite 'Submitted' status")
        finally:
            progress_manager.on_instance_end(cve_id, exit_status)


@app.command(help=_HELP_TEXT)
def main(
        dataset: str = typer.Option("", "-d", "--dataset", help="Dataset File", rich_help_panel="Basic"),
        output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
        workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing",
                                    rich_help_panel="Basic"),
        model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
        model_class: str | None = typer.Option(None, "--model-class",
                                               help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')",
                                               rich_help_panel="Advanced"),
        redo_existing: bool = typer.Option(True, "--redo-existing", help="Redo existing instances",
                                           rich_help_panel="Data selection"),
        config_spec: Path = typer.Option(builtin_config_dir / "extra" / "patcheval_exp.yaml", "-c", "--config",
                                         help="Path to a config file", rich_help_panel="Basic"),
        summary_path: list[Path] = typer.Option([], "--summary-path", help="Path(s) to summary.json from previous turns to filter out successful instances",
                                         rich_help_panel="Data selection"),
        run_mode: RunMode = typer.Option(RunMode.RETRIEVE_ONLY, "--mode", help="Run mode: baseline (no exp), retrieve-only (inject but no update)",
                                        rich_help_panel="Experience Enhancement"),
        exp_db_path: str = typer.Option(None, "--exp-db", help="ExpBank vector database path (default: <PROJECT_ROOT>/expbank/vec_DB)",
                                       rich_help_panel="Experience Enhancement"),
        exp_model_path: str = typer.Option(None, "--exp-model", help="BGE model path (default: <PROJECT_ROOT>/expbank/models/bge-large-en-v1.5)",
                                          rich_help_panel="Experience Enhancement"),
        exclude_current_cve: bool = typer.Option(True, "--exclude-current-cve/--no-exclude-current-cve", 
                                                 help="Whether to exclude current CVE's experiences from retrieval (default: True)",
                                                 rich_help_panel="Experience Enhancement"),
        max_instances: int = typer.Option(None, "--max-instances", help="Maximum number of instances to run (default: all)", rich_help_panel="Data selection"),
        auto_eval: bool = typer.Option(False, "--auto-eval/--no-auto-eval",
                                       help="Automatically run patch collection and evaluation after processing (default: False)",
                                       rich_help_panel="Post-processing"),
        eval_workers: int = typer.Option(32, "--eval-workers", help="Number of workers for evaluation (default: 32)",
                                        rich_help_panel="Post-processing"),
        docker_timeout: int = typer.Option(600, "--docker-timeout", help="Docker command timeout in seconds (default: 600)",
                                           rich_help_panel="Basic"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration without running (default: False)",
                                    rich_help_panel="Advanced"),
        temporal_filter: bool = typer.Option(False, "--temporal-filter/--no-temporal-filter",
                                             help="Only use experiences from earlier CVEs (by CVE ID year/number, default: False)",
                                             rich_help_panel="Experience Enhancement"),
        exp_retrieve_k: int = typer.Option(3, "--exp-retrieve-k",
                                           help="Number of top similar experiences to retrieve first (default: 3)",
                                           rich_help_panel="Experience Enhancement"),
        exp_return_n: int = typer.Option(3, "--exp-return-n",
                                         help="Number of experiences to return after quality sorting (default: 3)",
                                         rich_help_panel="Experience Enhancement"),
        exp_sort_by_quality: bool = typer.Option(True, "--exp-sort-by-quality/--no-exp-sort-by-quality",
                                                 help="Sort retrieved experiences by quality score (default: True)",
                                                 rich_help_panel="Experience Enhancement"),
) -> None:
    output_path = Path(output)
    
    # Calculate next turn directory
    base_output_path = output_path
    if not base_output_path.exists():
        # If base doesn't exist, we will create it and start at turn_1
        next_id = 1
    else:
        # Find existing turns
        existing_turns = []
        for p in base_output_path.iterdir():
            if p.is_dir() and p.name.startswith("turn_"):
                try:
                    turn_id = int(p.name.split("_")[1])
                    existing_turns.append(turn_id)
                except ValueError:
                    continue
        next_id = max(existing_turns) + 1 if existing_turns else 1
    
    output_path = base_output_path / f"turn_{next_id}"
    logger.info(f"Target output directory: {output_path}")
    
    # Auto-default summary_path to base output directory if not specified
    if not summary_path and base_output_path.exists():
        summary_path = [base_output_path]
        logger.info(f"Auto-defaulting summary path to: {base_output_path}")

    # Resolve config path
    config_path = get_config_path(config_spec)

    # DRY RUN: Validate configuration and exit
    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE: Validating Configuration")
        logger.info("=" * 60)
        validate_config(dataset, output_path, run_mode, exp_db_path, exp_model_path, auto_eval, summary_path, config_path)
             
        logger.info("=" * 60)
        logger.info("Configuration validation passed!")
        logger.info("=" * 60)
        return
    
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    # Add file handler for logging
    # log_file = output_path / "patcheval_exp.log"
    # add_file_handler(log_file)
    # logger.info(f"Logging to {log_file}")
    
    logger.info(f"Results will be saved to {output_path}")
    if os.path.exists(output_path / "minisweagent_exp.log"):
        os.remove(output_path / "minisweagent_exp.log")
    add_file_handler(output_path / "minisweagent_exp.log")

    # Log run mode
    logger.info(f"Run Mode: {run_mode.value.upper()}")
    if run_mode == RunMode.BASELINE:
        logger.info("  - Experience injection: DISABLED")
    elif run_mode == RunMode.RETRIEVE_ONLY:
        logger.info("  - Experience injection: ENABLED")

    # Resolve ExpBank paths relative to project root
    if exp_db_path is None:
        exp_db_path = str(EXPBANK_ROOT / "vec_DB")
    else:
        # Convert to absolute path if relative
        exp_db_path = str(Path(exp_db_path).resolve())
    
    if exp_model_path is None:
        exp_model_path = str(EXPBANK_ROOT / "models" / "bge-large-en-v1.5")
    else:
        # Convert to absolute path if relative
        exp_model_path = str(Path(exp_model_path).resolve())
    
    # Log resolved paths
    if run_mode != RunMode.BASELINE:
        logger.info(f"ExpBank paths resolved:")
        logger.info(f"  - Database: {exp_db_path}")
        logger.info(f"  - BGE Model: {exp_model_path}")
    
    # Initialize ExpBank components (only if needed)
    retriever = None
    
    if run_mode != RunMode.BASELINE:
        logger.info("Initializing ExpBank retriever...")
        retriever = ExperienceRetriever(
            db_path=exp_db_path,
            model_path=exp_model_path
        )

    logger.info(f"Loading dataset {dataset}")
    instances = read_patcheval_dataset(dataset)
    logger.info(f"Loading dataset complete, {len(instances)} instances in total")
    
    # Filter based on existing predictions
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["cve_id"] not in existing_instances]
        
    # Filter based on previous success (if summaries provided)
    if summary_path:
        successful_cves = load_successful_cves(summary_path)
        
        if successful_cves:
            logger.info(f"Total unique successful CVEs loaded: {len(successful_cves)}")
            
            initial_count = len(instances)
            instances = [inst for inst in instances if inst["cve_id"] not in successful_cves]
            skipped_count = initial_count - len(instances)
            
            if skipped_count > 0:
                logger.info(f"Skipping {skipped_count} instances that were already successful in previous turns")
    
    # Apply max instances limit
    if max_instances is not None and max_instances > 0:
        logger.info(f"Limiting to {max_instances} instances")
        instances = instances[:max_instances]
        
    logger.info(f"Running on {len(instances)} instances...")
    
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())

    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    
    # Store exclude_current_cve in config for process_instance
    config["exclude_current_cve"] = exclude_current_cve
    config["docker_timeout"] = docker_timeout  # Propagate docker_timeout into the config.
    
    # Store experience retrieval parameters
    config["exp_retrieve_k"] = exp_retrieve_k
    config["exp_return_n"] = exp_return_n
    config["exp_sort_by_quality"] = exp_sort_by_quality
    
    logger.info(f"Exclude current CVE: {exclude_current_cve}")
    logger.info(f"Docker timeout set to: {docker_timeout} seconds")
    logger.info(f"Temporal filter: {temporal_filter}")
    logger.info(f"Experience retrieval: retrieve_k={exp_retrieve_k}, return_n={exp_return_n}, sort_by_quality={exp_sort_by_quality}")

    # ========== SAVE RUN CONFIG ==========
    # Build and save run_config.json with all runtime parameters
    run_config = {
        "basic": {
            "dataset": str(Path(dataset).resolve()) if dataset else "",
            "output": str(output_path),
            "workers": workers,
            "model": model or config.get("model", {}).get("model_name", ""),
            "model_class": model_class,
            "config_file": str(config_path),
            "docker_timeout": docker_timeout,
        },
        "data_selection": {
            "redo_existing": redo_existing,
            "max_instances": max_instances,
            "summary_paths": [str(p) for p in summary_path] if summary_path else [],
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
        },
        "quick_cmd": {
            "rerun_same_config": f"python -m minisweagent.run.extra.patcheval_exp -d \"{dataset}\" -o \"{output}\" -w {workers} --mode {run_mode.value} --exp-retrieve-k {exp_retrieve_k} --exp-return-n {exp_return_n} {'--exp-sort-by-quality' if exp_sort_by_quality else '--no-exp-sort-by-quality'} {'--temporal-filter' if temporal_filter else ''}".strip(),
            "eval_only": f"python -m minisweagent.run.extra.patcheval_exp -d \"{dataset}\" -o \"{output}\" --auto-eval --eval-workers {eval_workers}",
        }
    }
    
    run_config_path = output_path / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
    logger.info(f"Run config saved to: {run_config_path}")

    # Use a static filename for exit statuses to avoid multiple files in multi-processing
    exit_statuses_file = output_path / "exit_statuses.yaml"
    progress_manager = RunBatchProgressManager(len(instances), exit_statuses_file)
    
    # Initialize run statistics collector
    run_statistics = RunStatistics() if run_mode != RunMode.BASELINE else None

    with Live(progress_manager.render_group, refresh_per_second=4):
        if workers <= 1:
            # Process each instance sequentially (single-threaded)
            for instance in instances:
                try:
                    process_instance(instance, output_path, config, progress_manager, retriever, run_mode, run_statistics=run_statistics, temporal_filter=temporal_filter)
                except Exception as e:
                    logger.error(f"Fatal error processing {instance.get('cve_id')}: {e}", exc_info=True)
                    continue
        else:
            # Multi-worker parallel processing
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

    # ========== SAVE RUN STATISTICS ==========
    if run_statistics is not None and len(run_statistics) > 0:
        stats_file = run_statistics.save(output_path)
        logger.info(f"📊 Run statistics saved to: {stats_file}")
        logger.info(f"   Total CVEs with experiences: {len(run_statistics)}")
        summary = run_statistics.compute_summary()
        logger.info(f"   Avg success exp: {summary['avg_success_exp']:.2f}, Avg failure exp: {summary['avg_failure_exp']:.2f}")
        logger.info(f"   Avg quality score: {summary['avg_quality_score']:.2f}, Avg similarity: {summary['avg_similarity_score']:.4f}")

    # ========== POST-PROCESSING: PATCH COLLECTION AND EVALUATION ==========
    if auto_eval:
        run_post_evaluation(output_path, dataset, eval_workers)


def validate_config(
    dataset: str, 
    output_path: Path, 
    run_mode: RunMode,
    exp_db_path: str,
    exp_model_path: str,
    auto_eval: bool,
    summary_paths: list[Path] = [],
    config_path: Path = None
) -> None:
    """Validate configuration and paths."""
    errors = []
    warnings = []
    
    # Check dataset
    if not dataset:
        errors.append("Dataset file not specified")
    else:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            errors.append(f"Dataset file not found: {dataset}")
        elif not dataset_path.is_file():
            errors.append(f"Dataset path is not a file: {dataset}")
        else:
            try:
                instances = read_patcheval_dataset(str(dataset_path))
                logger.info(f"✓ Dataset file: {dataset} ({len(instances)} instances)")
            except Exception as e:
                errors.append(f"Error reading dataset: {e}")
            
    # Check agent config
    if config_path:
        if not config_path.exists():
            errors.append(f"Config file not found: {config_path}")
        else:
            try:
                yaml.safe_load(config_path.read_text())
                logger.info(f"✓ Agent Config: {config_path} (valid YAML)")
            except Exception as e:
                errors.append(f"Error parsing config file: {e}")
    
    # Check summary paths
    if summary_paths:
        logger.info(f"Checking {len(summary_paths)} summary paths...")
        successful_cves = load_successful_cves(summary_paths)
        if successful_cves:
             logger.info(f"✓ Loaded total {len(successful_cves)} successful CVEs that will be filtered")

    # Check output directory (will be created)
    logger.info(f"✓ Output directory: {output_path} (will be created)")
    
    # Check run mode
    logger.info(f"✓ Run mode: {run_mode.value}")
    
    # Check ExpBank paths if not in baseline mode
    if run_mode != RunMode.BASELINE:
        # Resolve paths
        if exp_db_path is None:
            exp_db_path = str(EXPBANK_ROOT / "vec_DB")
        db_path = Path(exp_db_path)
        
        if exp_model_path is None:
            exp_model_path = str(EXPBANK_ROOT / "models" / "bge-large-en-v1.5")
        model_path = Path(exp_model_path)
        
        # Validate ExpBank paths
        if not db_path.exists():
            warnings.append(f"ExpBank database not found: {db_path} (will use empty if retrieve-only)")
        else:
            logger.info(f"✓ ExpBank database: {db_path}")
        
        if not model_path.exists():
            errors.append(f"BGE model not found: {model_path}")
        else:
            logger.info(f"✓ BGE model: {model_path}")
        

    
    # Check auto-eval paths
    if auto_eval:
        collect_script = PATCHEVAL_ROOT / "scripts" / "collect_patches.py"
        eval_script = PATCHEVAL_ROOT / "patcheval" / "evaluation" / "run_evaluation.py"
        input_json = PATCHEVAL_ROOT / "datasets" / "input.json"
        
        if not collect_script.exists():
            errors.append(f"collect_patches.py not found: {collect_script}")
        else:
            logger.info(f"✓ Collect patches script: {collect_script}")
        
        if not eval_script.exists():
            errors.append(f"run_evaluation.py not found: {eval_script}")
        else:
            logger.info(f"✓ Evaluation script: {eval_script}")
        
        if not input_json.exists():
            errors.append(f"input.json not found: {input_json}")
        else:
            logger.info(f"✓ Input JSON: {input_json}")
    
    # Print warnings
    if warnings:
        logger.warning("-" * 60)
        logger.warning("WARNINGS:")
        for warning in warnings:
            logger.warning(f"  ⚠ {warning}")
    
    # Print errors and exit if any
    if errors:
        logger.error("-" * 60)
        logger.error("ERRORS:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        raise typer.Exit(code=1)


def run_post_evaluation(output_path: Path, dataset_path: str, eval_workers: int = 4):
    """
    Run post-processing: collect patches and run evaluation.
    
    Args:
        output_path: Directory containing CVE subdirectories with trajectories
        dataset_path: Path to the original dataset file (for language mapping)
        eval_workers: Number of workers for evaluation
    """
    import subprocess
    
    logger.info("="*60)
    logger.info("Starting post-processing: Patch Collection & Evaluation")
    logger.info("="*60)
    
    # Paths
    collect_script = PATCHEVAL_ROOT / "scripts" / "collect_patches.py"
    eval_script = PATCHEVAL_ROOT / "patcheval" / "evaluation" / "run_evaluation.py"
    eval_cwd = PATCHEVAL_ROOT / "patcheval" / "evaluation"
    input_json = PATCHEVAL_ROOT / "datasets" / "input.json"
    
    # Create evaluation subdirectory in output
    eval_output_dir = output_path / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    patches_file = output_path / "patches.json"
    eval_run_name = f"run_{int(time.time())}"
    
    # Check if patches file exists
    if not patches_file.exists():
        logger.error(f"Patches file not found: {patches_file}. No patches to evaluate.")
        return
        
    logger.info(f"Using patches file: {patches_file}")
    
    # Step 2: Run evaluation
    logger.info(f"Step 2: Running evaluation")
    
    # Create evaluation_output directory in eval_cwd for run_evaluation.py
    eval_output_base = eval_cwd / "evaluation_output" / eval_run_name
    eval_output_base.mkdir(parents=True, exist_ok=True)
    
    eval_cmd = [
        sys.executable, str(eval_script),
        "--patch_file", str(patches_file),
        "--input_file", str(input_json),
        "--output", eval_run_name,
        "--max_workers", str(eval_workers)
    ]
    logger.info(f"Running: {' '.join(eval_cmd)}")
    logger.info(f"Working directory: {eval_cwd}")
    
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=str(eval_cwd))
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(f"Evaluation stderr: {result.stderr}")
        
        # Copy results to output directory
        eval_results_dir = eval_cwd / "evaluation_output" / eval_run_name
        if eval_results_dir.exists():
            import shutil
            # Copy contents of eval_results_dir to eval_output_dir
            for item in eval_results_dir.iterdir():
                 dest = eval_output_dir / item.name
                 if item.is_dir():
                     if dest.exists():
                         shutil.rmtree(dest)
                     shutil.copytree(item, dest)
                 else:
                     shutil.copy2(item, dest)
            
            logger.info(f"Evaluation results copied to: {eval_output_dir}")
        
        logger.info("Post-processing complete!")
        
    except Exception as e:
        logger.error(f"Error running run_evaluation.py: {e}")
        return


if __name__ == "__main__":
    app()
