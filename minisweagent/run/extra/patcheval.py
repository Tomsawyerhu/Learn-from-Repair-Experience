#!/usr/bin/env python3

"""Run mini-SWE-agent on PatchEval instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import os
import threading
import time
import traceback
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
from minisweagent.tools.evolve_tools import EvolveToolsManager,EvolveToolsManagerConfig

_HELP_TEXT = """Run mini-SWE-agent on PatchEval instances.

"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()


class ProgressTrackingAgent(EnhancedAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, tool_manager: ToolManager,progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, tool_manager=tool_manager,**kwargs)
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
    # Run the `docker images` command.
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
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = 'docker'
    image_name = get_patcheval_docker_image_name(instance)
    env_config["image"] = image_name
    env = get_environment(env_config)
    # if startup_command := config.get("run", {}).get("env_startup_command"):
    #     startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
    #     out = env.execute(startup_command)
    #     if out["returncode"] != 0:
    #         raise RuntimeError(f"Error executing startup command: {out}")
    # Avoid leaking the correct patch.
    env.execute('rm -rf fix.patch')
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


from tqdm import tqdm


def read_patcheval_dataset(file_path,language='all'):
    result = []
    docker_records = []  # Records whose Docker images are available.
    
    # Read and process the dataset file.
    with jsonlines.open(file_path, 'r') as f:
        for line in tqdm(f, desc="Processing dataset", unit="instance"):
            # if check_docker_image(get_patcheval_docker_image_name(line)):
            #     docker_records.append(line)  # Keep records whose Docker images exist.
            # else:
            #     result.append(line)  # Keep the remaining records.
            if language!='all' and line['programing_language']!=language:
                continue
            docker_records.append(line)
    
    # Overwrite the original file with the filtered results.
    # with jsonlines.open(file_path, 'w') as f:
    #     f.write_all(docker_records)  # Write all retained records.
    
    return docker_records[10:]



def process_instance(
        instance: dict,
        output_dir: Path,
        config: dict,
        progress_manager: RunBatchProgressManager,
        perfect_location: bool = True,
) -> None:
    """Process a single SWEBench instance."""
    cve_id = instance["cve_id"]
    instance_dir = output_dir / cve_id
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
    model_config = config.get("model", {})
    model_config["tools"] = tool_manager.get_all_schemas()
    model = get_model(config=model_config)

    cve_description = instance["cve_description"]
    cwe_info = instance["cwe_info"]

    progress_manager.on_instance_start(cve_id)
    progress_manager.update_instance_status(cve_id, "Pulling/starting docker")

    agent = None
    extra_info = None
    exit_status, result = None, None

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

        source_dir=instance['repo'].split('/')[-1]

        if perfect_location:
            vul_location = "\n".join(
                [f"- {source_dir}/{x['file_path']} (lines {x['start_line']}-{x['end_line']})" for x in
                 instance['vul_func']])
        else:
            vul_location = "You need to locate the vulnerability yourself."
        exit_status, result = agent.run(task='',
                                        cve_description=cve_description,
                                        cwe_info=cwe_info,
                                        source_dir=source_dir,
                                        vul_location=vul_location)
    except Exception as e:
        logger.error(f"Error processing instance {cve_id}: {e}", exc_info=True)
        exit_status, result = "Error", str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            instance_dir / f"{cve_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=cve_id,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / "preds.json", cve_id, model.config.model_name, result)
        progress_manager.on_instance_end(cve_id, exit_status)


@app.command(help=_HELP_TEXT)
def main(
        dataset: str = typer.Option("", "-d", "--dataset", help="Dataset File", rich_help_panel="Basic"),
        output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
        workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing",
                                    rich_help_panel="Basic"),
        model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
        model_class: str | None = typer.Option(None, "-c", "--model-class",
                                               help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')",
                                               rich_help_panel="Advanced"),
        redo_existing: bool = typer.Option(True, "--redo-existing", help="Redo existing instances",
                                           rich_help_panel="Data selection"),
        config_spec: Path = typer.Option(builtin_config_dir / "extra" / "patcheval.yaml", "-c", "--config",
                                         help="Path to a config file", rich_help_panel="Basic"),
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    if os.path.exists(output_path / "minisweagent_self_evolve.log"):
        os.remove(output_path / "minisweagent_self_evolve.log")
    add_file_handler(output_path / "minisweagent_self_evolve.log")

    logger.info(f"Loading dataset {dataset}")
    instances = read_patcheval_dataset(dataset)
    logger.info(f"Loading dataset complete, {len(instances)} instances in total")
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["vul_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())

    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    with Live(progress_manager.render_group, refresh_per_second=4):
        # Process each instance sequentially (single-threaded)
        for instance in instances:
            try:
                process_instance(instance, output_path, config, progress_manager)
            except Exception as e:
                continue



if __name__ == "__main__":
    app()

# python minisweagent/run/extra/patcheval.py -d /mnt/data/hhc/VulAgentEvolver/minisweagent/data/patcheval_dataset.jsonl -o /mnt/data/hhc/VulAgentEvolver/minisweagent/output_patcheval --model openai/gpt-5-mini  --model-class litellm_tool
