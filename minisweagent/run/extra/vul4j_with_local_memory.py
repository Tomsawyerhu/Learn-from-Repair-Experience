import json
import os
import subprocess
import threading
import time
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

_HELP_TEXT = """Run self evolved mini-SWE-agent on VUL4J instances.

"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = threading.Lock()

vul4j_root = '/root/autodl-tmp/vul_repo'


def vul_checkout(vul_id):
    def add_maven_local_before_central(file_path):
        """
        In non-Docker Vul4J runs, some packages may fail to download. After
        installing them manually, make Gradle prefer the local Maven cache.
        """
        if not os.path.isfile(file_path):
            print(f"Warning: file does not exist, skipping: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        output_lines = []

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            # Detect mavenCentral() lines, allowing minor formatting differences.
            if 'mavenCentral()' in stripped_line:
                # Check whether the previous line already contains mavenLocal().
                prev_line = output_lines[-1].strip() if output_lines else ""
                if 'mavenLocal()' not in prev_line:
                    output_lines.append(
                        ' ' * (len(line) - len(line.lstrip())) + "mavenLocal()            // Prefer local repository first\n")
                    modified = True

            output_lines.append(line)

        # Write the file back only when changes were made.
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)
            print(f"Updated: {file_path}")
        else:
            print(f"No update needed (already present or mavenCentral not found): {file_path}")

    if not os.path.exists(vul4j_root):
        os.makedirs(vul4j_root)
    path = f'{vul4j_root}/{vul_id}'
    if os.path.exists(path):
        subprocess.run(['rm', '-rf', path])
    # $ vul4j checkout --id VUL4J-1 -d /root/autodl-tmp/vul4j/repos/VUL4J-1
    result = subprocess.run([
        'vul4j', 'checkout', '--id', vul_id, '-d', path
    ], capture_output=True,  # Capture stdout and stderr.
        text=True,  # Return output as strings instead of bytes.
        timeout=100000)  # Optional timeout in seconds.
    GRADLE_SETTINGS = path + '/build.gradle'
    if os.path.exists(GRADLE_SETTINGS):
        add_maven_local_before_central(GRADLE_SETTINGS)
        subprocess.run([
            'git', 'add', GRADLE_SETTINGS
        ], cwd=path)

        subprocess.run([
            'git', 'commit', '-m', 'modify gradle settings'
        ], cwd=path)

    if result.returncode == 0:
        logger.info(f'Checkout vul {vul_id} to {path} success')
    else:
        logger.error(f'Checkout vul {vul_id} to {path} fail, error msg: {result}')


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
        root_dir:str,
        output_dir: Path,
        config: dict,
        progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["vul_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))

    CVE_ID = instance['CVE_id']
    buggy_file = instance['buggy_file']
    buggy_lines = str(instance['buggy_line']).replace('\n', '')
    buggy_methods = str(instance['buggy_method_with_comment']).replace('\n', '')

    progress_manager.on_instance_start(instance_id)

    agent = None
    extra_info = None

    try:
        env = LocalEnvironment(cwd=vul4j_root+'/'+instance['vul_id'], timeout=300)
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,

            **config.get("agent", {}),
        )
        vul_checkout(instance_id)
        exit_status, result = agent.run(CVE_ID=CVE_ID,
                                        vul_id=instance_id,
                                        buggy_file=buggy_file,
                                        buggy_lines=buggy_lines,
                                        buggy_methods=buggy_methods,
                                        root_dir=root_dir,
                                        task='')
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def read_vul4j_location(location_file: str):
    with open(location_file, 'r') as f:
        instances = json.load(f)
    instances = [x for x in instances if int(x['vul_id'].split('-')[-1])<=79 and  int(x['vul_id'].split('-')[-1]) not in [4] and
                 'buggy_line' in x and "buggy_method_with_comment" in x and 'single_flie' in x and int(
                     x['single_flie']) == 1]
    return instances


@app.command(help=_HELP_TEXT)
def main(
        dataset: str = typer.Option("", "-d", "--dataset", help="Dataset File", rich_help_panel="Basic"),
        output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
        root_dir: str = typer.Option("", "-r", "--root-dir", help="Root dir of mini SWE-Agent", rich_help_panel="Basic"),
        workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing",
                                    rich_help_panel="Basic"),
        model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
        model_class: str | None = typer.Option(None, "-c", "--model-class",
                                               help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')",
                                               rich_help_panel="Advanced"),
        redo_existing: bool = typer.Option(True, "--redo-existing", help="Redo existing instances",
                                           rich_help_panel="Data selection"),
        config_spec: Path = typer.Option(builtin_config_dir / "extra" / "vul4j_with_local_memory.yaml", "-c", "--config",
                                         help="Path to a config file", rich_help_panel="Basic"),
        # environment_class: str | None = typer.Option(None, "--environment-class",
        #                                              help="Environment type to use. Recommended are docker or local",
        #                                              rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent_self_evolve.log")

    logger.info(f"Loading dataset {dataset}")
    instances = read_vul4j_location(dataset)
    logger.info(f"Loading dataset complete, {len(instances)} instances in total")
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["vul_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    # if environment_class is not None:
    #     config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                vul_id = futures[future]
                logger.error(f"Error in future for instance {vul_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(vul_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, root_dir, output_path, config, progress_manager): instance[
                    "vul_id"
                ]
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


if __name__ == "__main__":
    app()

# python minisweagent/run/extra/vul4j_with_local_memory.py -d /root/autodl-tmp/MultiLocationFix/data/vul_location.json -o /root/autodl-tmp/mini-swe-agent-main/output_self_evolve --model dashscope/qwen3-max-preview --model-class litellm --root-dir /root/autodl-tmp/mini-swe-agent-main/src/minisweagent
# python minisweagent/run/extra/vul4j_with_local_memory.py -d /root/autodl-tmp/MultiLocationFix/data/vul_location.json -o /root/autodl-tmp/mini-swe-agent-main/output_self_evolve --model openai/gpt-5-mini --model-class litellm --root-dir /root/autodl-tmp/mini-swe-agent-main/src/minisweagent
