#!/usr/bin/env python3
"""
ExpBank Unified Pipeline

Process raw trajectory files into experiences and store them in vector database.

Usage:
    1. Command line batch processing:
        python -m src.pipeline --input-dir ./raw_traj --output-dir ./output

    2. Python programmatic call:
        from src.pipeline import Pipeline
        pipeline = Pipeline(config_path="src/config/run/pipeline_config.yaml")
        result = pipeline.run_all()
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock

import yaml
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Configure logging
log_dir = Path(__file__).resolve().parents[0] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Unified Pipeline Class

    Supports two usage modes:
    1. Command line batch processing directory
    2. Python import for single file processing (supports sync/async)

    Three-stage workflow:
    - Stage 1: Trajectory Extraction (raw_traj -> extracted)
    - Stage 2: Experience Extraction (extracted -> exp_json)
    - Stage 3: Experience Indexing (exp_json -> vec_DB)

    Output Directory Structure:
    output_dir/
    └── turn_1/
        ├── extracted/      # Stage 1 output
        ├── exp_json/       # Stage 2 output
        └── vec_DB/         # Stage 3 output (vector database)
    """

    def __init__(
        self,
        config_path: str = "/mnt/data/hhc/VulAgentEvolver/expbank/src/config/run/pipeline_config.yaml",
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        summary_path: Optional[str] = None,
        vul_info_path: Optional[str] = None,
        db_dir: Optional[str] = None,
    ):
        """
        Initialize Pipeline

        Args:
            config_path: Config file path
            input_dir: Input directory (overrides config file)
            output_dir: Output base directory (overrides config file)
            summary_path: Evaluation result file path (overrides config file)
            vul_info_path: Vulnerability info file path (overrides config file)
            db_dir: Specify existing vector database directory (if specified, no new turn directory is generated)
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Allow command line to override config
        if input_dir:
            self.config["input"]["raw_traj_dir"] = input_dir
        if output_dir:
            self.config["output"]["base_dir"] = output_dir
        if summary_path:
            self.config["input"]["summary_path"] = summary_path
        if vul_info_path:
            self.config["input"]["vulnerability_info"] = vul_info_path

        # Specified database directory (command line args priority, then config file)
        if db_dir:
            self._specified_db_dir = Path(db_dir)
        else:
            config_db_path = self.config.get("rag", {}).get("db_path")
            self._specified_db_dir = Path(config_db_path) if config_db_path else None

        # Background task management
        self._tasks: Dict[str, Dict] = {}
        self._tasks_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Lazy loaded components
        self._extractor = None
        self._embedder = None
        self._store = None

        # Current running turn directory (set in run_all)
        self._current_turn_dir: Optional[Path] = None

        logger.info(f"Pipeline initialized with config file: {config_path}")
        if self._specified_db_dir:
            logger.info(f"Using the specified database directory: {self._specified_db_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def _get_next_turn_number(self, base_dir: Path) -> int:
        """
        Get next turn number

        Args:
            base_dir: Output base directory

        Returns:
            Next available turn number
        """
        if not base_dir.exists():
            return 1

        # Find existing turn_* directories
        existing_turns = []
        pattern = re.compile(r"^turn_(\d+)$")

        for item in base_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    existing_turns.append(int(match.group(1)))

        if not existing_turns:
            return 1

        return max(existing_turns) + 1

    def _setup_turn_directory(self, base_dir: Path) -> Path:
        """
        Create the turn directory layout for this run

        Args:
            base_dir: Base output directory

        Returns:
            Path to the created turn directory
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        turn_num = self._get_next_turn_number(base_dir)
        turn_dir = base_dir / f"turn_{turn_num}"

        # Create subdirectories
        (turn_dir / "extracted").mkdir(parents=True, exist_ok=True)
        (turn_dir / "exp_json").mkdir(parents=True, exist_ok=True)
        (turn_dir / "vec_DB").mkdir(parents=True, exist_ok=True)

        logger.info(f"Created run directory: {turn_dir}")
        return turn_dir

    def _get_summary_path(self) -> Optional[Path]:
        """Get the evaluation summary file path"""
        # Prefer the path specified in the config
        configured_path = self.config["input"].get("summary_path")
        if configured_path:
            return Path(configured_path)

        raw_traj_dir = Path(self.config["input"]["raw_traj_dir"])

        # First look under raw_traj_dir (supports the exp_turn_N layout)
        # Layout: exp_turn_N/evaluation/summary.json
        inferred_path_in_dir = raw_traj_dir / "evaluation" / "summary.json"
        if inferred_path_in_dir.exists():
            return inferred_path_in_dir

        # Otherwise look for evaluation beside raw_traj_dir (compatible with the old layout)
        # Layout: turn_N/raw_traj and turn_N/evaluation/summary.json
        inferred_path = raw_traj_dir.parent / "evaluation" / "summary.json"
        if inferred_path.exists():
            return inferred_path

        return None

    def _get_vulnerability_info_path(self) -> Optional[Path]:
        """Get the vulnerability info file path"""
        # Prefer the path specified in the config
        configured_path = self.config["input"].get("vulnerability_info")
        if configured_path:
            return Path(configured_path)

        raw_traj_dir = Path(self.config["input"]["raw_traj_dir"])

        # Read the relative path from the config, defaulting to "datasets/input.json"
        relative_path = self.config["input"].get("vulnerability_info_relative_path", "datasets/input.json")

        # First look under raw_traj_dir (supports the exp_turn_N layout)
        inferred_path_in_dir = raw_traj_dir / relative_path
        if inferred_path_in_dir.exists():
            return inferred_path_in_dir

        # Otherwise infer the path beside raw_traj_dir (compatible with the old layout)
        inferred_path = raw_traj_dir.parent / relative_path
        if inferred_path.exists():
            return inferred_path

        return None

    def _get_success_set(self) -> set:
        """Get the set of successful CVEs from summary.json"""
        summary_path = self._get_summary_path()

        if not summary_path or not summary_path.exists():
            logger.warning(f"Evaluation summary file does not exist: {summary_path}，marking all trajectories as failed")
            return set()

        logger.info(f"Reading evaluation summary: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        success_set = set()
        if "strict_evaluation" in data:
            for cve_list in data["strict_evaluation"].get("successful_cves", {}).values():
                success_set.update(cve_list)

        logger.info(f"Strict successes: {len(success_set)} cases")
        return success_set

    def _get_poc_only_success_cves(self, base_input_dir: Path) -> set:
        """
        Get the CVEs that passed only the PoC test but failed strict_evaluation

        Experiences for these CVEs do not need to be updated and should be kept as-is

        Args:
            base_input_dir: Input base directory (for example metadata/exp_turn_1)

        Returns:
            Set of CVE IDs that passed only PoC, excluding strict_evaluation successes
        """
        summary_path = base_input_dir / "evaluation" / "summary.json"

        # Try looking in the parent directory (compatible with raw_traj being used as input_dir)
        if not summary_path.exists():
            parent_summary = base_input_dir.parent / "evaluation" / "summary.json"
            if parent_summary.exists():
                summary_path = parent_summary

        if not summary_path.exists():
            logger.warning(f"Evaluation summary file does not exist: {summary_path}")
            return set()

        logger.info(f"Reading evaluation summary: {summary_path}")

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Get CVEs that passed the PoC check
            poc_success = set()
            poc_eval = data.get("poc_only_evaluation", {})
            for cve_list in poc_eval.get("successful_cves", {}).values():
                poc_success.update(cve_list)

            # Get CVEs that passed strict_evaluation
            strict_success = set()
            strict_eval = data.get("strict_evaluation", {})
            for cve_list in strict_eval.get("successful_cves", {}).values():
                strict_success.update(cve_list)

            # CVEs that passed only PoC but failed strict = PoC successes - strict successes
            poc_only_success = poc_success - strict_success

            logger.info(
                f"PoC successes: {len(poc_success)} cases, Strict successes: {len(strict_success)} cases, "
                f"PoC-only successes: {len(poc_only_success)} cases"
            )
            return poc_only_success
        except Exception as e:
            logger.error(f"Failed to read evaluation results: {e}")
            return set()

    def _get_vulnerability_info(self) -> Dict[str, Dict]:
        """Load the vulnerability-info mapping"""
        vul_path = self._get_vulnerability_info_path()

        if not vul_path or not vul_path.exists():
            logger.warning(f"Vulnerability info file does not exist")
            return {}

        with open(vul_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vul_map = {}
        for item in data:
            cve_id = item.get("cve_id")
            if cve_id:
                vul_map[cve_id] = {
                    "cve_id": cve_id,
                    "cve_description": item.get("cve_description"),
                    "cwe_info": item.get("cwe_info"),
                    "vul_func": item.get("vul_func"),
                    "fix_func": item.get("fix_func"),
                    "programming_language": item.get("programming_language"),
                }

        logger.info(f"Loaded vulnerability info: {len(vul_map)} entries")
        return vul_map

    def _init_experience_extractor(self, extracted_dir: Path, exp_json_dir: Path):
        """
        Initialize the experience extractor

        Args:
            extracted_dir: Directory containing extracted trajectories
            exp_json_dir: Experience output directory
        """
        from src.agent.experience_extractor_agent import ExperienceExtractor

        self._extractor = ExperienceExtractor(
            config_path=self.config["llm"]["config_path"],
            extracted_dir=str(extracted_dir),
            output_dir=str(exp_json_dir),
            template_name=self.config["llm"]["template_name"],
            client_type=self.config["llm"].get("client_type", "openai"),
        )
        return self._extractor

    def _init_rag_components(self, db_path: Path, clear_db: bool = True):
        """
        Initialize RAG components with automatic CPU fallback on OOM

        Args:
            db_path: Vector database path
            clear_db: Whether to clear the existing database
        """
        from src.rag.embedder import BGEEmbedder
        from src.rag.experience_store import ExperienceStore

        # Clear the database if requested
        if clear_db and db_path.exists():
            logger.info(f"Clearing vector database: {db_path}")
            shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)

        try:
            # Try loading the embedder with CUDA, falling back to CPU automatically
            logger.info("Initializing BGEEmbedder...")
            self._embedder = BGEEmbedder(model_path=self.config["rag"]["model_path"])
            logger.info(f"BGEEmbedder initialized successfully on device: {self._embedder.device}")
        except Exception as e:
            logger.error(f"Failed to initialize BGEEmbedder: {e}")
            raise

        try:
            logger.info("Initializing ExperienceStore...")
            self._store = ExperienceStore(db_path=str(db_path), embedder=self._embedder)
            logger.info(f"ExperienceStore initialized successfully, database located at: {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ExperienceStore: {e}")
            raise

        return self._store

    # ========== Stage 1: trajectory extraction ==========

    def _extract_trajectory(
        self, traj_file: Path, success_set: set, vul_info_map: Dict, output_dir: Path
    ) -> Optional[Path]:
        """
        Extract conversation and vulnerability info from raw traj.json files

        Args:
            traj_file: Path to the raw .traj.json file
            success_set: Set of successful CVEs
            vul_info_map: Vulnerability info mapping
            output_dir: Output directory

        Returns:
            Path to the extracted file, or None on failure
        """
        try:
            with open(traj_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trajectory file {traj_file}: {e}")
            return None

        messages = data.get("messages")
        if not isinstance(messages, list):
            logger.warning(f"Trajectory file has an invalid format {traj_file}: missing the messages field")
            return None

        # Extract conversation
        traj_list = []
        for message in messages:
            msg = {
                "role": message.get("role"),
                "content": message.get("content"),
            }
            if message.get("role") == "assistant":
                msg["tool_calls"] = message.get("extra", {}).get("tool_calls", [])
            traj_list.append(msg)

        if not traj_list:
            logger.warning(f"Trajectory file is empty {traj_file}")
            return None

        # Parse the CVE ID
        stem = traj_file.stem.split(".")[0]

        # Determine the success/failure label
        if stem in success_set:
            output_name = f"{stem}_success.json"
        else:
            output_name = f"{stem}_fail.json"

        # Get vulnerability info
        vul_info = vul_info_map.get(
            stem, {"cve_id": stem, "cve_description": None, "cwe_info": None, "vul_func": None, "fix_func": None}
        )

        # Build the output data
        final_data = {"traj": traj_list, "vulnerability_info": vul_info}

        # Save the file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Trajectory extraction completed: {output_path}")
        return output_path

    # ========== Stage 2: experience extraction ==========

    async def _extract_experiences_batch(
        self,
        extracted_dir: Path,
        exp_json_dir: Path,
        workers: int = 4,
        max_files: Optional[int] = None,
        skip_cves: Optional[set] = None,
    ) -> List[Path]:
        """
        Extract experiences in batch

        Args:
            extracted_dir: Directory containing extracted trajectories
            exp_json_dir: Experience output directory
            workers: Number of concurrent workers
            max_files: Maximum number of files to process
            skip_cves: Set of CVE IDs to skip

        Returns:
            List of successfully extracted experience files
        """
        extractor = self._init_experience_extractor(extracted_dir, exp_json_dir)

        # Collect all trajectory files
        all_traj_files = list(extracted_dir.glob("*.json"))

        # Filter
        traj_files = []
        for f in all_traj_files:
            if skip_cves:
                # File names like CVE-2021-32783_fail.json map to CVE-2021-32783
                cve_id = f.stem.split("_")[0]
                if cve_id in skip_cves:
                    continue
            traj_files.append(f)

        if max_files:
            traj_files = traj_files[:max_files]

        logger.info(f"Trajectory files to process: {len(traj_files)} (skipped: {len(all_traj_files) - len(traj_files)})")

        if not traj_files:
            return []

        try:
            # Use a semaphore to control concurrency
            semaphore = asyncio.Semaphore(workers)

            async def process_file(traj_file: Path):
                async with semaphore:
                    # Call extract_single, which handles persistence automatically
                    # Note: extract_single expects a string path
                    return await extractor.extract_single(str(traj_file), save=True)

            # Create and execute tasks
            tasks = [process_file(f) for f in traj_files]

            # Display progress with tqdm.asyncio
            results = []
            with logging_redirect_tqdm():
                for f in atqdm.as_completed(tasks, total=len(tasks), desc="Extracting experiences"):
                    res = await f
                    results.append(res)

            # Collect generated file paths
            generated_files = []
            if extractor.agent.model_name and extractor.template_name:
                safe_model = "".join([c for c in extractor.agent.model_name if c.isalnum() or c in ("-", "_")]).strip()
                safe_template = "".join([c for c in extractor.template_name if c.isalnum() or c in ("-", "_")]).strip()
                target_dir = exp_json_dir / safe_model / safe_template
                if target_dir.exists():
                    generated_files = list(target_dir.glob("*.json"))

            return generated_files

        except Exception as e:
            logger.error(f"Batch experience extraction failed: {e}")
            return []

    # ========== Stage 3: experience indexing ==========

    def _index_experience(self, experience_file: Path, skip_cves: set = None) -> bool:
        """
        Index experiences into the vector database with automatic downgrade on OOM

        Args:
            experience_file: Path to the experience JSON file
            skip_cves: Set of CVEs to skip (already passed the PoC test)

        Returns:
            Whether the operation succeeded
        """
        # Extract the CVE ID from the file name, for example CVE-2022-23538_success.json -> CVE-2022-23538
        cve_id = experience_file.stem.split("_")[0]

        if skip_cves and cve_id in skip_cves:
            logger.info(f"Skipping experiences for CVEs that already passed the PoC test: {cve_id}")
            return True  # Return True to indicate successful handling even though the entry was skipped

        try:
            count = self._store.add_experience(str(experience_file), overwrite=True)
            logger.info(f"Experience indexing completed: {experience_file}, added {count} records, device: {self._store.embedder.device}")
            return count > 0
        except RuntimeError as e:
            # Handle out-of-memory exceptions
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.error(
                    f"Experience indexing failed {experience_file}: out of memory - {e}\n"
                    f"Current device: {self._store.embedder.device}\n"
                    f"Suggestion: reduce the worker count or switch to CPU"
                )
                return False
            else:
                logger.error(f"Experience indexing error {experience_file}: {e}")
                return False
        except Exception as e:
            logger.error(f"Experience indexing error {experience_file}: {e}")
            return False

    # ========== Batch processing entry point for CLI use==========

    def run_all(
        self,
        stages: Optional[List[str]] = None,
        dry_run: bool = False,
        max_files: Optional[int] = None,
        workers: int = 4,
        clear_db: bool = False,
        db_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline over a directory in batch mode

        Args:
            stages: List of stages to run ["extract", "experience", "index"], defaults to all
            dry_run: Validate the configuration without executing
            max_files: Maximum number of files to process
            workers: Number of concurrent workers
            clear_db: Whether to clear the vector database (default: False)
            db_dir: Specify an existing vector database directory (overrides the __init__ setting)

        Returns:
            Processing statistics
        """
        if stages is None:
            stages = ["extract", "experience", "index"]

        stats = {
            "total_files": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "stages_run": stages,
            "dry_run": dry_run,
            "turn_dir": None,
            "db_dir": None,
        }

        if dry_run:
            logger.info("=== DRY RUN mode: validate configuration only ===")
            self._validate_config()
            logger.info("Configuration validation passed")
            return stats

        # Determine which database directory to use: prefer the run_all argument, then the __init__ setting
        specified_db_dir = Path(db_dir) if db_dir else self._specified_db_dir

        if specified_db_dir:
            # Use the specified database directory without creating a new turn directory
            db_path = specified_db_dir
            db_path.mkdir(parents=True, exist_ok=True)
            stats["db_dir"] = str(db_path)
            logger.info(f"Using the specified vector database directory: {db_path}")

            # Temporary directories are still required for the extract and experience stages
            base_dir = Path(self.config["output"]["base_dir"])
            self._current_turn_dir = self._setup_turn_directory(base_dir)
            stats["turn_dir"] = str(self._current_turn_dir)
        else:
            # Create the turn directory for this run using the original logic
            base_dir = Path(self.config["output"]["base_dir"])
            self._current_turn_dir = self._setup_turn_directory(base_dir)
            stats["turn_dir"] = str(self._current_turn_dir)
            db_path = self._current_turn_dir / "vec_DB"
            stats["db_dir"] = str(db_path)

        # Define per-stage directories
        extracted_dir = self._current_turn_dir / "extracted"
        exp_json_dir = self._current_turn_dir / "exp_json"

        # Load auxiliary data
        success_set = self._get_success_set()
        vul_info_map = self._get_vulnerability_info()

        # Determine the input base directory, supporting the exp_turn_N layout
        base_input_dir = Path(self.config["input"]["raw_traj_dir"])

        # Stage 1: trajectory extraction
        if "extract" in stages:
            # Support the new directory layout by searching for raw_traj under base_input_dir
            raw_traj_dir = base_input_dir / "raw_traj"
            if not raw_traj_dir.exists():
                # Remain compatible with the old logic: if no raw_traj subdirectory exists, use the original directory
                raw_traj_dir = base_input_dir
                logger.info(f"Using the original input directory: {raw_traj_dir}")
            else:
                logger.info(f"Found a raw_traj subdirectory under the input directory: {raw_traj_dir}")

            prefix = self.config["input"].get("prefix", "CVE")

            # Find all .traj.json files
            traj_files = []
            if raw_traj_dir.exists():
                for subdir in raw_traj_dir.iterdir():
                    if subdir.is_dir() and subdir.name.startswith(prefix):
                        for f in subdir.glob("*.traj.json"):
                            traj_files.append(f)

            if max_files:
                traj_files = traj_files[:max_files]

            stats["total_files"] = len(traj_files)
            logger.info(f"Stage 1: found {len(traj_files)} trajectory files")

            with logging_redirect_tqdm():
                for i, traj_file in enumerate(tqdm(traj_files, desc="Extract trajectories")):
                    result = self._extract_trajectory(traj_file, success_set, vul_info_map, extracted_dir)
                    stats["processed"] += 1
                    if result:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1

        # Stage 2: experience extraction
        if "experience" in stages:
            logger.info(f"Stage 2: start batch experience extraction, workers={workers}")

            # Get the list of PoC-only successful CVEs (strict_evaluation failed) for skipping
            # Experiences for these CVEs do not need to be updated and should remain unchanged
            skip_cves = self._get_poc_only_success_cves(base_input_dir)
            if skip_cves:
                logger.info(f"Stage 2: skipping {len(skip_cves)} PoC-only successful CVEs (keeping existing experiences)")

            exp_files = asyncio.run(
                self._extract_experiences_batch(
                    extracted_dir=extracted_dir,
                    exp_json_dir=exp_json_dir,
                    workers=workers,
                    max_files=max_files,
                    skip_cves=skip_cves,
                )
            )

            if "extract" not in stages:
                stats["total_files"] = len(exp_files)
                stats["processed"] = len(exp_files)
                stats["success"] = len(exp_files)

        # Stage 3: experience indexing
        if "index" in stages:
            # Get the list of PoC-only successful CVEs (strict_evaluation failed) for skipping
            poc_only_success_cves = self._get_poc_only_success_cves(base_input_dir)
            if poc_only_success_cves:
                logger.info(
                    f"Stage 3: skipped {len(poc_only_success_cves)} PoC-only successful CVEs "
                    "(keeping existing experiences)"
                )

            # Initialize RAG components
            # Back up the existing database if it exists
            if db_path.exists() and any(db_path.iterdir()):
                backup_path = self._current_turn_dir / "vec_DB_before_update"
                # Avoid self-backups when db_path is already backup_path
                if db_path.resolve() != backup_path.resolve():
                    logger.info(f"Backing up the experience database: {db_path} -> {backup_path}")
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    try:
                        shutil.copytree(db_path, backup_path)
                    except Exception as e:
                        logger.error(f"Failed to back up the experience database: {e}")

            try:
                self._init_rag_components(db_path, clear_db=clear_db)
            except Exception as e:
                logger.error(f"Failed to initialize RAG components: {e}，skipping stage 3")
                stats["index_error"] = str(e)
                return stats

            # Collect all experience files
            exp_files_to_index = []
            for pattern_dir in exp_json_dir.glob("*/*"):
                if pattern_dir.is_dir():
                    exp_files_to_index.extend(pattern_dir.glob("*.json"))

            if not exp_files_to_index:
                # Try searching directly under exp_json_dir
                exp_files_to_index = list(exp_json_dir.glob("*.json"))

            logger.info(f"Stage 3: start indexing {len(exp_files_to_index)} experience files (device: {self._store.embedder.device})")

            indexed_count = 0
            skipped_count = 0
            index_errors = 0
            with logging_redirect_tqdm():
                for i, exp_file in enumerate(tqdm(exp_files_to_index, desc="Indexing experiences")):
                    cve_id = exp_file.stem.split("_")[0]
                    if poc_only_success_cves and cve_id in poc_only_success_cves:
                        skipped_count += 1
                        continue
                    if self._index_experience(exp_file, skip_cves=poc_only_success_cves):
                        indexed_count += 1
                    else:
                        index_errors += 1
                        stats["failed"] += 1

            stats["indexed"] = indexed_count
            stats["skipped_poc_only"] = skipped_count
            stats["index_errors"] = index_errors
            stats["embedding_device"] = self._store.embedder.device
            logger.info(
                f"Indexing finished: indexed {indexed_count}, skipped {skipped_count} PoC-only experiences, "
                f"failed {index_errors} items (device: {self._store.embedder.device})"
            )

        logger.info(f"Processing completed: {stats}")
        return stats

    def _validate_config(self):
        """Validate configuration entries"""
        required_keys = [
            ("input.raw_traj_dir", self.config.get("input", {}).get("raw_traj_dir")),
            ("output.base_dir", self.config.get("output", {}).get("base_dir")),
            ("llm.config_path", self.config.get("llm", {}).get("config_path")),
        ]

        for name, value in required_keys:
            if not value:
                logger.warning(f"Missing configuration entry: {name}")
            else:
                p = Path(value)
                if name.endswith("_dir") and not p.exists():
                    logger.warning(f"Directory does not exist: {name} = {value}")

        # Check the RAG model
        model_path = Path(self.config.get("rag", {}).get("model_path", ""))
        if model_path and not model_path.exists():
            logger.warning(f"BGE model path does not exist: {model_path}")

        # Check auxiliary files (summary.json, input.json)
        raw_traj_dir = self.config.get("input", {}).get("raw_traj_dir")
        if raw_traj_dir:
            base_input_dir = Path(raw_traj_dir)
            logger.info("Checking auxiliary files:")

            # Check the skip list for PoC-only successful CVEs that failed strict_evaluation
            skip_cves = self._get_poc_only_success_cves(base_input_dir)
            if skip_cves:
                logger.info(f"✓ Found evaluation results: skipping {len(skip_cves)} PoC-only successful CVEs (keeping existing experiences)")
            else:
                logger.info(f"ℹ No CVEs need to be skipped, or the evaluation summary file does not exist (summary.json)")

            # Check vulnerability info
            vul_info = self._get_vulnerability_info()
            if vul_info:
                logger.info(f"✓ Found vulnerability metadata, loaded {len(vul_info)} records (input.json)")
            else:
                logger.warning(f"⚠ Vulnerability metadata not found (input.json)")

        logger.info("Configuration checks completed")


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="ExpBank unified pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python -m src.pipeline --input-dir ./raw_traj --output-dir ./output
  
  # Trajectory extraction only
  python -m src.pipeline --input-dir ./raw_traj --output-dir ./output --stages extract
  
  # Use the specified vector database directory without creating a new turn directory to store it
  python -m src.pipeline --input-dir ./raw_traj --output-dir ./output --db-dir ./my_vec_db
  
  # Clear the vector database (default: keep existing data)
  python -m src.pipeline --input-dir ./raw_traj --output-dir ./output --clear-db
  
  # Validate the configuration
  python -m src.pipeline --config ... --dry-run
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="src/config/run/pipeline_config.yaml",
        help="Configuration file path (default: src/config/run/pipeline_config.yaml)",
    )
    parser.add_argument("--input-dir", "-i", default=None, help="Input directory (overrides raw_traj_dir in the config)")
    parser.add_argument("--output-dir", "-o", default=None, help="Base output directory (overrides base_dir in the config)")
    parser.add_argument("--db-dir", default=None, help="Specify an existing vector database directory (if set, do not create a new vec_DB inside the turn directory)")
    parser.add_argument("--summary-path", default=None, help="Evaluation summary file path (overrides summary_path in the config)")
    parser.add_argument("--vul-info-path", default=None, help="Vulnerability info file path (overrides vulnerability_info in the config)")
    parser.add_argument("--stages", "-s", default=None, help="Stages to run, comma-separated (extract,experience,index)")
    parser.add_argument("--dry-run", action="store_true", help="Validate the configuration without executing")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument(
        "--clear-db", action="store_true", default=False, help="Clear the vector database (default: False, keep existing data)"
    )

    args = parser.parse_args()

    # Parse stages
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]

    # Run the pipeline
    pipeline = Pipeline(
        config_path=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        vul_info_path=args.vul_info_path,
        db_dir=args.db_dir,
    )
    stats = pipeline.run_all(
        stages=stages, dry_run=args.dry_run, max_files=args.max_files, workers=args.workers, clear_db=args.clear_db
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
