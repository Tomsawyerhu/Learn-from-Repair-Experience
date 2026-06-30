"""
Experience Extraction Agent
Extracts reusable CWE-level experience from CVE repair trajectories

Usage:
    from src.agent.experience_extractor_agent import ExperienceExtractor

    extractor = ExperienceExtractor(config_path="src/config/model/experience_extraction_config.yaml")
    await extractor.extract_all()  # Process all trajectories
    await extractor.extract_single("path/to/trajectory.json")  # Process single trajectory
"""

import os
import json
import re
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

# Add project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from datetime import datetime
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
from src.agent.agent import Agent
import matplotlib.pyplot as plt
from collections import Counter


def setup_logger(
    name: str = __name__,
    log_dir: str = str(Path(__file__).resolve().parents[1] / "logs"),
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Setup logger with both console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Log format
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler with rotation
    log_filename = log_path / f"experience_extractor_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(log_filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Also create a separate error log
    error_log_filename = log_path / f"experience_extractor_errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = RotatingFileHandler(
        error_log_filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    logger.addHandler(error_handler)

    return logger


# Initialize logger
logger = setup_logger()


class ExperienceExtractor:
    """
    Experience Extraction Agent

    Features:
    1. Parse CVE/CWE info and dialog trajectory from trajectory JSON files
    2. Use LLM to extract reusable experience at CWE level
    3. Classify and store extracted experience by CWE to specified directory
    """

    def __init__(
        self,
        config_path: str = "src/config/model/experience_extraction_config.yaml",
        extracted_dir: str = "extracted",
        output_dir: str = "src/exp_json",
        template_name: str = "exmaple_focused",
        max_trajectory_length: int = 50000,
        client_type: str = "openai",
        max_retries: int = 2,
    ):
        """
        Initialize Experience Extraction Agent

        Args:
            config_path: Config file path
            extracted_dir: Trajectory file directory
            output_dir: Experience output directory
            template_name: Template name to use
            max_trajectory_length: Maximum trajectory character length
            client_type: Client type ("openai" or "litellm")
            max_retries: Maximum retry attempts on JSON parsing errors
        """
        self.config_path = config_path
        self.extracted_dir = Path(extracted_dir)
        self.output_dir = Path(output_dir)
        self.template_name = template_name
        self.max_trajectory_length = max_trajectory_length
        self.max_retries = max_retries
        self.client_type = client_type

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Agent
        self.agent = Agent(config_path=config_path, client_type=client_type)

        logger.info(f"ExperienceExtractor initialized")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Extracted dir: {self.extracted_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Template: {template_name}")
        logger.info(f"  Model: {self.agent.model_name}")
        logger.info(f"  Max retries: {self.max_retries}")

    @property
    def storage_dir(self) -> Path:
        """
        Get dedicated storage directory based on model and template
        Structure: output_dir/model_name/template_name/
        """
        # Clean name to ensure directory safety
        safe_model = "".join([c for c in self.agent.model_name if c.isalnum() or c in ("-", "_")]).strip()
        safe_template = "".join([c for c in self.template_name if c.isalnum() or c in ("-", "_")]).strip()

        return self.output_dir / safe_model / safe_template

    def _parse_repair_result(self, filename: str) -> str:
        """
        Parse repair result from filename

        Args:
            filename: Trajectory filename (e.g., CVE-2017-16025_fail.json)

        Returns:
            "success" or "failure"
        """
        if "_success" in filename:
            return "success"
        elif "_fail" in filename:
            return "failure"
        else:
            return "unknown"

    def _parse_cve_id(self, filename: str) -> str:
        """
        Parse CVE ID from filename

        Args:
            filename: Trajectory filename

        Returns:
            CVE ID (e.g. CVE-2017-16025)
        """
        match = re.match(r"(CVE-\d{4}-\d+)", filename)
        if match:
            return match.group(1)
        return filename.replace(".json", "").split("_")[0]

    def _load_trajectory(self, file_path: Path) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Load the trajectory file

        Args:
            file_path: Path to the trajectory file

        Returns:
            (trajectory dialogue list, vulnerability_info dict or None)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check the vulnerability_info field
        vulnerability_info = None
        if isinstance(data, dict):
            vulnerability_info = data.get("vulnerability_info")
            if "traj" in data:
                return data["traj"], vulnerability_info

        if isinstance(data, list):
            return data, None

        raise ValueError(f"Unexpected trajectory format in {file_path}")

    def _extract_context_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        cve_id: str,
        repair_result: str,
        vulnerability_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract context variables from vulnerability_info and the trajectory for template filling

        Args:
            trajectory: Trajectory dialogue list
            cve_id: CVE ID
            repair_result: Repair result
            vulnerability_info: Optional vulnerability info dict (from the vulnerability_info field in the JSON file)

        Returns:
            Context variable dictionary
        """
        context = {
            "cve_id": cve_id,
            "repair_result": repair_result,
            "cve_description": "",
            "cwe_info": "",
            "vul_location": "",
            "trajectory": "",
            "programing_language": "",
        }

        # If vulnerability_info is provided, extract directly from it
        if vulnerability_info:
            # Extract the CVE description
            context["cve_description"] = vulnerability_info.get("cve_description", "")

            # Extract CWE information and convert the dict into a string format
            cwe_info = vulnerability_info.get("cwe_info", {})
            if isinstance(cwe_info, dict):
                context["cwe_info"] = str(cwe_info)
            else:
                context["cwe_info"] = str(cwe_info) if cwe_info else ""

            # Extract the vulnerability location from vul_func, including file_path and line numbers
            vul_func_list = vulnerability_info.get("vul_func", [])
            if vul_func_list:
                locations = []
                for vf in vul_func_list:
                    file_path = vf.get("file_path", "")
                    start_line = vf.get("start_line", "")
                    end_line = vf.get("end_line", "")
                    if file_path:
                        if start_line and end_line:
                            locations.append(f"- {file_path} (lines {start_line}-{end_line})")
                        else:
                            locations.append(f"- {file_path}")
                context["vul_location"] = "\n".join(locations)

            # Extract the programming language
            programming_language = vulnerability_info.get("programming_language", "")
            if programming_language:
                context["programing_language"] = programming_language

        # Format the trajectory
        context["trajectory"] = self._format_trajectory(trajectory)

        return context

    def _format_trajectory(self, trajectory: List[Dict[str, Any]]) -> str:
        """
        Format the trajectory as readable text

        Args:
            trajectory: Trajectory dialogue list

        Returns:
            Formatted trajectory string
        """
        formatted_parts = []

        for i, msg in enumerate(trajectory):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            # Handle tool_calls
            tool_calls = msg.get("tool_calls", [])

            # Format the message
            part = f"### [{role}] (Turn {i + 1})\n"

            if content:
                # Truncate overly long content
                if len(content) > 2000:
                    content = content[:1000] + "\n...[content truncated]...\n" + content[-500:]
                part += content

            # Add tool call information
            if tool_calls:
                part += "\n\n**Tool Calls:**\n"
                for tc in tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_id = tc.get("id", "")
                    arguments = tc.get("arguments", "")

                    # Truncate overly long arguments
                    if len(arguments) > 500:
                        arguments = arguments[:200] + "...[arguments truncated]..."

                    part += f"- {tool_name} (id: {tool_id}): {arguments}\n"

            formatted_parts.append(part)

        result = "\n\n---\n\n".join(formatted_parts)

        # Overall length limit
        if len(result) > self.max_trajectory_length:
            # Smart truncation: keep the beginning and the end
            head_len = self.max_trajectory_length // 2
            tail_len = self.max_trajectory_length // 4
            result = result[:head_len] + "\n\n---\n\n[...middle portion of the trajectory omitted...]\n\n---\n\n" + result[-tail_len:]

        return result

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response returned by the LLM, with enhanced fault tolerance

        Args:
            response: LLM response string

        Returns:
            Parsed JSON object
        """
        if not response or not isinstance(response, str):
            logger.warning("Empty or invalid LLM response")
            return {"parse_error": True, "raw_response": str(response)}

        # Strategy 1: try extracting JSON from fenced code blocks
        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                json_str = match.group(1).strip()
                result = self._try_parse_json(json_str)
                if result is not None:
                    return result

        # Strategy 2: directly search for the outermost {}
        brace_match = re.search(r"\{[\s\S]*\}", response)
        if brace_match:
            json_str = brace_match.group(0)
            result = self._try_parse_json(json_str)
            if result is not None:
                return result

        # Strategy 3: try extracting from the first { to the last }
        first_brace = response.find("{")
        last_brace = response.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = response[first_brace : last_brace + 1]
            result = self._try_parse_json(json_str)
            if result is not None:
                return result

        # Strategy 4: try fixing common JSON format errors
        if first_brace != -1 and last_brace != -1:
            json_str = response[first_brace : last_brace + 1]
            result = self._try_fix_and_parse_json(json_str)
            if result is not None:
                return result

        # If parsing fails, return the wrapped raw response
        logger.warning(f"Failed to parse JSON from LLM response (length: {len(response)})")
        return {
            "parse_error": True,
            "raw_response": response[:2000] if len(response) > 2000 else response,  # Truncate overly long responses
        }

    def _try_parse_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Try to parse a JSON string"""
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    def _try_fix_and_parse_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Try to fix common JSON format errors and parse the result"""
        # Try fixing common issues
        fixes = [
            # Fix unescaped newline characters
            (r"(?<!\\)\n", r"\\n"),
            # Fix unescaped tab characters
            (r"(?<!\\)\t", r"\\t"),
            # Fix trailing commas
            (r",\s*}", "}"),
            (r",\s*]", "]"),
        ]

        fixed_str = json_str
        for pattern, replacement in fixes:
            fixed_str = re.sub(pattern, replacement, fixed_str)

        result = self._try_parse_json(fixed_str)
        if result is not None:
            logger.info("Successfully parsed JSON after applying fixes")
            return result

        # Try a more permissive parse (handle single quotes)
        try:
            # Replace single quotes with double quotes as a simple fallback
            if "'" in fixed_str and '"' not in fixed_str:
                fixed_str = fixed_str.replace("'", '"')
                result = self._try_parse_json(fixed_str)
                if result is not None:
                    logger.info("Successfully parsed JSON after replacing single quotes")
                    return result
        except Exception:
            pass

        return None

    def _extract_cwe_ids(self, experience: Dict[str, Any]) -> List[str]:
        """
        Get the list of CWE IDs from the extracted experience

        Args:
            experience: Extracted experience JSON

        Returns:
            List of CWE IDs
        """
        cwe_ids = []

        # Extract from the new simplified format (cwe_experiences -> cwe_id)
        if "cwe_experiences" in experience:
            for exp in experience.get("cwe_experiences", []):
                # New simplified format: use the cwe_id field directly
                cwe_id = exp.get("cwe_id", "")
                if cwe_id and cwe_id not in cwe_ids:
                    cwe_ids.append(cwe_id)
                # Compatible with the old format: cwe_profile.id
                if not cwe_id:
                    cwe_profile = exp.get("cwe_profile", {})
                    cwe_id = cwe_profile.get("id", "")
                    if cwe_id and cwe_id not in cwe_ids:
                        cwe_ids.append(cwe_id)

        # Extract from the old format
        elif "cwe_profile" in experience:
            cwe_id = experience.get("cwe_profile", {}).get("id", "")
            if cwe_id:
                cwe_ids.append(cwe_id)

        return cwe_ids if cwe_ids else ["UNKNOWN"]

    def _save_experience(
        self,
        experience: Dict[str, Any],
        cve_id: str,
        repair_result: str,
        cwe_ids: List[str],
        cve_description: str = "",
        cwe_info: Optional[Dict[str, Any]] = None,
        cost: float = 0.0,
    ) -> List[Path]:
        """
        Save the extracted experience to file

        Args:
            experience: Extracted experience
            cve_id: CVE ID
            repair_result: Repair result
            cwe_ids: List of CWE IDs
            cve_description: CVE description
            cwe_info: Full CWE info dict, including the name and description for each CWE
            cost: Cost of generating the experience (USD)

        Returns:
            List of saved file paths
        """
        saved_paths = []

        # Use the new hierarchical directory structure
        target_dir = self.storage_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / f"{cve_id}.json"

        # Build output data: metadata and experience content are stored at the same level
        output_data = {
            "cve_id": cve_id,
            "cve_description": cve_description,
            "repair_result": repair_result,
            "cwe_ids": cwe_ids,
            "cwe_info": cwe_info or {},  # Full CWE information used for RAG indexing
            "extracted_at": datetime.now().isoformat(),
            "template_used": self.template_name,
            "model_name": self.agent.model_name,
            "cost": cost,  # Cost of generating the experience
        }

        # Merge the experience content generated by the LLM
        output_data.update(experience)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        saved_paths.append(output_file)
        logger.info(f"Saved experience to: {output_file}")

        return saved_paths

    async def extract_single(self, trajectory_path: str, save: bool = True) -> Tuple[Dict[str, Any], List[Path]]:
        """
        Extract experience from a single trajectory file

        Args:
            trajectory_path: Path to the trajectory file
            save: Whether to save to file

        Returns:
            (Extracted experience, list of saved file paths)
        """
        path = Path(trajectory_path)
        filename = path.name

        logger.info(f"Processing: {filename}")

        # Parse basic information
        cve_id = self._parse_cve_id(filename)
        repair_result = self._parse_repair_result(filename)

        # Load the trajectory
        trajectory, vulnerability_info = self._load_trajectory(path)

        # Check the vulnerability_info field
        if vulnerability_info is None:
            logger.warning(f"Skipping {filename}: missing vulnerability_info field")
            return {"skipped": True, "reason": "missing vulnerability_info"}, []

        # Extract cve_description from vulnerability_info
        cve_description = vulnerability_info.get("cve_description", "")

        # Extract context, passing vulnerability_info for direct extraction
        context = self._extract_context_from_trajectory(trajectory, cve_id, repair_result, vulnerability_info)

        # If cve_description is missing from context, use the value from vulnerability_info
        if not context["cve_description"] and cve_description:
            context["cve_description"] = cve_description

        # Extract programming_language from vulnerability_info
        if vulnerability_info:
            programming_language = vulnerability_info.get("programming_language", "")
            if programming_language:
                context["programing_language"] = programming_language

        # Create a local Agent instance to ensure thread safety
        local_agent = Agent(config_path=self.config_path, client_type=self.client_type)
        # Clear the previous conversation history
        local_agent.clear_history()

        # Call the LLM to extract experience, with retries
        experience = None
        total_cost = 0.0  # Accumulated cost, including retries

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} for {cve_id}")
                    # Clear history before retrying and start over
                    local_agent.clear_history()

                response = await local_agent.chat(context, self.template_name)
                total_cost += local_agent.get_last_cost()  # Accumulated cost
                experience = self._parse_llm_response(response)

                # Check whether there is a parsing error
                if isinstance(experience, dict) and experience.get("parse_error"):
                    last_error = experience.get("raw_response", "JSON parse error")
                    if attempt < self.max_retries:
                        logger.warning(f"JSON parse error for {cve_id}, will retry ({attempt + 1}/{self.max_retries})")
                        continue
                    else:
                        logger.warning(f"JSON parse error for {cve_id} after {self.max_retries} retries")
                else:
                    # Parsing succeeded, break out of the loop
                    break

            except Exception as e:
                last_error = str(e)
                logger.error(f"Error extracting experience for {cve_id} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    logger.info(f"Will retry ({attempt + 1}/{self.max_retries})")
                    continue
                else:
                    experience = {"extraction_error": str(e), "cve_id": cve_id, "repair_result": repair_result}

        # Extract CWE IDs and full CWE info from vulnerability_info
        cwe_ids = []
        cwe_info = {}
        if vulnerability_info and "cwe_info" in vulnerability_info:
            cwe_info = vulnerability_info["cwe_info"]
            if isinstance(cwe_info, dict):
                # cwe_info is a dict whose keys are CWE IDs (e.g., "CWE-20")
                cwe_ids = list(cwe_info.keys())

        # If no CWE is available from vulnerability_info, use the default value
        if not cwe_ids:
            cwe_ids = ["UNKNOWN"]

        # Save only when there is no error
        saved_paths = []
        has_error = isinstance(experience, dict) and (
            experience.get("parse_error") or experience.get("extraction_error")
        )

        if save and not has_error:
            saved_paths = self._save_experience(
                experience,
                cve_id,
                repair_result,
                cwe_ids,
                cve_description=context.get("cve_description", ""),
                cwe_info=cwe_info,
                cost=total_cost,
            )
        elif save and has_error:
            logger.warning(f"Skipping save for {cve_id} due to parse/extraction error")

        return experience, saved_paths

    async def extract_all(
        self,
        filter_pattern: Optional[str] = None,
        max_files: Optional[int] = None,
        skip_existing: bool = True,
        workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Extract experience from all trajectory files

        Args:
            filter_pattern: Filename filter pattern (regular expression)
            max_files: Maximum number of files to process
            skip_existing: Whether to skip already processed files

        Returns:
            Processing statistics
        """
        # Get all trajectory files
        trajectory_files = list(self.extracted_dir.glob("*.json"))

        # Apply filtering
        if filter_pattern:
            pattern = re.compile(filter_pattern)
            trajectory_files = [f for f in trajectory_files if pattern.search(f.name)]

        # Limit the number of files
        if max_files:
            trajectory_files = trajectory_files[:max_files]

        # Statistics
        stats = {
            "total": len(trajectory_files),
            "processed": 0,
            "skipped": 0,
            "skipped_no_vul_info": 0,
            "success": 0,
            "parse_errors": 0,
            "failed": 0,
            "errors": [],
            "experience_lengths": [],  # Record the length of each successfully extracted experience
        }

        if stats["total"] == 0:
            logger.warning("No trajectory files found matching pattern!")
            return stats

        logger.info("=" * 50)
        logger.info(f"Start Processing {stats['total']} Trajectory Files")
        logger.info(f"Source Directory: {self.extracted_dir}")
        logger.info(f"Output Directory: {self.storage_dir}")
        logger.info(f"Log Directory: {Path(__file__).resolve().parents[1] / 'logs'}")
        logger.info("=" * 50)

        # Record the start time
        start_time = time.time()

        # Use a semaphore to control concurrency
        semaphore = asyncio.Semaphore(workers)
        logger.info(f"Starting parallel extraction with {workers} workers")

        # Initialize the progress bar
        pbar = tqdm(total=stats["total"], desc="Extracting experiences", unit="file")

        async def process_file(traj_file: Path, index: int, total: int):
            cve_id = self._parse_cve_id(traj_file.name)

            # Check whether it already exists in the new storage directory
            if skip_existing:
                existing_file = self.storage_dir / f"{cve_id}.json"
                if existing_file.exists():
                    logger.info(f"[{index}/{total}] Skipping {cve_id} (already exists)")
                    pbar.update(1)
                    return "skipped"

            async with semaphore:
                try:
                    logger.info(f"[{index}/{total}] Processing {cve_id}...")
                    experience, paths = await self.extract_single(str(traj_file), save=True)

                    # Check whether it was skipped because vulnerability_info is missing
                    if isinstance(experience, dict) and experience.get("skipped"):
                        return "skipped_no_vul_info"

                    # Check whether a parsing or extraction error occurred
                    if isinstance(experience, dict):
                        if experience.get("parse_error") or experience.get("extraction_error"):
                            error_msg = experience.get("extraction_error", "JSON parse error")
                            logger.warning(f"Parse/extraction error for {cve_id}: {error_msg}")
                            return {"type": "parse_error", "cve_id": cve_id, "error": error_msg}

                    # Record the experience length
                    exp_len = 0
                    if isinstance(experience, dict) and "experience" in experience:
                        exp_text = experience.get("experience", "")
                        if isinstance(exp_text, str):
                            exp_len = len(exp_text)

                    return {"type": "success", "length": exp_len}

                except Exception as e:
                    logger.error(f"Error processing {cve_id}: {e}")
                    return {"type": "failed", "cve_id": cve_id, "error": str(e)}
                finally:
                    pbar.update(1)

                    # Periodic logging every 10 files (using global stats if possible, but here using index is simpler)
                    # Note: index is static for the task, not dynamic completion order.
                    # For strict progress logging we might want to use a shared counter, but simple periodic log is okay.
                    if (index) % 10 == 0 or index == total:
                        elapsed = time.time() - start_time
                        # Rate calculation is rough here as index is launch order not completion
                        logger.info(f"Processed [{index}/{total}] cve_id={cve_id}")

        # Create and execute tasks
        tasks = [process_file(f, i + 1, stats["total"]) for i, f in enumerate(trajectory_files)]

        results = await asyncio.gather(*tasks)
        pbar.close()

        # Aggregate statistics
        for res in results:
            if res == "skipped":
                stats["skipped"] += 1
            elif res == "skipped_no_vul_info":
                stats["skipped_no_vul_info"] += 1
            elif isinstance(res, dict):
                if res["type"] == "success":
                    stats["processed"] += 1
                    stats["success"] += 1
                    if res.get("length"):
                        stats["experience_lengths"].append(res["length"])
                elif res["type"] == "parse_error":
                    stats["processed"] += 1
                    stats["parse_errors"] += 1
                    stats["errors"].append(res)
                elif res["type"] == "failed":
                    stats["failed"] += 1
                    stats["errors"].append(res)

        # Compute runtime
        end_time = time.time()
        total_runtime = end_time - start_time
        stats["total_runtime_seconds"] = round(total_runtime, 2)

        # Automatically run analysis after batch completion
        logger.info(f"Extraction complete: {stats}")

        # Final summary log
        logger.info("=" * 50)
        logger.info(f"Processing Completed in {total_runtime:.2f}s")
        logger.info(f"Total: {stats['total']}")
        logger.info(f"Success: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Parse Errors: {stats['parse_errors']}")
        logger.info(f"Skipped (Existing): {stats['skipped']}")
        logger.info(f"Skipped (No Vul Info): {stats['skipped_no_vul_info']}")

        if stats["experience_lengths"]:
            avg_len = sum(stats["experience_lengths"]) / len(stats["experience_lengths"])
            logger.info(f"Average Experience Length: {avg_len:.2f} chars")

        if stats["errors"]:
            logger.info("-" * 20 + " Errors " + "-" * 20)
            for err in stats["errors"][:10]:  # Show first 10 errors
                logger.info(f"  {err['cve_id']}: {err['error']}")
            if len(stats["errors"]) > 10:
                logger.info(f"  ... and {len(stats['errors']) - 10} more errors")

        logger.info("=" * 50)

        if stats["success"] > 0 or stats["skipped"] > 0:
            self.analyze_batch(runtime_seconds=total_runtime)

        return stats

    def analyze_batch(self, runtime_seconds: float = 0.0):
        """
        Analyze the experience files from the current batch and generate a report

        Args:
            runtime_seconds: Batch runtime in seconds
        """
        target_dir = self.storage_dir
        if not target_dir.exists():
            logger.warning(f"No directory to analyze at {target_dir}")
            return

        analysis_dir = target_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running analysis for {target_dir}...")

        scores = []
        tags = []
        lengths = []

        files = list(target_dir.glob("*.json"))

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Extract score
                    if "score" in data:
                        scores.append(data["score"])

                    # Extract tags
                    if "tag" in data:
                        t = data["tag"]
                        if isinstance(t, list):
                            tags.extend(t)
                        elif isinstance(t, str):
                            tags.append(t)

                    # Extract experience length
                    if "experience" in data and isinstance(data["experience"], str):
                        lengths.append(len(data["experience"]))
            except Exception as e:
                logger.warning(f"Error reading {filepath} for analysis: {e}")

        # Prepare the prefix
        safe_template = "".join([c for c in self.template_name if c.isalnum() or c in ("-", "_")]).strip()
        safe_model = "".join([c for c in self.agent.model_name if c.isalnum() or c in ("-", "_")]).strip()
        prefix = f"{safe_template}_{safe_model}_"

        # 1. Analyze Score Distribution
        if scores:
            plt.figure(figsize=(10, 6))
            score_counts = Counter(scores)
            sorted_scores = sorted(score_counts.items())
            keys = [str(k) for k, v in sorted_scores]
            values = [v for k, v in sorted_scores]

            plt.bar(keys, values, color="skyblue", edgecolor="black")
            plt.title(f"Distribution of Scores ({self.template_name} - {self.agent.model_name})")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            output_path_score = analysis_dir / f"{prefix}score_distribution.png"
            plt.savefig(output_path_score)
            plt.close()
            logger.info(f"Analysis: Score distribution saved to {output_path_score}")

        # 2. Analyze Tag Distribution
        if tags:
            plt.figure(figsize=(12, 8))
            tag_counts = Counter(tags)
            sorted_tags = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))

            plt.bar(sorted_tags.keys(), sorted_tags.values(), color="lightgreen", edgecolor="black")
            plt.title(f"Distribution of Tags ({self.template_name} - {self.agent.model_name})")
            plt.xlabel("Tag")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            output_path_tag = analysis_dir / f"{prefix}tag_distribution.png"
            plt.savefig(output_path_tag)
            plt.close()
            logger.info(f"Analysis: Tag distribution saved to {output_path_tag}")

        # 3. Analyze Statistics (Avg Length)
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0

        stats_data = {
            "template_used": self.template_name,
            "model_name": self.agent.model_name,
            "client_type": getattr(self.agent, "client_type", "unknown"),
            "sample_count": len(files),
            "average_experience_length": round(avg_length, 2),
            "min_experience_length": min_length,
            "max_experience_length": max_length,
            "average_score": round(avg_score, 2),
            "score_distribution": dict(Counter(scores)) if scores else {},
            "tag_distribution": dict(Counter(tags)) if tags else {},
            "total_runtime_seconds": round(runtime_seconds, 2),
            "avg_time_per_sample": round(runtime_seconds / len(files), 2) if files and runtime_seconds > 0 else 0,
            "generated_at": datetime.now().isoformat(),
        }

        output_path_stats = analysis_dir / f"{prefix}statistics.json"
        with open(output_path_stats, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Analysis: Statistics saved to {output_path_stats}")

    def get_experience_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the experience library

        Returns:
            Experience library statistics
        """
        summary = {"total_experiences": 0, "by_cwe": {}, "by_result": {"success": 0, "failure": 0}, "files": []}

        for exp_file in self.output_dir.glob("*.json"):
            summary["total_experiences"] += 1
            summary["files"].append(str(exp_file))

            # Read metadata from the JSON file, now stored at the top level
            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    result = data.get("repair_result", "unknown")
                    cwe_ids = data.get("cwe_ids", [])

                    if result in ["success", "failure"]:
                        summary["by_result"][result] += 1

                    for cwe_id in cwe_ids:
                        if cwe_id not in summary["by_cwe"]:
                            summary["by_cwe"][cwe_id] = {"success": 0, "failure": 0}
                        if result in ["success", "failure"]:
                            summary["by_cwe"][cwe_id][result] += 1
            except Exception as e:
                logger.warning(f"Error reading {exp_file}: {e}")

        return summary


# Convenience function
async def extract_experiences(
    config_path: str = "src/config/model/experience_extraction_config.yaml",
    extracted_dir: str = "extracted",
    output_dir: str = "src/exp_json",
    filter_pattern: Optional[str] = None,
    max_files: Optional[int] = None,
    skip_existing: bool = True,
    workers: int = 4,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Convenience function: extract all experiences

    Args:
        config_path: Configuration file path
        extracted_dir: Trajectory file directory
        output_dir: Output directory
        filter_pattern: Filename filter pattern
        max_files: Maximum number of files to process
        skip_existing: Whether to skip existing files
        max_retries: Maximum retry count for JSON parsing errors

    Returns:
        Processing summary statistics
    """
    extractor = ExperienceExtractor(
        config_path=config_path, extracted_dir=extracted_dir, output_dir=output_dir, max_retries=max_retries
    )
    return await extractor.extract_all(
        filter_pattern=filter_pattern, max_files=max_files, skip_existing=skip_existing, workers=workers
    )


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experience Extraction Agent")
    parser.add_argument("--config", default="src/config/model/experience_extraction_config.yaml", help="Configuration file path")
    parser.add_argument("--extracted-dir", default="extracted", help="Trajectory file directory")
    parser.add_argument("--output-dir", default="src/exp_json", help="Experience output directory")
    parser.add_argument("--filter", default=None, help="Filename filter pattern (regular expression)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip already processed files")
    parser.add_argument("--single", default=None, help="Process a single trajectory file")
    parser.add_argument("--summary", action="store_true", help="Show the experience library summary")
    parser.add_argument(
        "--client-type", default="openai", choices=["openai", "litellm"], help="Client type (openai or litellm)"
    )
    parser.add_argument(
        "--template",
        default="markdown_sytle",
        help="Prompt template name to use (exmaple_focused, report_style, cwe_focused_extraction, etc.)",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum retry count for JSON parsing errors (default: 2)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent extraction workers (default: 4)")

    args = parser.parse_args()

    async def main():
        extractor = ExperienceExtractor(
            config_path=args.config,
            extracted_dir=args.extracted_dir,
            output_dir=args.output_dir,
            template_name=args.template,
            client_type=args.client_type,
            max_retries=args.max_retries,
        )

        if args.summary:
            summary = extractor.get_experience_summary()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.single:
            experience, paths = await extractor.extract_single(args.single)
            print(json.dumps(experience, ensure_ascii=False, indent=2))
            print(f"\nSaved to: {paths}")
        else:
            stats = await extractor.extract_all(
                filter_pattern=args.filter,
                max_files=args.max_files,
                skip_existing=not args.no_skip,
                workers=args.workers,
            )

            # Print detailed statistics
            print("\n" + "=" * 60)
            print("📊 Experience Extraction Statistics Report")
            print("=" * 60)
            print(f"Template: {args.template}")
            print(f"Client: {args.client_type}")
            print(f"Model: {extractor.agent.model_name}")
            print("-" * 60)
            print(f"Total trajectory files: {stats['total']}")
            print(f"Processed:             {stats['processed']}")
            print(f"✅ Success:             {stats['success']}")
            print(f"⚠️  Parse errors:       {stats['parse_errors']}")
            print(f"❌ Failed:              {stats['failed']}")
            print(f"⏭️  Skipped (already exists): {stats['skipped']}")
            print(f"⏭️  Skipped (missing info):  {stats['skipped_no_vul_info']}")
            print("-" * 60)
            runtime = stats.get("total_runtime_seconds", 0)
            print(f"Total runtime:            {runtime:.2f} seconds")
            if stats["processed"] > 0:
                avg_time = runtime / stats["processed"]
                print(f"Average processing time:  {avg_time:.2f} seconds/item")
            success_rate = (stats["success"] / stats["processed"] * 100) if stats["processed"] > 0 else 0
            print(f"Success rate:             {success_rate:.1f}%")

            # Print output-length statistics
            exp_lengths = stats.get("experience_lengths", [])
            if exp_lengths:
                avg_len = sum(exp_lengths) / len(exp_lengths)
                print("-" * 60)
                print(f"📝 Experience output statistics:")
                print(f"  Average length:         {avg_len:.0f} chars")
                print(f"  Minimum length:         {min(exp_lengths)} chars")
                print(f"  Maximum length:         {max(exp_lengths)} chars")
            print("=" * 60)

            if stats["errors"]:
                print("\n❌ Error details:")
                for err in stats["errors"]:
                    print(f"  - {err['cve_id']}: {err.get('type', 'error')} - {err['error'][:100]}...")

            print("\nRaw JSON statistics:")
            print(json.dumps(stats, ensure_ascii=False, indent=2))

    asyncio.run(main())
