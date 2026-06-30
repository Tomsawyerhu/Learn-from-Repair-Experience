"""
Experience Store Cold Start Module

Reads is_poc=false vulnerability data from input.json, automatically generates experience descriptions,
builds experience JSON files and stores them in vector database, to solve the problem of empty initial experience store.

Usage:
    python -m src.rag.cli cold-start --input datasets/input.json --output datasets/cold_start
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_input_data(input_path: str) -> list[dict[str, Any]]:
    """
    Load input.json and filter records with is_poc=false

    Args:
        input_path: input.json file path

    Returns:
        Filtered vulnerability record list
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    logger.info(f"Loading input data: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Filter records with is_poc=false
    filtered = [record for record in data if not record.get("is_poc")]
    logger.info(f"Total {len(data)} records, filtered to {len(filtered)} (is_poc=false)")

    return filtered


def generate_experience_text(record: dict[str, Any]) -> str:
    """
    Generate experience description text based on vulnerability information

    Args:
        record: Single vulnerability record

    Returns:
        Concatenated experience description text
    """
    cve_id = record.get("cve_id", "Unknown")
    cve_description = record.get("cve_description", "")
    cwe_ids = record.get("cwe_id", [])
    cwe_info = record.get("cwe_info", {})
    programming_language = record.get("programing_language", "Unknown")
    vul_funcs = record.get("vul_func", [])
    fix_funcs = record.get("fix_func", [])

    # Build the CWE description
    cwe_parts = []
    for cwe_id in cwe_ids:
        cwe_data = cwe_info.get(cwe_id, {})
        cwe_name = cwe_data.get("name", cwe_id)
        cwe_desc = cwe_data.get("description", "")
        if cwe_desc:
            cwe_parts.append(f"{cwe_id} ({cwe_name}): {cwe_desc}")
        else:
            cwe_parts.append(f"{cwe_id} ({cwe_name})")

    cwe_section = "\n".join(cwe_parts) if cwe_parts else "Unknown vulnerability type"

    # Build vulnerable-location and code information
    vul_sections = []
    for vul in vul_funcs:
        file_path = vul.get("file_path", "unknown")
        start_line = vul.get("start_line", "?")
        end_line = vul.get("end_line", "?")
        snippet = vul.get("snippet", "")

        vul_sections.append(
            f"File: `{file_path}` (lines {start_line}-{end_line})\n```{programming_language.lower()}\n{snippet}\n```"
        )

    vul_code_section = "\n\n".join(vul_sections) if vul_sections else "Vulnerable code not available."

    # Build fixed-code information
    fix_sections = []
    for fix in fix_funcs:
        file_path = fix.get("file_path", "unknown")
        start_line = fix.get("start_line", "?")
        end_line = fix.get("end_line", "?")
        snippet = fix.get("snippet", "")

        fix_sections.append(
            f"File: `{file_path}` (lines {start_line}-{end_line})\n```{programming_language.lower()}\n{snippet}\n```"
        )

    fix_code_section = "\n\n".join(fix_sections) if fix_sections else "Fix code not available."

    # Assemble the full experience text
    experience_text = f"""# Vulnerability Repair Experience: {cve_id}

## Vulnerability Overview

This CVE ({cve_id}) is classified as:
{cwe_section}

**Description:** {cve_description}

**Programming Language:** {programming_language}

## Vulnerable Code Location

{vul_code_section}

## Correct Fix

{fix_code_section}

## Key Repair Insights

- **Vulnerability Type:** {", ".join(cwe_ids) if cwe_ids else "Unknown"}
- **Language:** {programming_language}
- The fix addresses the security issue by modifying the vulnerable code section with proper security controls.
- This experience is auto-generated from patch data and can serve as a reference for similar vulnerability patterns.
"""

    return experience_text.strip()


def build_experience_json(record: dict[str, Any]) -> dict[str, Any]:
    """
    Build experience JSON in specified format

    Args:
        record: Single vulnerability record

    Returns:
        Experience JSON dictionary
    """
    cve_id = record.get("cve_id", "")
    cve_description = record.get("cve_description", "")
    cwe_ids = record.get("cwe_id", [])
    cwe_info = record.get("cwe_info", {})

    # Generate the experience text
    experience_text = generate_experience_text(record)

    experience_json = {
        "cve_id": cve_id,
        "cve_description": cve_description,
        "repair_result": "failure",
        "cwe_ids": cwe_ids,
        "cwe_info": cwe_info,
        "extracted_at": datetime.now().isoformat(),
        "template_used": "cold_start",
        "model_name": "",
        "tag": ["cold_start"],
        "score": 3,
        "programing_language": record.get("programing_language") or record.get("programming_language", ""),
        "experience": experience_text,
    }

    return experience_json


def run_cold_start(
    input_path: str,
    output_dir: str,
    db_path: str = "vec_DB",
    model_path: str = "models/bge-large-en-v1.5",
    no_index: bool = False,
) -> dict[str, Any]:
    """
    Execute cold start process

    Args:
        input_path: input.json file path
        output_dir: Output experience JSON directory
        db_path: ChromaDB database path
        model_path: BGE model path
        no_index: Whether to skip indexing to database

    Returns:
        Statistics dictionary
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    records = load_input_data(input_path)

    if not records:
        logger.warning("No records with is_poc=false found")
        return {"total": 0, "generated": 0, "indexed": 0, "errors": []}

    stats = {"total": len(records), "generated": 0, "indexed": 0, "errors": []}

    generated_files = []

    # Generate experience JSON files
    logger.info(f"Start generating experience JSON files to: {output_path}")
    for i, record in enumerate(records):
        try:
            cve_id = record.get("cve_id", f"unknown_{i}")
            experience_json = build_experience_json(record)

            # Save to file
            json_file = output_path / f"{cve_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(experience_json, f, ensure_ascii=False, indent=2)

            generated_files.append(str(json_file))
            stats["generated"] += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Generation progress: {i + 1}/{len(records)}")

        except Exception as e:
            logger.error(f"Failed to generate experience {record.get('cve_id', 'unknown')}: {e}")
            stats["errors"].append({"cve_id": record.get("cve_id", "unknown"), "error": str(e)})

    logger.info(f"Experience JSON generation completed: {stats['generated']}/{stats['total']}")

    # Index to database
    if not no_index and generated_files:
        logger.info("Start indexing experiences to vector database...")

        try:
            from src.rag.embedder import BGEEmbedder
            from src.rag.experience_store import ExperienceStore

            embedder = BGEEmbedder(model_path=model_path)
            store = ExperienceStore(db_path=db_path, embedder=embedder)

            index_stats = store.add_experiences_from_dir(str(output_path), pattern="*.json", overwrite=True)

            stats["indexed"] = index_stats.get("added_records", 0)
            stats["index_errors"] = index_stats.get("errors", [])

            logger.info(f"Indexing completed: {stats['indexed']} records")

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            stats["index_error"] = str(e)

    return stats
