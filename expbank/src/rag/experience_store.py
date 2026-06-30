"""
Experience Storage Module

Based on ChromaDB vector database for storing and indexing vulnerability repair experiences at the CWE level.
Supports single and batch indexing, as well as local persistent storage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperienceStore:
    """
    Experience Vector Database Store

    Uses ChromaDB as the underlying vector database, supporting:
    - Single experience indexing
    - Batch directory indexing
    - Local persistent storage

    Data Structure:
    - ID: {cve_id}_{cwe_id} as primary key
    - Embedding: Vector encoding of cve_description
    - Metadata: cve_id, cwe_id, cwe_name, repair_result, experience, etc.

    Attributes:
        db_path: Database persistence path
        collection_name: ChromaDB collection name
        embedder: Vector encoder instance
        client: ChromaDB client
        collection: ChromaDB collection

    Example:
        >>> from src.rag.embedder import BGEEmbedder
        >>> embedder = BGEEmbedder()
        >>> store = ExperienceStore("vec_DB", embedder)
        >>> store.add_experience("src/exp_json/CVE-2015-3295.json")
    """

    def __init__(
        self,
        db_path: str = "vec_DB",
        embedder=None,
        collection_name: str = "cwe_experiences",
        input_json_path: str | None = None,
    ):
        """
        Initialize experience store

        Args:
            db_path: ChromaDB persistent storage path
            embedder: Vector encoder instance (e.g., BGEEmbedder)
            collection_name: Collection name for distinguishing different experience stores
            input_json_path: input.json path for loading CVE->language mapping

        Raises:
           ImportError: chromadb library not installed
            ValueError: embedder is None
        """
        if embedder is None:
            raise ValueError("embedder parameter must be provided")

        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedder = embedder

        # Ensure database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Lazy import ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Please install chromadb library first: pip install chromadb")

        # Initialize ChromaDB client (persistent mode)
        # Note: New ChromaDB versions use multi-threading by default, no extra config needed
        logger.info(f"Initializing ChromaDB, storage path: {self.db_path}")
        self.client = chromadb.PersistentClient(path=str(self.db_path), settings=Settings(anonymized_telemetry=False))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"description": "CWE-level vulnerability repair experience store"}
        )

        # Load CVE->language mapping
        self.cve_language_map = self._load_cve_language_map(input_json_path)
        if self.cve_language_map:
            logger.info(f"Loaded {len(self.cve_language_map)} CVE language mappings")

        logger.info(f"Experience store ready, current record count: {self.collection.count()}")

    def _load_cve_language_map(self, input_json_path: str | None) -> dict[str, str]:
        """
        Load CVE->language mapping from input.json

        Args:
            input_json_path: input.json file path

        Returns:
            Mapping dictionary from CVE ID to language
        """
        if not input_json_path:
            logger.debug("input_json_path not provided, skipping language mapping load")
            return {}

        path = Path(input_json_path)
        if not path.exists():
            logger.warning(f"input.json does not exist: {path}")
            return {}

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            cve_lang_map = {}
            if isinstance(data, list):
                for item in data:
                    cve_id = item.get("cve_id")
                    # Note: Field is spelled as programing_language (missing an m)
                    lang = item.get("programing_language") or item.get("programming_language")
                    if cve_id and lang:
                        cve_lang_map[cve_id] = lang

            logger.info(f"Loaded {len(cve_lang_map)} CVE language mappings from {path}")
            return cve_lang_map

        except Exception as e:
            logger.error(f"Failed to load language mapping: {e}")
            return {}

    def _parse_experience_json(self, json_path: Path) -> list[dict[str, Any]]:
        """
        Parse experience JSON file, extract CVE-level experience records

        New format supports:
        - Use cve_id as primary key (no longer split by CWE)
        - Concatenate descriptions from cwe_info into cve_description as embedding text
        - Save score as quality_score for retrieval filtering

        Args:
            json_path: Experience JSON file path

        Returns:
            List of experience records, each containing:
            - id: Primary key (cve_id)
            - embedding_text: CVE description + CWE description (for vectorization)
            - cve_id: CVE ID
            - cwe_ids: CWE ID list (JSON string)
            - quality_score: Experience quality score (1-5)
            - tags: Tag list (JSON string)
            - repair_result: Repair result
            - experience: Experience content (Markdown string)
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        records = []
        cve_id = data.get("cve_id", "")
        if not cve_id:
            logger.warning(f"File does not contain cve_id: {json_path}")
            return records

        cve_description = data.get("cve_description", "")
        repair_result = data.get("repair_result", "unknown")

        # New format fields
        cwe_ids = data.get("cwe_ids", [])
        cwe_info = data.get("cwe_info", {})
        tags = data.get("tag", [])
        quality_score = data.get("score", 0)
        experience = data.get("experience", "")

        # Build embedding text: CVE description + CWE description
        embedding_text = cve_description
        if isinstance(cwe_info, dict):
            for cwe_id_key, cwe_data in cwe_info.items():
                if isinstance(cwe_data, dict):
                    cwe_name = cwe_data.get("name", "")
                    cwe_desc = cwe_data.get("description", "")
                    if cwe_name or cwe_desc:
                        embedding_text += f"\n{cwe_id_key}"
                        if cwe_name:
                            embedding_text += f" ({cwe_name})"
                        if cwe_desc:
                            embedding_text += f": {cwe_desc}"

        # Use cve_id as primary key
        record_id = cve_id

        # Get language from mapping
        programming_language = (
            self.cve_language_map.get(cve_id, "")
            or data.get("programing_language")
            or data.get("programming_language", "")
        )

        records.append(
            {
                "id": record_id,
                "embedding_text": embedding_text,
                "cve_id": cve_id,
                "cve_description": cve_description,
                "cwe_ids": json.dumps(cwe_ids, ensure_ascii=False),
                "cwe_info": json.dumps(cwe_info, ensure_ascii=False),
                "tags": json.dumps(tags, ensure_ascii=False),
                "quality_score": quality_score,
                "repair_result": repair_result,
                "programming_language": programming_language,  # Added field
                "experience": experience if isinstance(experience, str) else json.dumps(experience, ensure_ascii=False),
                "source_file": str(json_path.name),
            }
        )

        return records

        cve_description = data.get("cve_description", "")
        repair_result = data.get("repair_result", "unknown")

        # New-format fields
        cwe_ids = data.get("cwe_ids", [])
        cwe_info = data.get("cwe_info", {})
        tags = data.get("tag", [])
        quality_score = data.get("score", 0)
        experience = data.get("experience", "")

        # Build the embedding text: CVE description + CWE description
        embedding_text = cve_description
        if isinstance(cwe_info, dict):
            for cwe_id_key, cwe_data in cwe_info.items():
                if isinstance(cwe_data, dict):
                    cwe_name = cwe_data.get("name", "")
                    cwe_desc = cwe_data.get("description", "")
                    if cwe_name or cwe_desc:
                        embedding_text += f"\n{cwe_id_key}"
                        if cwe_name:
                            embedding_text += f" ({cwe_name})"
                        if cwe_desc:
                            embedding_text += f": {cwe_desc}"

        # Use cve_id as the primary key
        record_id = cve_id

        # Get the language from the mapping
        programming_language = (
            self.cve_language_map.get(cve_id, "")
            or data.get("programing_language")
            or data.get("programming_language", "")
        )

        records.append(
            {
                "id": record_id,
                "embedding_text": embedding_text,
                "cve_id": cve_id,
                "cve_description": cve_description,
                "cwe_ids": json.dumps(cwe_ids, ensure_ascii=False),
                "cwe_info": json.dumps(cwe_info, ensure_ascii=False),
                "tags": json.dumps(tags, ensure_ascii=False),
                "quality_score": quality_score,
                "repair_result": repair_result,
                "programming_language": programming_language,  # Added
                "experience": experience if isinstance(experience, str) else json.dumps(experience, ensure_ascii=False),
                "source_file": str(json_path.name),
            }
        )

        return records

    def add_experience(self, json_path: str, overwrite: bool = True) -> int:
        """
        Index a single experience JSON file, automatically fallback when GPU memory is insufficient

        Args:
            json_path: Experience JSON file path
            overwrite: Whether to overwrite existing records (default True, new experiences overwrite old)

        Returns:
            Number of successfully added/updated records

        Raises:
            FileNotFoundError: File does not exist
            json.JSONDecodeError: JSON parse error
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        # Parse JSON
        records = self._parse_experience_json(path)

        if not records:
            logger.warning(f"File contains no experience records: {path}")
            return 0

        added_count = 0
        for record in records:
            record_id = record["id"]

            # Check if already exists (skip only when not overwriting)
            if not overwrite:
                existing = self.collection.get(ids=[record_id])
                if existing and existing["ids"]:
                    logger.debug(f"Skipping existing record: {record_id}")
                    continue

            # Use text concatenated with CWE description to generate embedding
            try:
                embedding = self.embedder.encode_single(record["embedding_text"])
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    logger.warning(f"Vectorization error ({e}), current device: {self.embedder.device}")
                    raise
                else:
                    raise

            # Add to ChromaDB (new metadata structure)
            self.collection.upsert(
                ids=[record_id],
                embeddings=[embedding],
                metadatas=[
                    {
                        "cve_id": record["cve_id"],
                        "cve_description": record["cve_description"],
                        "cwe_ids": record["cwe_ids"],  # JSON string
                        "cwe_info": record["cwe_info"],  # JSON string
                        "tags": record["tags"],  # JSON string
                        "quality_score": record["quality_score"],  # Experience quality score (1-5)
                        "repair_result": record["repair_result"],
                        "programming_language": record.get("programming_language", ""),  # New
                        "experience": record["experience"],
                        "source_file": record["source_file"],
                        "indexed_at": datetime.now().isoformat(),
                    }
                ],
            )

            added_count += 1
            logger.info(
                f"Indexed: {record_id} (quality score: {record['quality_score']}) [device: {self.embedder.device}]"
            )

        return added_count

    def add_experiences_from_dir(
        self, dir_path: str, pattern: str = "*.json", overwrite: bool = True
    ) -> dict[str, int]:
        """
        Batch index all experience JSON files in a directory

        Args:
            dir_path: Experience JSON file directory
            pattern: File matching pattern
            overwrite: Whether to overwrite existing records (default True)

        Returns:
            Statistics dictionary:
            - total_files: Total file count
            - processed_files: Processed file count
            - added_records: Added record count
            - errors: Error list
        """
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory does not exist: {path}")

        json_files = list(path.glob(pattern))

        stats = {
            "total_files": len(json_files),
            "processed_files": 0,
            "added_records": 0,
            "skipped_records": 0,
            "errors": [],
        }

        logger.info(f"Starting batch indexing, {stats['total_files']} files total")

        for i, json_file in enumerate(json_files):
            try:
                count = self.add_experience(str(json_file), overwrite=overwrite)
                stats["processed_files"] += 1
                stats["added_records"] += count

                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{stats['total_files']}")

            except Exception as e:
                logger.error(f"Failed to process file {json_file}: {e}")
                stats["errors"].append({"file": str(json_file), "error": str(e)})

        logger.info(f"Batch indexing completed: {stats}")
        return stats

    def get_stats(self) -> dict[str, Any]:
        """
        Get experience store statistics

        Returns:
            Statistics dictionary
        """
        return {
            "collection_name": self.collection_name,
            "total_records": self.collection.count(),
            "db_path": str(self.db_path),
        }

    def get_cwe_stats(self) -> dict[str, int]:
        """
        Get experience count statistics for each CWE type

        Returns:
            CWE statistics dictionary, key is CWE ID, value is experience count
        """
        # Get metadata of all records
        all_records = self.collection.get(include=["metadatas"])

        cwe_counts = {}
        if all_records and all_records["metadatas"]:
            for metadata in all_records["metadatas"]:
                cwe_id = metadata.get("cwe_id", "UNKNOWN")
                cwe_counts[cwe_id] = cwe_counts.get(cwe_id, 0) + 1

        # Sort by count in descending order
        sorted_counts = dict(sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True))
        return sorted_counts

    def delete_by_cve(self, cve_id: str) -> int:
        """
        Delete all experience records for a specified CVE

        Args:
            cve_id: CVE ID

        Returns:
            Number of deleted records
        """
        # Query all records for this CVE
        results = self.collection.get(where={"cve_id": cve_id})

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} records (CVE: {cve_id})")
            return len(results["ids"])

        return 0

    def clear(self):
        """
        Clear all experience records (use with caution)
        """
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"description": "CWE-level vulnerability repair experience store"}
        )
        logger.warning("Experience store has been cleared")
