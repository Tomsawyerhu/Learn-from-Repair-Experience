"""
Experience Retrieval Module

Provides semantic similarity-based experience retrieval, supports:
- CWE type filtering
- CVE exclusion (avoid retrieving experiences for the same CVE)
- Top-K similarity matching
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperienceRetriever:
    """
    Experience Retriever

    Performs semantic similarity retrieval based on ChromaDB.

    Attributes:
        store: ExperienceStore instance
        embedder: Vector encoder instance

    Example:
        >>> # Method 1: Auto initialize
        >>> retriever = ExperienceRetriever(db_path="vec_DB", model_path="models/bge")
        >>>
        >>> # Method 2: Manual injection
        >>> retriever = ExperienceRetriever(store=existing_store)
        >>>
        >>> # Retrieve
        >>> results = retriever.retrieve("SQL injection...", k=3)
    """

    def __init__(
        self,
        store=None,  # Type hint to avoid circular imports
        embedder=None,
        db_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the experience retriever

        Args:
            store: ExperienceStore instance (if provided, db_path/model_path are ignored)
            embedder: Vector encoder (if store is provided without embedder, try store.embedder)
            db_path: Vector database path (used when store=None)
            model_path: BGE model path (used when store=None)
        """
        if store:
            self.store = store
            self.embedder = embedder or getattr(store, "embedder", None)
            if not self.embedder and model_path:
                # Try to attach an embedder to the store
                from src.rag.embedder import BGEEmbedder

                self.embedder = BGEEmbedder(model_path=model_path)
        else:
            # Auto-initialization mode
            db_path = db_path or "vec_DB"
            model_path = model_path or "models/bge-large-en-v1.5"

            logger.info(f"Auto-initializing retriever: db={db_path}, model={model_path}")
            from src.rag.embedder import BGEEmbedder
            from src.rag.experience_store import ExperienceStore

            self.embedder = BGEEmbedder(model_path=model_path)
            self.store = ExperienceStore(db_path=db_path, embedder=self.embedder)

        if not self.embedder:
            logger.warning("No embedder found when initializing ExperienceRetriever; some features may be limited")

        self.collection = self.store.collection

    def retrieve(
        self,
        cve_description: str,
        cwe_ids: Optional[List[str]] = None,
        exclude_cve_id: Optional[str] = None,
        k: int = 5,
        min_similarity_score: float = 0.0,
        min_quality_score: int = 0,
        retrieve_top_k: Optional[int] = None,
        sort_by_quality: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Core retrieval method

        Args:
            cve_description: Query text
            cwe_ids: CWE ID filter list (applied after retrieval)
            exclude_cve_id: CVE ID to exclude
            k: Final number of experiences to return
            min_similarity_score: Semantic similarity threshold (0.0-1.0)
            min_quality_score: Experience quality threshold (1-5, 0 means no filtering)
            retrieve_top_k: Number of candidates to retrieve first (for two-stage retrieval). If set, first retrieve
                           the top retrieve_top_k most similar entries, then sort by quality score and return k entries
            sort_by_quality: Whether to sort by quality score descending (default is similarity)

        Returns:
            Experience list including similarity_score and quality_score
        """
        # Build the query embedding
        query_embedding = self.embedder.encode_single(cve_description)

        # Build filter conditions (quality_score is filtered at the database layer)
        where_filter = self._build_where_filter(min_quality_score)

        # Determine the retrieval count
        # If two-stage retrieval is enabled, use retrieve_top_k; otherwise keep the original logic
        if retrieve_top_k is not None and retrieve_top_k > 0:
            n_results = retrieve_top_k
        else:
            n_results = k * 3 if (exclude_cve_id or cwe_ids) else k

        # Execute the query
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

        # Process results without limiting count, then sort and slice later
        process_k = n_results if (retrieve_top_k or sort_by_quality) else k
        experiences = self._process_results(results, exclude_cve_id, cwe_ids, min_similarity_score, process_k)

        # Sort by quality score
        if sort_by_quality and experiences:
            experiences.sort(key=lambda x: (x.get("quality_score", 0), x.get("similarity_score", 0)), reverse=True)

        # Slice to the final return count
        return experiences[:k]

    def retrieve_simple(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Simplified retrieval that returns core fields

        Args:
            query: Query text
            k: Count

        Returns:
            [{\"id\": \"...\", \"experience\": \"...\", \"similarity_score\": 0.85}, ...]
        """
        results = self.retrieve(query, k=k)
        unique_results = []

        # Deduplicate based on experience content
        seen_content = set()

        for res in results:
            exp_content = res.get("experience", "")
            content_str = str(exp_content)

            if content_str in seen_content:
                continue

            seen_content.add(content_str)

            unique_results.append(
                {
                    "id": res["id"],
                    "cve_id": res["cve_id"],
                    "cwe_ids": res["cwe_ids"],
                    "similarity_score": res["similarity_score"],
                    "quality_score": res["quality_score"],
                    "experience": exp_content,
                }
            )

        return unique_results

    def retrieve_formatted(
        self,
        query: str,
        k: int = 3,
        template: str = "[{cve_id}] (similarity: {similarity_score:.2f}, quality: {quality_score}/5)\n{content}\n",
    ) -> str:
        """
        Retrieve and return a formatted string suitable for LLM prompt injection

        Args:
            query: Query text
            k: Count
            template: Formatting template

        Returns:
            Combined string
        """
        results = self.retrieve(query, k=k)
        formatted_texts = []

        for res in results:
            exp_content = res.get("experience", "")
            # Truncate overly long content
            content = exp_content if len(exp_content) <= 1000 else exp_content[:1000] + "..."

            text = template.format(
                id=res["id"],
                cve_id=res["cve_id"],
                cwe_ids=res["cwe_ids"],
                similarity_score=res["similarity_score"],
                quality_score=res["quality_score"],
                content=content,
            )
            formatted_texts.append(text)

        return "\n".join(formatted_texts)

    def _build_where_filter(self, min_quality_score: int = 0) -> Optional[Dict]:
        """
        Build the ChromaDB where filter

        Args:
            min_quality_score: Minimum experience quality score

        Returns:
            ChromaDB where-condition dictionary
        """
        conditions = []

        # Filter by experience quality score
        if min_quality_score > 0:
            conditions.append({"quality_score": {"$gte": min_quality_score}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _process_results(
        self,
        results: Dict,
        exclude_cve_id: Optional[str],
        cwe_ids: Optional[List[str]],
        min_similarity_score: float,
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Process retrieval results

        Args:
            results: ChromaDB query results
            exclude_cve_id: CVE ID to exclude
            cwe_ids: CWE ID filter list
            min_similarity_score: Minimum similarity threshold
            k: Maximum number of results to return

        Returns:
            Processed experience list including similarity_score and quality_score
        """
        experiences = []

        if not results["ids"] or not results["ids"][0]:
            return experiences

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (record_id, metadata, distance) in enumerate(zip(ids, metadatas, distances)):
            # ChromaDB returns distances; convert them to similarity scores
            # For cosine distance: score = 1 - distance / 2 (distance range 0-2)
            similarity_score = max(0, 1 - distance / 2)

            # Filter out low-similarity results
            if similarity_score < min_similarity_score:
                continue

            # Exclude the specified CVE
            if exclude_cve_id and metadata.get("cve_id") == exclude_cve_id:
                continue

            # Filter by CWE by checking whether the cwe_ids JSON string contains the target CWE
            if cwe_ids:
                record_cwe_ids_str = metadata.get("cwe_ids", "[]")
                try:
                    record_cwe_ids = json.loads(record_cwe_ids_str)
                except json.JSONDecodeError:
                    record_cwe_ids = []
                # Check whether there is any intersection
                if not any(cwe in record_cwe_ids for cwe in cwe_ids):
                    continue

            # Parse the tags field
            tags_str = metadata.get("tags", "[]")
            try:
                tags = json.loads(tags_str)
            except json.JSONDecodeError:
                tags = []

            # Parse the cwe_ids field
            cwe_ids_str = metadata.get("cwe_ids", "[]")
            try:
                parsed_cwe_ids = json.loads(cwe_ids_str)
            except json.JSONDecodeError:
                parsed_cwe_ids = []

            experiences.append(
                {
                    "id": record_id,
                    "cve_id": metadata.get("cve_id", ""),
                    "cwe_ids": parsed_cwe_ids,
                    "tags": tags,
                    "repair_result": metadata.get("repair_result", ""),
                    "programming_language": metadata.get("programming_language", ""),  # Added field
                    "experience": metadata.get("experience", ""),
                    "similarity_score": round(similarity_score, 4),
                    "quality_score": metadata.get("quality_score", 0),
                }
            )

            # Stop after collecting k results
            if len(experiences) >= k:
                break

        return experiences

    def retrieve_by_cwe(self, cwe_id: str, k: int = 10, min_quality_score: int = 0) -> List[Dict[str, Any]]:
        """
        Get all experiences containing the specified CWE without semantic similarity retrieval

        Note: because cwe_ids is stored as a JSON string, this method uses post-filtering

        Args:
            cwe_id: CWE ID
            k: Maximum number of results to return
            min_quality_score: Minimum experience quality score

        Returns:
            Experience list
        """
        # Build filter conditions
        where_filter = None
        if min_quality_score > 0:
            where_filter = {"quality_score": {"$gte": min_quality_score}}

        # Retrieve additional results for post-filtering
        results = self.collection.get(
            where=where_filter,
            limit=k * 5,  # Retrieve more entries to support post-filtering
            include=["metadatas"],
        )

        experiences = []
        if results["ids"]:
            for record_id, metadata in zip(results["ids"], results["metadatas"]):
                # Post-filter: check whether cwe_ids contains the target CWE
                cwe_ids_str = metadata.get("cwe_ids", "[]")
                try:
                    record_cwe_ids = json.loads(cwe_ids_str)
                except json.JSONDecodeError:
                    record_cwe_ids = []

                if cwe_id not in record_cwe_ids:
                    continue

                # Parse tags
                tags_str = metadata.get("tags", "[]")
                try:
                    tags = json.loads(tags_str)
                except json.JSONDecodeError:
                    tags = []

                experiences.append(
                    {
                        "id": record_id,
                        "cve_id": metadata.get("cve_id", ""),
                        "cwe_ids": record_cwe_ids,
                        "tags": tags,
                        "repair_result": metadata.get("repair_result", "unknown"),
                        "quality_score": metadata.get("quality_score", 0),
                        "experience": metadata.get("experience", ""),
                    }
                )

                if len(experiences) >= k:
                    break

        return experiences


def retrieve_similar_experiences(
    store, cve_description: str, cwe_ids: Optional[List[str]] = None, exclude_cve_id: Optional[str] = None, k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function: retrieve similar experiences

    Args:
        store: ExperienceStore instance
        cve_description: Query CVE description
        cwe_ids: CWE filter list
        exclude_cve_id: CVE ID to exclude
        k: Number of results to return

    Returns:
        Experience list
    """
    retriever = ExperienceRetriever(store)
    return retriever.retrieve(cve_description=cve_description, cwe_ids=cwe_ids, exclude_cve_id=exclude_cve_id, k=k)
