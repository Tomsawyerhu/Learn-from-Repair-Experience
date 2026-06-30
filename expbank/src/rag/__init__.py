"""
Experience RAG module.

This module provides ChromaDB-based experience storage and retrieval for
experience augmentation within the RAG pipeline.

Main components:
- BGEEmbedder: Uses BGE-large-en-v1.5 for text embedding
- ExperienceStore: ChromaDB-backed experience storage with indexing and persistence
- ExperienceRetriever: Experience retrieval with CWE filtering and semantic similarity matching
"""

from src.rag.embedder import BGEEmbedder
from src.rag.experience_store import ExperienceStore
from src.rag.experience_retriever import ExperienceRetriever

__all__ = [
    "BGEEmbedder",
    "ExperienceStore",
    "ExperienceRetriever",
]
