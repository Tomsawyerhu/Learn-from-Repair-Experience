#!/usr/bin/env python3
"""
Run Statistics Module for patcheval_exp.py

Collect and manage per-run statistics, including experience retrieval details for later analysis.
"""

import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RetrievedExperience:
    """Single retrieved experience record."""
    cve_id: str
    similarity_score: float
    quality_score: int
    repair_result: str
    programming_language: str = ""  # Added: programming language of the experience
    cwe_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class CVEStatistics:
    """Statistics for a single CVE."""
    cve_id: str
    programing_language: str
    retrieved_experiences: list[RetrievedExperience] = field(default_factory=list)
    used_self_experience: bool = False
    success_exp_count: int = 0
    failure_exp_count: int = 0
    cross_language_count: int = 0
    temporal_filtered_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "cve_id": self.cve_id,
            "programing_language": self.programing_language,
            "used_self_experience": self.used_self_experience,
            "success_exp_count": self.success_exp_count,
            "failure_exp_count": self.failure_exp_count,
            "cross_language_count": self.cross_language_count,
            "temporal_filtered_count": self.temporal_filtered_count,
            "retrieved_experiences": [asdict(exp) for exp in self.retrieved_experiences]
        }


class RunStatistics:
    """Run statistics manager (thread-safe)."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._cve_records: list[CVEStatistics] = []
    
    def add_cve_record(self, record: CVEStatistics) -> None:
        """Add one CVE statistics record."""
        with self._lock:
            self._cve_records.append(record)
    
    def compute_summary(self) -> dict:
        """Compute summary statistics."""
        if not self._cve_records:
            return {
                "total_cves": 0,
                "avg_success_exp": 0.0,
                "avg_failure_exp": 0.0,
                "avg_cross_language_exp": 0.0,
                "avg_quality_score": 0.0,
                "avg_similarity_score": 0.0
            }
        
        n = len(self._cve_records)
        total_success = sum(r.success_exp_count for r in self._cve_records)
        total_failure = sum(r.failure_exp_count for r in self._cve_records)
        total_cross_lang = sum(r.cross_language_count for r in self._cve_records)
        
        # Collect quality scores and similarity scores for all experiences.
        all_quality_scores = []
        all_similarity_scores = []
        for record in self._cve_records:
            for exp in record.retrieved_experiences:
                all_quality_scores.append(exp.quality_score)
                all_similarity_scores.append(exp.similarity_score)
        
        avg_quality = sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0
        avg_similarity = sum(all_similarity_scores) / len(all_similarity_scores) if all_similarity_scores else 0.0
        
        return {
            "total_cves": n,
            "avg_success_exp": round(total_success / n, 3),
            "avg_failure_exp": round(total_failure / n, 3),
            "avg_cross_language_exp": round(total_cross_lang / n, 3),
            "avg_quality_score": round(avg_quality, 3),
            "avg_similarity_score": round(avg_similarity, 4)
        }
    
    def save(self, output_path: Path) -> Path:
        """Save statistics to a JSON file.
        
        Args:
            output_path: Output directory path
            
        Returns:
            Path to the saved file
        """
        stats_file = output_path / "run_statistics.json"
        
        data = {
            "summary": self.compute_summary(),
            "cve_records": [r.to_dict() for r in self._cve_records]
        }
        
        stats_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return stats_file
    
    def __len__(self) -> int:
        return len(self._cve_records)


def create_cve_statistics(
    cve_id: str,
    programing_language: str,
    experiences: list[dict],
    used_self_experience: bool = False,
    temporal_filtered_count: int = 0
) -> CVEStatistics:
    """Create a CVE statistics record from an experience list.
    
    Args:
        cve_id: Current CVE ID
        programing_language: Programming language of the current CVE
        experiences: Retrieved experience list in dictionary form
        used_self_experience: Whether self experience was used
        temporal_filtered_count: Number of experiences excluded by temporal filtering
        
    Returns:
        A CVEStatistics instance
    """
    retrieved_experiences = []
    success_count = 0
    failure_count = 0
    cross_lang_count = 0
    
    for exp in experiences:
        exp_lang = exp.get("programing_language", "")
        exp_result = exp.get("repair_result", "")
        
        # Count success/failure outcomes.
        if exp_result == "success":
            success_count += 1
        elif exp_result == "failure":
            failure_count += 1
        
        # Count cross-language experiences.
        if exp_lang and programing_language and exp_lang != programing_language:
            cross_lang_count += 1
        
        # Build retrieved-experience records.
        retrieved_experiences.append(RetrievedExperience(
            cve_id=exp.get("cve_id", ""),
            similarity_score=float(exp.get("similarity_score", 0)),
            quality_score=int(exp.get("quality_score", 0)),
            repair_result=exp_result,
            programming_language=exp.get("programming_language", ""),  # Added: language from retrieval results
            cwe_ids=exp.get("cwe_ids", []),
            tags=exp.get("tags", [])
        ))
    
    return CVEStatistics(
        cve_id=cve_id,
        programing_language=programing_language,
        retrieved_experiences=retrieved_experiences,
        used_self_experience=used_self_experience,
        success_exp_count=success_count,
        failure_exp_count=failure_count,
        cross_language_count=cross_lang_count,
        temporal_filtered_count=temporal_filtered_count
    )
