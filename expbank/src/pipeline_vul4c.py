#!/usr/bin/env python3
"""
ExpBank Vul4C pipeline.

This module reuses the base ExpBank Pipeline implementation and only adapts:
1) Success-set loading (from Vul4C evaluator outputs),
2) Vulnerability metadata loading (from SoK/Vul4C_Src/Vul4C.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.pipeline import Pipeline, logger

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPBANK_ROOT = Path(__file__).resolve().parents[1]


class Vul4CPipeline(Pipeline):
    """Pipeline adapter for Vul4C data/evaluation formats."""

    def __init__(
        self,
        config_path: str = "src/config/run/pipeline_config_vul4c.yaml",
        input_dir: str | None = None,
        output_dir: str | None = None,
        summary_path: str | None = None,
        vul_info_path: str | None = None,
        db_dir: str | None = None,
        results_path: str | None = None,
    ):
        super().__init__(
            config_path=config_path,
            input_dir=input_dir,
            output_dir=output_dir,
            summary_path=summary_path,
            vul_info_path=vul_info_path,
            db_dir=db_dir,
        )
        if results_path:
            self.config.setdefault("input", {})["results_path"] = results_path

    def _setup_turn_directory(self, base_dir: Path) -> Path:
        """
        Use a flat output layout for Vul4C.

        Multi-turn Vul4C runners already pass per-turn output directories
        (for example: ``turn_k/expbank``), so we should not auto-create
        nested ``turn_1/turn_2`` directories here.
        """
        turn_dir = Path(base_dir)
        turn_dir.mkdir(parents=True, exist_ok=True)

        # Keep stage subdirectories compatible with the base pipeline.
        (turn_dir / "extracted").mkdir(parents=True, exist_ok=True)
        (turn_dir / "exp_json").mkdir(parents=True, exist_ok=True)
        (turn_dir / "vec_DB").mkdir(parents=True, exist_ok=True)

        logger.info(f"Created run directory (Vul4C flat): {turn_dir}")
        return turn_dir

    @staticmethod
    def _unique_paths(paths: list[Path]) -> list[Path]:
        seen: set[str] = set()
        uniq: list[Path] = []
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(path)
        return uniq

    @staticmethod
    def _resolve_existing_path(path_value: str) -> Path | None:
        raw_path = Path(path_value).expanduser()
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([Path.cwd() / raw_path, EXPBANK_ROOT / raw_path, REPO_ROOT / raw_path, raw_path])

        for candidate in Vul4CPipeline._unique_paths(candidates):
            if candidate.exists():
                return candidate
        return None

    def _get_vulnerability_info_path(self) -> Path | None:
        """Resolve Vul4C vulnerability_info path in a portable way."""
        input_cfg = self.config.get("input", {})
        raw_traj_dir = Path(input_cfg.get("raw_traj_dir", "."))
        relative_path = Path(input_cfg.get("vulnerability_info_relative_path", "SoK/Vul4C_Src/Vul4C.json"))

        candidates: list[Path] = []

        configured_path = input_cfg.get("vulnerability_info")
        if configured_path:
            resolved = self._resolve_existing_path(str(configured_path))
            if resolved:
                return resolved
            raw_configured = Path(str(configured_path)).expanduser()
            if raw_configured.is_absolute():
                candidates.append(raw_configured)
            else:
                candidates.extend(
                    [Path.cwd() / raw_configured, EXPBANK_ROOT / raw_configured, REPO_ROOT / raw_configured]
                )

        # Keep compatibility with base pipeline inference rules + repo-root lookup.
        candidates.extend(
            [
                raw_traj_dir / relative_path,
                raw_traj_dir.parent / relative_path,
                Path.cwd() / relative_path,
                EXPBANK_ROOT / relative_path,
                REPO_ROOT / relative_path,
            ]
        )

        for candidate in self._unique_paths(candidates):
            if candidate.exists():
                return candidate

        logger.warning(f"Vulnerability info file not found, attempted paths: {len(self._unique_paths(candidates))}")
        return None

    def _get_results_path(self) -> Path | None:
        """Get evaluation/results.json path for Vul4C."""
        configured_path = self.config.get("input", {}).get("results_path")
        if configured_path:
            resolved = self._resolve_existing_path(str(configured_path))
            if resolved:
                return resolved
            return Path(configured_path)

        summary_path = self._get_summary_path()
        if summary_path:
            candidate = summary_path.parent / "results.json"
            if candidate.exists():
                return candidate

        raw_traj_dir = Path(self.config["input"]["raw_traj_dir"])
        candidates = [
            raw_traj_dir / "evaluation" / "results.json",
            raw_traj_dir.parent / "evaluation" / "results.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _normalize_cve_collection(value: Any) -> set[str]:
        """Normalize list/dict CVE collections into a flat CVE set."""
        cves: set[str] = set()
        if isinstance(value, list):
            cves.update(str(item) for item in value if item)
        elif isinstance(value, dict):
            for _, sub_value in value.items():
                if isinstance(sub_value, list):
                    cves.update(str(item) for item in sub_value if item)
        return cves

    def _get_success_set(self) -> set:
        """Load successful CVEs from Vul4C summary/results."""
        success_set: set[str] = set()

        summary_path = self._get_summary_path()
        if summary_path and summary_path.exists():
            logger.info(f"Read evaluation summary: {summary_path}")
            with open(summary_path, encoding="utf-8") as f:
                summary_data = json.load(f)

            # Preferred Vul4C fields in summary.json.
            success_set.update(self._normalize_cve_collection(summary_data.get("successful_cves")))
            success_set.update(self._normalize_cve_collection(summary_data.get("success_cves")))

            # Backward compatibility: old strict format.
            strict_eval = summary_data.get("strict_evaluation", {})
            success_set.update(self._normalize_cve_collection(strict_eval.get("successful_cves")))

            if success_set:
                logger.info(f"Successful repairs (summary): {len(success_set)} items")
                return success_set

        # Fallback to evaluation/results.json if summary has no explicit list.
        results_path = self._get_results_path()
        if not results_path or not results_path.exists():
            logger.warning(
                "No usable Vul4C evaluation results found (summary/results), will mark all trajectories as fail"
            )
            return set()

        logger.info(f"Read evaluation details: {results_path}")
        with open(results_path, encoding="utf-8") as f:
            results_data = json.load(f)

        if not isinstance(results_data, dict):
            logger.warning(f"Evaluation details format error: {results_path}")
            return set()

        for cve_id, item in results_data.items():
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")).strip().lower() == "success":
                success_set.add(str(cve_id))

        logger.info(f"Successful repairs (results): {len(success_set)} items")
        return success_set

    def _get_poc_only_success_cves(self, base_input_dir: Path) -> set:
        """Vul4C has no poc_only_evaluation split; no skip set is needed."""
        _ = base_input_dir
        return set()

    @staticmethod
    def _build_cwe_info(item: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Map Vul4C cwe/cwe_details into ExpBank cwe_info schema."""
        existing = item.get("cwe_info")
        if isinstance(existing, dict) and existing:
            return existing

        cwe_info: dict[str, dict[str, str]] = {}

        cwe_details = item.get("cwe_details")
        if isinstance(cwe_details, list):
            for detail in cwe_details:
                if not isinstance(detail, dict):
                    continue
                cwe_id = str(detail.get("id", "")).strip()
                if not cwe_id:
                    continue
                cwe_info[cwe_id] = {
                    "name": str(detail.get("name", "") or ""),
                    "description": str(detail.get("description", "") or ""),
                }

        if cwe_info:
            return cwe_info

        cwe_ids = item.get("cwe")
        if isinstance(cwe_ids, str):
            cwe_ids = [cwe_ids]
        if isinstance(cwe_ids, list):
            for cwe_id in cwe_ids:
                cwe_key = str(cwe_id).strip()
                if cwe_key:
                    cwe_info[cwe_key] = {"name": "", "description": ""}

        return cwe_info

    @staticmethod
    def _build_vul_func(item: dict[str, Any]) -> list[dict[str, Any]]:
        """Keep Vul4C vul_location empty in experience prompts."""
        _ = item
        return []

    def _normalize_vul_item(self, cve_id: str, item: dict[str, Any]) -> dict[str, Any]:
        """Normalize a single Vul4C item into ExpBank vulnerability_info schema."""
        cwe_info = self._build_cwe_info(item)
        return {
            "cve_id": cve_id,
            "cve_description": item.get("cve_description") or item.get("nvd_info"),
            "cwe_info": cwe_info,
            "vul_func": self._build_vul_func(item),
            "fix_func": item.get("fix_func") if isinstance(item.get("fix_func"), list) else [],
            "programming_language": "C/C++",
        }

    def _get_vulnerability_info(self) -> dict[str, dict]:
        """Load vulnerability map from Vul4C JSON (dict keyed by CVE)."""
        vul_path = self._get_vulnerability_info_path()
        if not vul_path or not vul_path.exists():
            logger.warning("Vulnerability info file does not exist")
            return {}

        with open(vul_path, encoding="utf-8") as f:
            data = json.load(f)

        vul_map: dict[str, dict[str, Any]] = {}

        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                cve_id = str(value.get("cve_id") or key).strip()
                if not cve_id:
                    continue
                vul_map[cve_id] = self._normalize_vul_item(cve_id, value)
        elif isinstance(data, list):
            # Keep compatibility with list-based metadata files.
            for item in data:
                if not isinstance(item, dict):
                    continue
                cve_id = str(item.get("cve_id", "")).strip()
                if not cve_id:
                    continue
                vul_map[cve_id] = self._normalize_vul_item(cve_id, item)
        else:
            logger.warning(f"Vulnerability info format error: {vul_path}")
            return {}

        logger.info(f"Loaded Vul4C vulnerability info: {len(vul_map)} items")
        return vul_map

    def _init_experience_extractor(self, extracted_dir: Path, exp_json_dir: Path):
        """Initialize extractor with portable llm config path resolution."""
        from src.agent.experience_extractor_agent import ExperienceExtractor

        llm_cfg = str(self.config["llm"]["config_path"])
        resolved_llm_cfg = self._resolve_existing_path(llm_cfg)
        if resolved_llm_cfg is None:
            raise FileNotFoundError(f"LLM config file does not exist: {llm_cfg}")

        self._extractor = ExperienceExtractor(
            config_path=str(resolved_llm_cfg),
            extracted_dir=str(extracted_dir),
            output_dir=str(exp_json_dir),
            template_name=self.config["llm"]["template_name"],
            client_type=self.config["llm"].get("client_type", "openai"),
        )
        return self._extractor


def main():
    """CLI entrypoint for Vul4C pipeline."""
    parser = argparse.ArgumentParser(
        description="ExpBank Vul4C Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        default="src/config/run/pipeline_config_vul4c.yaml",
        help="Config file path (default: src/config/run/pipeline_config_vul4c.yaml)",
    )
    parser.add_argument("--input-dir", "-i", default=None, help="Input directory (overrides raw_traj_dir)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (overrides base_dir)")
    parser.add_argument("--db-dir", default=None, help="Specify existing vector database directory")
    parser.add_argument("--summary-path", default=None, help="Evaluation summary path (overrides summary_path)")
    parser.add_argument("--results-path", default=None, help="Evaluation details path (overrides results_path)")
    parser.add_argument("--vul-info-path", default=None, help="Vulnerability info path (overrides vulnerability_info)")
    parser.add_argument(
        "--stages", "-s", default=None, help="Execution stages, comma-separated (extract,experience,index)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Only validate config, do not execute")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument("--clear-db", action="store_true", default=False, help="Clear vector database")

    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    pipeline = Vul4CPipeline(
        config_path=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        vul_info_path=args.vul_info_path,
        db_dir=args.db_dir,
        results_path=args.results_path,
    )
    stats = pipeline.run_all(
        stages=stages,
        dry_run=args.dry_run,
        max_files=args.max_files,
        workers=args.workers,
        clear_db=args.clear_db,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
